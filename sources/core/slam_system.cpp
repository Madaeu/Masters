//
// Created by madaeu on 5/5/21.
//

#include "slam_system.h"

namespace msc
{
    template<typename Scalar, int CS>
    SlamSystem<Scalar, CS>::SlamSystem()
    :forceKeyframe_(false), forceFrame_(false),
    bootstrapped_(false), currentKeyframe_(0)
    {}

    template<typename Scalar, int CS>
    SlamSystem<Scalar, CS>::~SlamSystem()
    {
        cuda::popContext();
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar,CS>::initializeSystem(CameraT &camera, SlamSystemOptions &options)
    {
        options_ = options;
        camera_ = camera;

        networkConfiguration_ = msc::loadJsonNetworkConfiguration(options_.networkPath);

        initializeGPU(options_.gpu);

        CameraT networkCamera(networkConfiguration_.camera.fy,
                              networkConfiguration_.camera.fy,
                              networkConfiguration_.camera.u0,
                              networkConfiguration_.camera.v0,
                              networkConfiguration_.inputWidth,
                              networkConfiguration_.inputHeight);

        cameraPyramid_ = CameraPyramidT(networkCamera, networkConfiguration_.pyramidLevels);

        typename CameraTrackerT::TrackerConfig trackerConfiguration;
        trackerConfiguration.pyramidLevels = networkConfiguration_.pyramidLevels;
        trackerConfiguration.iterationsPerLevel = options_.trackingIterations;
        trackerConfiguration.huberDelta = options_.trackingHuber;
        cameraTracker_ = std::make_shared<CameraTrackerT>(cameraPyramid_, trackerConfiguration);

        MapperOptions mapperOptions;
        mapperOptions.codePrior = options_.codePrior;
        mapperOptions.posePrior = options_.posePrior;
        mapperOptions.predictCode = options_.predictCode;

        mapperOptions.usePhotometric = options_.usePhotometric;
        mapperOptions.photometricIterations = options_.photometricIterations;
        mapperOptions.SFMParameters.huberDelta = options_.photometricHuber;
        mapperOptions.SFMParameters.avgDepth = networkConfiguration_.avgDepth;
        mapperOptions.SFMStepThreads = options_.sfmStepThreads;
        mapperOptions.SFMStepBlocks = options_.sfmStepBlocks;
        mapperOptions.SFMEvalThreads = options_.sfmEvalThreads;
        mapperOptions.SFMEvalBlocks = options_.sfmEvalBlocks;

        mapperOptions.useReprojection = options_.useReprojection;
        mapperOptions.reprojectionNFeatures = options_.reprojectionNFeatures;
        mapperOptions.reprojectionScaleFactor = options_.reprojectionScaleFactor;
        mapperOptions.reprojectionNLevels = options_.reprojectionNLevels;
        mapperOptions.reprojectionMaxDistance = options_.reprojectionMaxDistance;
        mapperOptions.reprojectionHuber = options_.reprojectionHuber;
        mapperOptions.reprojectionIterations = options_.reprojectionIterations;
        mapperOptions.reprojectionSigma = options_.reprojectionSigma;

        mapperOptions.connectionMode = options_.connectionMode;
        mapperOptions.maxBackConnections = options_.maxBackConnections;

        mapperOptions.iSAMParameters.enablePartialRelinearizationCheck = options_.partialRelinearizeCheck;
        mapperOptions.iSAMParameters.factorization = gtsam::ISAM2Params::CHOLESKY;
        mapperOptions.iSAMParameters.findUnusedFactorSlots = true;
        mapperOptions.iSAMParameters.relinearizeSkip = options_.relinearizeSkip;
        mapperOptions.iSAMParameters.relinearizeThreshold = options_.relinearizeThreshold;
        mapper_ = std::make_shared<MapperT>(mapperOptions, cameraPyramid_, network_);

        LoopDetectorConfiguration<Scalar> loopConfig;
        loopConfig.trackerConfiguration = trackerConfiguration;
        loopConfig.iterations = trackerConfiguration.iterationsPerLevel;
        loopConfig.activeWindows = options_.loopActiveWindows;
        loopConfig.maxDistance = options_.loopMaxDistance;
        loopConfig.minSimiliarity = options_.loopMinSimilarity;
        loopConfig.maxCandidates = options_.loopMaxCandidates;
        loopDetector_ = std::make_shared<LoopDetectorT>(options_.vocabularyPath, mapper_->getMap(),
                                                        loopConfig, cameraPyramid_);

        featureDetector_ = std::make_unique<ORBDetector>();

        liveImagePyramid_ = std::make_shared<ImagePyramidT>(networkConfiguration_.pyramidLevels, networkConfiguration_.inputWidth,
                                                            networkConfiguration_.inputHeight);
        liveGradientPyramid_ = std::make_shared<GradientPyramidT>(networkConfiguration_.pyramidLevels, networkConfiguration_.inputWidth,
                                                                  networkConfiguration_.inputHeight);

        se3Aligner_ = std::make_shared<SE3AlignerT>();

    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::resetSystem()
    {
        loopDetector_->reset();
        cameraTracker_->reset();
        mapper_->reset();
        notifyMapObservers();
        bootstrapped_ = false;
        trackingLost_ = false;
        forceKeyframe_ = false;
        forceFrame_ = false;
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::processFrame(double timestamp, const cv::Mat &frame)
    {
        if (!bootstrapped_)
        {
            throw std::runtime_error("Calling processFrame before system is bootstrapped!");
        }

        //TODO: timing stuff
        cv::Mat liveColorImage;
        Features features;
        cv::Mat liveFrame = preprocessImage(frame, liveColorImage, features);
        uploadLiveFrame(liveFrame);

        SE3T newPose = trackingLost_ ? relocalize() : trackFrame();

        trackingLost_ = checkTrackingLost(newPose);

        std::cout << (trackingLost_ ? "Lost \n" : "Tracking \n");
        if(trackingLost_)
        {
            return;
        }


        notifyPoseObservers();

        if (options_.loopClosure)
        {
            auto currentKeyframe = mapper_->getMap()->keyframes_.get(currentKeyframe_);
            int loopId = loopDetector_->detectLocalLoop(*liveImagePyramid_, *liveGradientPyramid_,
                                                        features, currentKeyframe,
                                                        currentPose_);
            if (loopId > 0)
            {
                if (mapper_->getMap()->keyframes_.linkExists(currentKeyframe_, loopId))
                {
                    mapper_->enqueueLink(currentKeyframe_, loopId, options_.loopSigma, true, false);
                }
            }

            auto loopInfo = loopDetector_->detectLoop(*liveImagePyramid_, *liveGradientPyramid_,
                                                      features, currentKeyframe,
                                                      currentPose_);
            if(loopInfo.detected)
            {
                if(!mapper_->getMap()->keyframes_.linkExists(currentKeyframe_, loopInfo.loopId))
                {
                    mapper_->enqueueLink(currentKeyframe_, loopInfo.loopId, options_.loopSigma, false, true);

                    loopLinks_.push_back({currentKeyframe_, loopInfo.loopId});
                }
            }
        }

        if(newKeyframeRequired())
        {
            auto keyframe = mapper_->enqueueKeyframe(timestamp, liveFrame, liveColorImage,
                                                     currentPose_, features);
            loopDetector_->addKeyframe(keyframe);

            notifyMapObservers();
            return;
        }

        if(newFrameRequired())
        {
            mapper_->enqueueFrame(liveFrame, liveColorImage, currentPose_, features, currentKeyframe_);
        }

        do
        {
            try
            {
                mapper_->mappingStep();
            }
            catch(std::exception &e)
            {
                //TODO debugging
                throw;
            }

        } while (mapper_->hasWork() && !options_.interleaveMapping);
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::bootstrapOneFrame(double timestamp, const cv::Mat &image)
    {
        resetSystem();

        Features features;
        cv::Mat colorImage;
        cv::Mat processedImage = preprocessImage(image, colorImage, features);
        mapper_->initOneFrame(timestamp, processedImage, colorImage, features);
        bootstrapped_ = true;

        currentKeyframe_ = mapper_->getMap()->keyframes_.lastID();
        cameraTracker_->setKeyframe(mapper_->getMap()->keyframes_.get(currentKeyframe_));
        cameraTracker_->reset();
        currentPose_ = cameraTracker_->getPoseEstimate();

        loopDetector_->addKeyframe(mapper_->getMap()->keyframes_.last());
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::bootstrapTwoFrames(double timestamp0, double timestamp1, const cv::Mat &image0,
                                                    const cv::Mat &image1)
    {
        resetSystem();

        Features features0, features1;
        cv::Mat colorImage0, colorImage1;
        cv::Mat processedImage0 = preprocessImage(image0, colorImage0, features0);
        cv::Mat processedImage1 = preprocessImage(image1, colorImage1, features1);

        mapper_->initTwoFrames(image0, image1,
                               colorImage0, colorImage1,
                               features0, features1,
                               timestamp0, timestamp1);
        bootstrapped_ = true;

        currentKeyframe_ = mapper_->getMap()->keyframes_.lastID();
        cameraTracker_->setKeyframe(mapper_->getMap()->keyframes_.get(currentKeyframe_));
        cameraTracker_->reset();
        currentPose_ = cameraTracker_->getPoseEstimate();

        auto keyframes = mapper_->getMap()->keyframes_;
        for( auto keyframe : keyframes)
        {
            loopDetector_->addKeyframe(keyframe.second);
        }
    }

    template<typename Scalar, int CS>
    typename SlamSystem<Scalar,CS>::CameraT
    SlamSystem<Scalar,CS>::getNetworkCamera()
    {
        return CameraT(networkConfiguration_.camera.fx,
                       networkConfiguration_.camera.fy,
                       networkConfiguration_.camera.u0,
                       networkConfiguration_.camera.v0,
                       networkConfiguration_.inputWidth,
                       networkConfiguration_.inputHeight);
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar,CS>::notifyMapObservers()
    {
        if(poseCallback_)
        {
            poseCallback_(currentPose_);
        }
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::notifyPoseObservers()
    {
        if(mapCallback_)
        {
            mapCallback_(mapper_->getMap());
        }
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::initializeGPU(std::size_t deviceId)
    {
        cuda::init();
        cuda::createAndBindContext(deviceId);

        {
            auto scopedPop = cuda::ScopedContextPop();
            network_ = std::make_shared<DecoderNetwork>(networkConfiguration_);
        }
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::uploadLiveFrame(const cv::Mat &frame)
    {
        for(std::size_t i = 0; i < networkConfiguration_.pyramidLevels; ++i)
        {
            if ( i == 0)
            {
                vc::Image2DView<Scalar, vc::TargetHost> temp(frame);
                (*liveImagePyramid_)[0].copyFrom(temp);
                continue;
            }
            gaussianBlurDown((*liveImagePyramid_)[i-1], (*liveImagePyramid_)[i]);
            sobelGradients((*liveImagePyramid_)[i], (*liveGradientPyramid_)[i]);
        }
    }

    template<typename Scalar, int CS>
    cv::Mat SlamSystem<Scalar,CS>::preprocessImage(const cv::Mat &frame, cv::Mat &colorImageOut, Features &features)
    {
        camera_.template resizeViewport(frame.cols, frame.rows);
        const cv::Mat cameraIntrinsics = (cv::Mat1d(3,3) << camera_.fx(), 0,camera_.u0(), 0, camera_.fy(), camera_.v0(), 0, 0, 1);
        const cv::Mat networkIntrinsics = (cv::Mat1d(3,3)<< networkConfiguration_.camera.fx, 0,networkConfiguration_.camera.u0, 0, networkConfiguration_.camera.fy, networkConfiguration_.camera.v0, 0, 0, 1);
        cv::Size networkSize(networkConfiguration_.inputWidth, networkConfiguration_.inputHeight);
        cv::initUndistortRectifyMap(cameraIntrinsics, cv::Mat{}, cv::Mat{}, networkIntrinsics, networkSize, CV_32FC1, map1_, map2_);

        cv::Mat outFrame;
        cv::remap(frame, outFrame, map1_, map2_, cv::INTER_LINEAR);
        colorImageOut = outFrame;

        cv::Mat outFrameGray;
        cv::cvtColor(outFrame, outFrameGray, cv::COLOR_RGB2GRAY);

        cv::Mat outFrameFloat;
        outFrameGray.convertTo(outFrameFloat, CV_32FC1, 1/255.0);

        features = featureDetector_->DetectAndCompute(outFrameGray);

        return outFrameFloat;

    }

    template<typename Scalar, int CS>
    typename SlamSystem<Scalar, CS>::SE3T
    SlamSystem<Scalar, CS>::trackFrame()
    {
        SE3T newPose;

        //TODO timing
        auto newKeyframeId = selectKeyframe();
        if(newKeyframeId != currentKeyframe_)
        {
            previousKeyframe_ = currentKeyframe_;
            currentKeyframe_ = newKeyframeId;
            cameraTracker_->setKeyframe(mapper_->getMap()->keyframes_.get(currentKeyframe_));
        }
        //TODO timing
        cameraTracker_->trackFrame(*liveImagePyramid_, *liveGradientPyramid_);
        newPose = cameraTracker_->getPoseEstimate();

        return newPose;
    }

    template<typename Scalar, int CS>
    typename SlamSystem<Scalar,CS>::SE3T
    SlamSystem<Scalar, CS>::relocalize()
    {
        SE3T newPose;
        auto keyframeMap = mapper_->getMap();
        Scalar bestError = std::numeric_limits<Scalar>::infinity();
        KeyframeId bestId = 1;
        SE3T bestPose = keyframeMap->keyframes_.get(bestId)->pose_;
        for (const auto& id : keyframeMap->keyframes_.ids())
        {
            auto keyframe = keyframeMap->keyframes_.get(id);
            cameraTracker_->setKeyframe(keyframe);
            cameraTracker_->reset();
            cameraTracker_->trackFrame(*liveImagePyramid_, *liveGradientPyramid_);
            auto error = cameraTracker_->getError();
            if(error < bestError)
            {
                bestError = error;
                bestId = id;
                bestPose = cameraTracker_->getPoseEstimate();
            }
        }

        currentKeyframe_ = bestId;
        newPose = bestPose;
        cameraTracker_->setPose(newPose);
        cameraTracker_->setKeyframe(keyframeMap->keyframes_.get(bestId));

        return newPose;
    }

    template<typename Scalar, int CS>
    bool SlamSystem<Scalar,CS>::newKeyframeRequired()
    {
        if(forceKeyframe_)
        {
            forceKeyframe_ = false;
            return true;
        }

        auto inliers = cameraTracker_->getInliers();
        // undersøg inliers
        auto currentKeyframe = mapper_->getMap()->keyframes_.get(currentKeyframe_);
        auto distance = poseDistance(currentKeyframe->pose_, currentPose_);

        switch (options_.keyframeMode)
        {
            case SlamSystemOptions::AUTO:
            {
                bool inlierBad = inliers < options_.inlierThreshold;
                bool distanceFar = distance > options_.distanceThreshold;
                return inlierBad || distanceFar;
            }
            case SlamSystemOptions::AUTO_COMBINED:
            {
                bool inlierBad = inliers < options_.inlierThreshold;
                float rotationDistance = (currentKeyframe->pose_.so3() * currentPose_.so3().inverse()).log().norm();
                float delta = distance * 5 + rotationDistance * 3;
                return delta > options_.combinedThreshold || inlierBad;
            }
            case SlamSystemOptions::NEVER:
                return false;
        }
        return false;
    }

    template<typename Scalar, int CS>
    bool SlamSystem<Scalar, CS>::newFrameRequired()
    {
        if (forceFrame_)
        {
            forceKeyframe_ = false;
            return true;
        }
        if (options_.keyframeMode == SlamSystemOptions::NEVER)
        {
            return false;
        }

        Scalar minFrameDistance = options_.frameDistanceThreshold;
        auto currentKeyframe = mapper_->getMap()->keyframes_.get(currentKeyframe_);
        Scalar keyframeDistance = poseDistance(currentKeyframe->pose_, currentPose_, 1.0f, 0.0f);
        bool farFromKeyframe = keyframeDistance > minFrameDistance;

        bool farFromFrames = true;
        for (auto frame : mapper_->getMap()->frames_)
        {
            Scalar frameDistance = poseDistance(frame.second->pose_, currentPose_, 1.0f, 0.0f);
            if ( frameDistance < minFrameDistance)
            {
                farFromFrames = false;
            }
        }

        return farFromFrames && farFromKeyframe && !mapper_->hasWork();
    }

    template<typename Scalar, int CS>
    typename SlamSystem<Scalar, CS>::KeyframeId
    SlamSystem<Scalar, CS>::selectKeyframe()
    {
        if (mapper_->numberOfKeyframes() == 0)
        {
            throw std::runtime_error("KeyframeMap is empty (should not happen at this stage)");
        }

        KeyframeId  keyframeId = 0;
        if(options_.trackingMode == SlamSystemOptions::LAST)
        {
            keyframeId = mapper_->getMap()->keyframes_.lastID();
        }
        else if (options_.trackingMode == SlamSystemOptions::CLOSEST)
        {
            auto keyframeMap = mapper_->getMap();
            Scalar closestDistance = std::numeric_limits<Scalar>::infinity();

            for(auto id : keyframeMap->keyframes_.ids())
            {
                Scalar distance = poseDistance(keyframeMap->keyframes_.get(id)->pose_, currentPose_);
                if(distance < closestDistance)
                {
                    closestDistance = distance;
                    keyframeId = id;
                }
            }
        }
        else if( options_.trackingMode == SlamSystemOptions::FIRST)
        {
            keyframeId = mapper_->getMap()->keyframes_.ids()[0];
        }
        else
        {
            throw std::runtime_error("Unhandled tracking mode");
        }

        return keyframeId;
    }

    template<typename Scalar, int CS>
    bool SlamSystem<Scalar, CS>::checkTrackingLost(const SE3T &pose)
    {
        bool errorTooBig = cameraTracker_->getError() > options_.trackingErrorThreshold;

        auto poseKeyframe = mapper_->getMap()->keyframes_.get(currentKeyframe_)->pose_;
        Scalar distance = poseDistance(poseKeyframe, pose);
        bool keyframeTooFar = distance > options_.trackingDistanceThreshold;

        bool trackingLost = errorTooBig || keyframeTooFar;

        //TODO logging

        return trackingLost;
    }

    template class SlamSystem<float, 32>;
} // namespace msc