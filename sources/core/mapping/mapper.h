//
// Created by madaeu on 3/5/21.
//

#ifndef MASTERS_MAPPER_H
#define MASTERS_MAPPER_H

#include "factor_graph.h"
#include "keyframe_map.h"
#include "cuda_image_proc.h"
#include "cuda_SE3_aligner.h"
#include "decoder_network.h"
#include "feature_detector.h"
#include "work_manager.h"
#include "work_implementation.h"
#include "cuda_context.h"

#include "sophus/se3.hpp"
#include "gtsam/nonlinear/ISAM2.h"
#include "gtsam/base/Vector.h"
#include "gtsam/nonlinear/LinearContainerFactor.h"
#include "opencv2/opencv.hpp"
#include "VisionCore/Image/BufferOps.hpp"

#include <memory>
#include <cuda_SE3_aligner.h>

namespace msc
{
    struct MapperOptions
    {
        enum ConnectionMode {FULL, LASTN, FIRST, LAST};

        double codePrior = 1;
        double posePrior = 0.3;
        bool predictCode = false;

        bool usePhotometric = true;
        std::vector<int> photometricIterations;
        msc::DenseSFMParameters SFMParameters;
        int SFMStepThreads = 32;
        int SFMStepBlocks = 11;
        int SFMEvalThreads = 66;
        int SFMEvalBlocks = 224;

        bool useReprojection = false;
        int reprojectionNFeatures = 500;
        float reprojectionScaleFactor = 1.2f;
        int reprojectionNLevels = 1;
        float reprojectionMaxDistance = 30.0f;
        float reprojectionHuber = 0.1f;
        int reprojectionIterations = 15;
        float reprojectionSigma = 1.0f;
        int reprojectionRansacIterations = 1000;
        float reprojectionRansacThreshold = 0.0001f;

        ConnectionMode connectionMode = LAST;
        int maxBackConnections = 4;

        gtsam::ISAM2Params iSAMParameters;
    };

    template<typename Scalar, int CS>
    class Mapper
    {
    public:
        using CodeT = gtsam::Vector;
        using SE3T = Sophus::SE3<Scalar>;
        using MapT = msc::Map<Scalar>;
        using CameraPyramidT = msc::CameraPyramid<Scalar>;
        using KeyframePtr = typename msc::Keyframe<Scalar>::Ptr;
        using KeyframeId = typename msc::Keyframe<Scalar>::IdType;
        using FramePtr = typename msc::Frame<Scalar>::Ptr;
        using SFMAlignerPtr = typename msc::SFMAligner<Scalar, CS>::Ptr;
        using SE3AlignerPtr = typename msc::SE3Aligner<Scalar>::Ptr;
        using FrameId = typename MapT::FrameID;
        using MapPtr = typename MapT::Ptr;
        using MapCallback = std::function<void(MapPtr)>;
        using IteratorList = std::vector<int>;

        Mapper(const MapperOptions& options,
               const CameraPyramidT& cameraPyramid,
               DecoderNetwork::Ptr network);

        void enqueueFrame(const cv::Mat& image, const cv::Mat colorImage,
                          const SE3T& initialPose, const Features& features,
                          FrameId keyframeId);

        KeyframePtr enqueueKeyframe(double timestamp, const cv::Mat& image,
                                    const cv::Mat& imageColor, const SE3T& initialPose,
                                    const Features& features);

        KeyframePtr enqueueKeyframe(double timestamp, const cv::Mat& image, const cv::Mat& colorImage,
                                    const SE3T initialPose, const Features& features,
                                    const std::vector<FrameId>& connections);

        void enqueueLink(FrameId id0, FrameId id1, Scalar reprojectionSigma,
                         bool photometric = true, bool reprojection = true);

        void marginalizeFrames(const std::vector<FrameId>& frames);

        void mappingStep();

        void initTwoFrames(const cv::Mat& image0, const cv::Mat& image1,
                           const cv::Mat& colorImage0, const cv::Mat& colorImage1,
                           const Features& features0, const Features& features1,
                           double timestamp0, double timestamp1);

        void initOneFrame(double timestamp, const cv::Mat& image,
                          const cv::Mat& colorImage, const Features& features);

        void reset();

        std::vector<FrameId> nonMarginalizedFrames();
        std::map<KeyframeId, bool> keyframeRelinearization();

        void setMapCallback(MapCallback callback) { mapCallback_ = callback; }
        MapPtr getMap() { return map_; }
        std::size_t numberOfKeyframes() const { return map_->numberOfKeyframes(); }
        bool hasWork() const { return !workManager_.empty(); }

    private:

        void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                         gtsam::FactorIndices& removeIndices,
                         gtsam::Values& initialValues);

        FramePtr buildFrame(const cv::Mat& image, const cv::Mat& colorImage,
                            const SE3T& initialPose, const Features& features);

        KeyframePtr buildKeyframe(double timestamp, const cv::Mat& image,
                                  const cv::Mat& colorImage, const SE3T& initialPose,
                                  const Features& features);

        std::vector<FrameId> buildBackConnections();

        void updateMap(const gtsam::Values& values, const gtsam::VectorValues& delta);

        void notifyMapObservers();

        gtsam::Values estimate_;
        MapPtr map_;
        std::unique_ptr<gtsam::ISAM2> graphISAM_;
        std::shared_ptr<DecoderNetwork> network_;
        CameraPyramidT cameraPyramid_;
        SE3AlignerPtr se3Aligner_;
        SFMAlignerPtr sfmAligner_;

        MapperOptions options_;
        MapCallback mapCallback_;

        work::WorkManager workManager_;

        gtsam::ISAM2Result resultsISAM_;
    };

    template<typename Scalar, int CS>
    Mapper<Scalar, CS>::Mapper(const MapperOptions &options,
                               const CameraPyramidT &cameraPyramid,
                               DecoderNetwork::Ptr network)
    : network_(network), cameraPyramid_(cameraPyramid), options_(options)
    {
        SFMAlignerParameters sfmParameters;
        sfmParameters.SFMParameters = options.SFMParameters;
        sfmParameters.step_threads = options.SFMStepThreads;
        sfmParameters.step_blocks = options.SFMStepBlocks;
        sfmParameters.eval_threads = options.SFMEvalThreads;
        sfmParameters.eval_blocks = options.SFMEvalBlocks;
        sfmAligner_ = std::make_shared<SFMAligner<Scalar, CS>>(sfmParameters);

        map_ = std::make_shared<MapT>();

        options_.iSAMParameters.enableDetailedResults = true;
        options_.iSAMParameters.print("ISAM2 parameters");
        graphISAM_ = std::make_unique<gtsam::ISAM2>(options_.iSAMParameters);

        se3Aligner_ = std::make_shared<SE3Aligner<Scalar>>();
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::initTwoFrames(const cv::Mat &image0, const cv::Mat &image1, const cv::Mat &colorImage0,
                                           const cv::Mat &colorImage1, const Features &features0,
                                           const Features &features1, double timestamp0, double timestamp1)
    {
        reset();
        auto keyframe0 = buildKeyframe(timestamp0, image0, colorImage0, SE3T{}, features0);
        auto keyframe1 = buildKeyframe(timestamp1, image1, colorImage1, SE3T{}, features1);
        map_->addKeyframe(keyframe0);
        map_->addKeyframe(keyframe1);

        workManager_.addWork<work::InitVariables<Scalar,CS>>(keyframe0, options_.codePrior, options_.posePrior);
        workManager_.addWork<work::InitVariables<Scalar,CS>>(keyframe1, options_.codePrior);

        workManager_.addWork<work::OptimizePhotometric<Scalar, CS>>(keyframe0, keyframe1, options_.photometricIterations,
                                                                    cameraPyramid_, sfmAligner_);
        workManager_.addWork<work::OptimizePhotometric<Scalar, CS>>(keyframe0, keyframe1, options_.photometricIterations,
                                                                    cameraPyramid_, sfmAligner_);

        while(!workManager_.empty())
        {
            mappingStep();
        }
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::initOneFrame(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                          const Features &features)
    {
        reset();

        auto keyframe = buildKeyframe(timestamp, image, colorImage, SE3T{}, features);
        map_->addKeyframe(keyframe);

        workManager_.addWork<work::InitVariables<Scalar,CS>>(keyframe, options_.codePrior, options_.posePrior);

        mappingStep();
    }

    template<typename Scalar, int CS>
    std::vector<typename Mapper<Scalar, CS>::FrameId>
    Mapper<Scalar,CS>::nonMarginalizedFrames()
    {
        std::vector<FrameId> nonMarginalizedFrames;
        for(auto& id : map_->frames_.ids()) {
            if (!map_->frames_.get(id)->marginalized_) {
                nonMarginalizedFrames.push_back(id);
            }
        }
        return nonMarginalizedFrames;
    }

    template<typename Scalar, int CS>
    std::map<typename Mapper<Scalar, CS>::KeyframeId, bool>
    Mapper<Scalar,CS>::keyframeRelinearization()
    {
        std::map<KeyframeId, bool> info;
        auto& variableStatus = (*resultsISAM_.detail).variableStatus;
        for (auto& id : map_->keyframes_.ids())
        {
            auto pKey = poseKey(id);
            auto cKey = codeKey(id);
            info[id] = variableStatus[pKey].isRelinearized || variableStatus[cKey].isRelinearized;
        }
        return info;
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar,CS>::enqueueFrame(const cv::Mat &image, const cv::Mat colorImage, const SE3T &initialPose,
                                         const Features &features, FrameId keyframeId)
    {
        marginalizeFrames(nonMarginalizedFrames());

        auto frame = buildFrame(image, colorImage, initialPose, features);
        map_->addFrame(frame);
        auto id = frame->id_;

        auto keyframe = map_->keyframes_.get(keyframeId);

        workManager_.addWork<work::InitVariables<Scalar,CS>>(frame);
        workManager_.addWork<work::OptimizePhotometric<Scalar,CS>>(keyframe, frame,
                                                                   options_.photometricIterations,
                                                                   cameraPyramid_, sfmAligner_);

        map_->frames_.addLink(frame->id_, keyframeId);
    }

    template<typename Scalar, int CS>
    typename Mapper<Scalar,CS>::KeyframePtr
    Mapper<Scalar, CS>::enqueueKeyframe(double timestamp, const cv::Mat &image, const cv::Mat &imageColor,
                                        const SE3T &initialPose, const Features &features)
    {
        auto connections = buildBackConnections();
        return enqueueKeyframe(timestamp, image, imageColor, initialPose, features, connections);
    }

    template<typename Scalar, int CS>
    typename Mapper<Scalar, CS>::KeyframePtr
    Mapper<Scalar,CS>::enqueueKeyframe(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                       const SE3T initialPose, const Features &features,
                                       const std::vector<FrameId> &connections)
    {
        auto keyframe = buildKeyframe(timestamp, image, colorImage, initialPose, features);
        map_->addKeyframe(keyframe);

        marginalizeFrames(nonMarginalizedFrames());

        workManager_.addWork<work::InitVariables<Scalar, CS>>(keyframe, options_.codePrior);
        for (auto& id : connections)
        {
            auto backKeyframe = map_->keyframes_.get(id);

            work::WorkManager::WorkPtr ptr;
            if(options_.usePhotometric)
            {
                workManager_.addWork<work::OptimizePhotometric<Scalar,CS>>(keyframe, backKeyframe,
                                                                           options_.photometricIterations,
                                                                           cameraPyramid_, sfmAligner_);
                ptr = workManager_.addWork<work::OptimizePhotometric<Scalar, CS>>(backKeyframe, keyframe,
                                                                                  options_.photometricIterations,
                                                                                  cameraPyramid_, sfmAligner_, true);
            }

            if (options_.useReprojection)
            {
                workManager_.addWork<work::OptimizeReprojection<Scalar,CS>>(keyframe, backKeyframe,
                                                                            options_.reprojectionIterations,
                                                                            cameraPyramid_[0],
                                                                            options_.reprojectionMaxDistance,
                                                                            options_.reprojectionHuber,
                                                                            options_.reprojectionSigma,
                                                                            options_.reprojectionRansacIterations,
                                                                            options_.reprojectionRansacThreshold);

                workManager_.addWork<work::OptimizeReprojection<Scalar,CS>>(backKeyframe, keyframe,
                                                                            options_.reprojectionIterations,
                                                                            cameraPyramid_[0],
                                                                            options_.reprojectionMaxDistance,
                                                                            options_.reprojectionHuber,
                                                                            options_.reprojectionSigma,
                                                                            options_.reprojectionRansacIterations,
                                                                            options_.reprojectionRansacThreshold);
            }

            // TODO: Add Geometric Factor

            map_->keyframes_.addLink(keyframe->id_, id);
            map_->keyframes_.addLink(id, keyframe->id_);
        }
        return keyframe;
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar,CS>::enqueueLink(FrameId id0, FrameId id1, Scalar reprojectionSigma, bool photometric,
                                        bool reprojection)
    {
        auto keyframe0 = map_->keyframes_.get(id0);
        auto keyframe1 = map_->keyframes_.get(id1);

        work::WorkManager::WorkPtr ptr;

        marginalizeFrames(nonMarginalizedFrames());

        if ( photometric)
        {
            workManager_.addWork<work::OptimizePhotometric<Scalar,CS>>(keyframe0, keyframe1,
                                                                       options_.photometricIterations,
                                                                       cameraPyramid_, sfmAligner_);

            ptr = workManager_.addWork<work::OptimizePhotometric<Scalar,CS>>(keyframe1, keyframe0,
                                                                             options_.photometricIterations,
                                                                             cameraPyramid_, sfmAligner_, true);
        }

        if (reprojection)
        {
            workManager_.addWork<work::OptimizeReprojection<Scalar,CS>>(keyframe0, keyframe1,
                                                                        options_.reprojectionIterations,
                                                                        cameraPyramid_[0],
                                                                        options_.reprojectionMaxDistance,
                                                                        options_.reprojectionHuber,
                                                                        options_.reprojectionSigma,
                                                                        options_.reprojectionRansacIterations,
                                                                        options_.reprojectionRansacThreshold);

            workManager_.addWork<work::OptimizeReprojection<Scalar,CS>>(keyframe1, keyframe0,
                                                                        options_.reprojectionIterations,
                                                                        cameraPyramid_[0],
                                                                        options_.reprojectionMaxDistance,
                                                                        options_.reprojectionHuber,
                                                                        options_.reprojectionSigma,
                                                                        options_.reprojectionRansacIterations,
                                                                        options_.reprojectionRansacThreshold);

        }
        // TODO: Add Geometric Factor

        map_->keyframes_.addLink(id0, id1);
        map_->keyframes_.addLink(id1, id0);
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::marginalizeFrames(const std::vector<FrameId> &frames)
    {
        if (frames.empty())
        {
            return;
        }

        gtsam::FastList<gtsam::Key> marginalizedKeys;
        for( const auto& id : frames)
        {
            auto frame = map_->frames_.get(id);
            if(!frame->marginalized_)
            {
                marginalizedKeys.push_back(auxPoseKey(id));
                frame->marginalized_ = true;

                auto lambda = [&] (work::Work::Ptr w) {
                    auto ow = std::dynamic_pointer_cast<work::OptimizeWork<Scalar>>(w);
                    if (!ow)
                    {
                        return false;
                    }
                    return ow && ow->involves(frame);
                };

                workManager_.erase(lambda);
            }
        }

        graphISAM_->marginalizeLeaves(marginalizedKeys);
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::bookkeeping(gtsam::NonlinearFactorGraph &newFactors, gtsam::FactorIndices &removeIndices,
                                         gtsam::Values &initialValues)
    {
        workManager_.bookkeeping(newFactors, removeIndices, initialValues);
        workManager_.update();
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::mappingStep()
    {
        if (workManager_.empty())
        {
            return;
        }

        gtsam::NonlinearFactorGraph newFactors;
        gtsam::FactorIndices removeIndices;
        gtsam::Values initialValues;

        bookkeeping(newFactors, removeIndices, initialValues);

        gtsam::FastMap<gtsam::Key, int> constrainedKeys;

        auto it = std::find_if(map_->frames_.begin(), map_->frames_.end(),
                               [] (auto& frame) -> bool {return !frame.second->marginalized_; } );
        bool hasMarginalizedFrames = it != map_->frames_.end();

        int i = 0;
        for(auto& frame : map_->keyframes_)
        {
            if(frame.second->id_ != map_->keyframes_.lastID())
            {
                auto id = frame.second->id_;
                constrainedKeys[poseKey(id)] = i;
                constrainedKeys[codeKey(id)] = i;
            }
        }
        i++;

        if (hasMarginalizedFrames)
        {
            for (auto& frame : map_->frames_)
            {
                if (frame.second->marginalized_)
                {
                    continue;
                }
                auto id = frame.second->id_;
                constrainedKeys[auxPoseKey(id)] = i;
            }
            i++;
        }

        auto lastId = map_->keyframes_.lastID();
        constrainedKeys[poseKey(lastId)] = i;
        constrainedKeys[codeKey(lastId)] = i;

        // TODO: timing
        resultsISAM_ = graphISAM_->update(newFactors, initialValues, removeIndices,
                                          constrainedKeys, boost::none, boost::none, true);

        // TODO: ISAM Logging
        workManager_.distributeIndices(resultsISAM_.newFactorsIndices);

        // TODO: More timing
        estimate_ = graphISAM_->calculateEstimate();
        updateMap(estimate_, graphISAM_->getDelta());

        if ( resultsISAM_.variablesRelinearized == 0)
        {
            workManager_.signalNoRelinearize();
        }

        // TODO: Debugging and display factors
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::reset()
    {
        map_->clear();
        estimate_.clear();
        graphISAM_ = std::make_unique<gtsam::ISAM2>(options_.iSAMParameters);
        workManager_.clear();
    }

    template <typename Scalar, int CS>
    void Mapper<Scalar, CS>::updateMap(const gtsam::Values &values, const gtsam::VectorValues &delta)
    {
        std::vector<FrameId> changedKeyframes;
        for (auto& id : map_->keyframes_.ids())
        {
            gtsam::Key pKey = poseKey(id);
            gtsam::Key cKey = codeKey(id);
            if(delta[pKey].norm() > 1e-5 || delta[cKey].norm() > 1e-5)
            {
                changedKeyframes.push_back(id);
            }
        }

        for(const auto& id : changedKeyframes)
        {
            auto keyframe = map_->keyframes_.get(id);
            keyframe->code_ = values.at(codeKey(id)).template cast<gtsam::Vector>().template cast<float>();
            keyframe->pose_ = values.at(poseKey(id)).template cast<Sophus::SE3f>();

            Eigen::Matrix<float, CS, 1> code = keyframe->code_;
            for (uint i = 0; i < keyframe->imagePyramid_.levels(); ++i)
            {
                updateDepth(code, keyframe->proximityPyramid_.getLevelGPU(i),
                            keyframe->jacobianPyramid_.getLevelGPU(i),
                            (float)options_.SFMParameters.avgDepth,
                            keyframe->depthPyramid_.getLevelGPU(i));
            }
        }
        for (const auto& id : map_->frames_.ids())
        {
            auto frame = map_->frames_.get(id);
            if( !frame->marginalized_)
            {
                frame->pose_ = values.at(auxPoseKey(id)).template cast<Sophus::SE3f>();
            }
        }

        notifyMapObservers();
    }

    template<typename Scalar, int CS>
    typename Mapper<Scalar, CS>::FramePtr
    Mapper<Scalar,CS>::buildFrame(const cv::Mat &image, const cv::Mat &colorImage, const SE3T &initialPose,
                                  const Features &features)
    {
        auto frame = std::make_shared<Frame<Scalar>>(cameraPyramid_.levels(), cameraPyramid_[0].width(),
                                             cameraPyramid_[0].height());
        frame->pose_ = initialPose;
        frame->colorImage_ = colorImage.clone();
        frame->fillPyramids(image, cameraPyramid_.levels());
        frame->features_ = features;
        frame->hasKeypoints_ = true;
        return frame;
    }

    template<typename Scalar, int CS>
    typename Mapper<Scalar, CS>::KeyframePtr
    Mapper<Scalar, CS>::buildKeyframe(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                      const SE3T &initialPose, const Features &features)
    {
        auto keyframe = std::make_shared<Keyframe<float>>(cameraPyramid_.levels(),
                                                          cameraPyramid_[0].width(),
                                                          cameraPyramid_[0].height(),
                                                          CS);
        keyframe->pose_ = initialPose;
        keyframe->colorImage_ = colorImage.clone();
        keyframe->timestamp_ = timestamp;

        for( uint i = 0; i < cameraPyramid_.levels(); ++i)
        {
            vc::image::fillBuffer(keyframe->validPyramid_.getLevelGPU(i), 1.0f);

            if (i == 0)
            {
                vc::Image2DView<float, vc::TargetHost> temp(image);
                keyframe->imagePyramid_.getLevelGPU(0).copyFrom(temp);
                sobelGradients(keyframe->imagePyramid_.getLevelGPU(0), keyframe->gradientPyramid_.getLevelGPU(0));
                continue;
            }
            gaussianBlurDown(keyframe->imagePyramid_.getLevelGPU(i-1), keyframe->imagePyramid_.getLevelGPU(i));
            sobelGradients(keyframe->imagePyramid_.getLevelGPU(i), keyframe->gradientPyramid_.getLevelGPU(i));
        }

        cv::Mat networkImage;
        cv::cvtColor(keyframe->colorImage_, networkImage, cv::COLOR_RGB2GRAY);
        networkImage.convertTo(networkImage, CV_32FC1, 1/255.0);
        vc::Image2DView<float, vc::TargetHost> networkImageView(networkImage);

        auto proximityPtr = keyframe->proximityPyramid_.getCPUMutable();
        auto jacobianPtr = keyframe->jacobianPyramid_.getCPUMutable();
        auto uncertaintyPtr = keyframe->uncertaintyPyramid_.getCPUMutable();
        // TODO: timing
        cuda::ScopedContextPop pop;
        const Eigen::VectorXf zeroCode = Eigen::VectorXf::Zero(CS);

        if(options_.predictCode)
        {
            Eigen::MatrixXf predictedCode(CS,1);
            network_->predictAndDecode(networkImageView, zeroCode, &predictedCode,
                                       proximityPtr.get(), uncertaintyPtr.get(),
                                       jacobianPtr.get());
            keyframe->code_ = predictedCode;
        }
        else
        {
            network_->decode(networkImageView, zeroCode, proximityPtr.get(),
                             uncertaintyPtr.get(), jacobianPtr.get());
            keyframe->code_ = zeroCode;
        }

        for( uint i = 0; i < cameraPyramid_.levels(); ++i)
        {
            updateDepth((Eigen::Matrix<Scalar, CS, 1>)keyframe->code_,
                        keyframe->proximityPyramid_.getLevelGPU(i),
                        keyframe->jacobianPyramid_.getLevelGPU(i),
                        options_.SFMParameters.avgDepth,
                        keyframe->depthPyramid_.getLevelGPU(i));
        }

        // TODO: Add Geometric Factor

        keyframe->features_ = features;
        keyframe->hasKeypoints_ = true;

        return keyframe;
    }

    template<typename Scalar, int CS>
    std::vector<typename Mapper<Scalar,CS>::FrameId>
    Mapper<Scalar, CS>::buildBackConnections()
    {
        int start = map_->keyframes_.lastID();
        int stop = 1;
        switch (options_.connectionMode)
        {
            case MapperOptions::LASTN:
                stop = std::max(1, start-options_.maxBackConnections + 1);
                break;
            case MapperOptions::FULL:
                stop = 1;
                break;
            case MapperOptions::FIRST:
                stop = start = 1;
                break;
            case MapperOptions::LAST:
                start = stop = map_->keyframes_.lastID();
                break;
        }
        std::vector<FrameId> connections;
        for (int i = start; i >= stop; i--)
        {
            connections.push_back(i);
        }
        return connections;
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::notifyMapObservers()
    {
        if (mapCallback_)
            mapCallback_(map_);
    }

} // namespace msc



#endif //MASTERS_MAPPER_H
