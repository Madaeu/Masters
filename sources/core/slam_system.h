//
// Created by madaeu on 2/15/21.
//

#ifndef MASTERS_SLAM_SYSTEM_H
#define MASTERS_SLAM_SYSTEM_H

#include "pinhole_camera.h"
#include "camera_pyramid.h"
#include "feature_detector.h"
#include "mapper.h"
#include "keyframe_map.h"
#include "camera_tracker.h"
#include "cuda_image_proc.h"

#include "VisionCore/Buffers/BufferPyramid.hpp"
#include "VisionCore/Buffers/Image2D.hpp"
#include <iostream>
#include <memory>

namespace msc {
    template<typename Scalar, int CS>
    class SlamSystem {
    public:
        using ScalarT = Scalar;
        using SE3T = Sophus::SE3<Scalar>;
        using KeyframeT = Keyframe<Scalar>;
        using KeyframeID = typename KeyframeT::IdType;
        using MapperT = Mapper<Scalar, CS>;
        using TrackerT = CameraTracker<Scalar>;
        using ImagePyramidT = vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA>;
        using GradientPyramidT = vc::RuntimeBufferPyramidManaged<Eigen::Matrix<Scalar, 1, 2>, vc::TargetDeviceCUDA>;


        SlamSystem() = default;

        virtual ~SlamSystem();

        void initializeSystem(msc::PinholeCamera<Scalar> &cam);

        void reset();

        void processFrame(double timestamp, const cv::Mat &frame);

        void bootstrapOneFrame(double timestamp, const cv::Mat &frame);

        void bootstrapTwoFrames(double timestamp0, double timestamp1, const cv::Mat &image0, const cv::Mat &image1);


    private:
        //Functions
        cv::Mat preprocessImage(const cv::Mat &frame, cv::Mat &out_color, msc::Features &features);

        void uploadLiveFrame(const cv::Mat liveFrame);

        SE3T trackFrame();

        KeyframeID selectKeyframe();

        //Variables
        msc::PinholeCamera<Scalar> origCam_;
        msc::CameraPyramid<Scalar> camera_pyr_;

        std::unique_ptr<TrackerT> tracker_;
        std::unique_ptr<MapperT> mapper_;
        std::unique_ptr<msc::FeatureDetector> featureDetector_;

        std::shared_ptr<ImagePyramidT> liveImagePyramid;
        std::shared_ptr<GradientPyramidT> liveGradientPyramid;

        bool bootstrapped_{false};
        bool trackingLost_{false};
        KeyframeID currentKeyframe_;
        KeyframeID previousKeyframe_;

    };

    template<typename Scalar, int CS>
    SlamSystem<Scalar, CS>::~SlamSystem() {

    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::initializeSystem(msc::PinholeCamera<Scalar> &cam) {
        origCam_ = cam;
        camera_pyr_ = msc::CameraPyramid<Scalar>(cam, 4);

        tracker_ = std::make_unique<TrackerT>(camera_pyr_);
        mapper_ = std::make_unique<MapperT>(camera_pyr_);

        featureDetector_ = std::make_unique<msc::ORBDetector>();

        liveImagePyramid = std::make_shared<ImagePyramidT>(4, cam.width(), cam.height());
        liveGradientPyramid = std::make_shared<GradientPyramidT>(4, cam.width(), cam.height());

    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::processFrame(double timestamp, const cv::Mat &frame) {
        cv::Mat liveFrameColor;
        msc::Features features;
        cv::Mat liveFrame = preprocessImage(frame, liveFrameColor, features);
    }

    template<typename Scalar, int CS>
    cv::Mat SlamSystem<Scalar, CS>::preprocessImage(const cv::Mat &frame, cv::Mat &out_color, msc::Features &features) {
        origCam_.template resizeViewport(frame.cols, frame.rows);

        cv::Mat outframe_gray;

        cv::cvtColor(frame, outframe_gray, cv::COLOR_RGB2GRAY);

        cv::Mat outframe_float;

        outframe_gray.convertTo(outframe_float, CV_32FC1, 1 / 255.0);

        features = featureDetector_->DetectAndCompute(outframe_gray);

        cv::Mat keyImg = outframe_gray.clone();
        cv::drawKeypoints(keyImg, features.keypoints, keyImg, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

        /*
        cv::imshow("Orig", frame);
        cv::imshow("Gray", outframe_gray);
        cv::imshow("other", outframe_float);
        cv::imshow("features", keyImg);
        cv::waitKey();*/

        return outframe_float;
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::reset() {
        tracker_->reset();
        mapper_->reset();
        bootstrapped_ = false;
        trackingLost_ = false;

    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::bootstrapOneFrame(double timestamp, const cv::Mat &frame) {

        reset();

        msc::Features features;
        cv::Mat imageColor;
        cv::Mat imageProcessed = preprocessImage(frame, imageColor, features);
        mapper_->initOneFrame(timestamp, imageProcessed, imageColor, features);
        bootstrapped_ = true;

        uploadLiveFrame(imageProcessed);
        //currentKeyframe_ = mapper_->getMap()->keyframes_.lastID();

        //TODO: Add tracker part

    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::bootstrapTwoFrames(double timestamp0, double timestamp1, const cv::Mat &image0,
                                                    const cv::Mat &image1) {
        reset();

        msc::Features features0;
        msc::Features features1;
        cv::Mat image0Color;
        cv::Mat image1Color;
        cv::Mat image0Processed = preprocessImage(image0, image0Color, features0);
        cv::Mat image1Processed = preprocessImage(image1, image1Color, features1);
        mapper_->initTwoFrames(timestamp0, timestamp1,
                               image0Processed, image1Processed,
                               image0Color, image1Color,
                               features0, features1);
        bootstrapped_ = true;

        currentKeyframe_ = mapper_->getMap()->keyframes.lastID();
    }

/* Select the latest keyframe */
    template<typename Scalar, int CS>
    typename SlamSystem<Scalar, CS>::KeyframeID SlamSystem<Scalar, CS>::selectKeyframe() {
        KeyframeID keyframeID = 0;
        keyframeID = mapper_->getMap()->keyframes.lastID();

        return keyframeID;
    }

    template<typename Scalar, int CS>
    typename SlamSystem<Scalar, CS>::SE3T SlamSystem<Scalar, CS>::trackFrame() {
        SE3T newPose;
        auto newKeyframeID = selectKeyframe();
        if (newKeyframeID != currentKeyframe_) {
            previousKeyframe_ = currentKeyframe_;
            currentKeyframe_ = newKeyframeID;
            tracker_->setKeyframe(mapper_->getMap()->keyframes.get(currentKeyframe_));
            //TODO: Set tracker's keyframe
        }

        //TODO Tracker.trackFrame
    }

    template<typename Scalar, int CS>
    void SlamSystem<Scalar, CS>::uploadLiveFrame(const cv::Mat liveFrame) {
        for (std::size_t i = 0; i < 4; ++i) {
            if (i == 0) {
                vc::Image2DView<float, vc::TargetHost> temp(liveFrame);
                (*liveImagePyramid)[0].copyFrom(temp);
                continue;
            }

            gaussianBlurDown((*liveImagePyramid)[i - 1], (*liveImagePyramid)[i]);
            SobelGradients((*liveImagePyramid)[i], (*liveGradientPyramid)[i]);
        }
    }

} //namespace msc

#endif //MASTERS_SLAM_SYSTEM_H
