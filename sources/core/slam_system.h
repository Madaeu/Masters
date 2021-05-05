//
// Created by madaeu on 5/5/21.
//

#ifndef MASTERS_SLAM_SYSTEM_H
#define MASTERS_SLAM_SYSTEM_H

#include "slam_system_options.h"
#include "pinhole_camera.h"
#include "mapper.h"
#include "keyframe_map.h"
#include "decoder_network.h"
#include "camera_tracker.h"
#include "mapper.h"
#include "loop_detector.h"
#include "feature_detector.h"
#include "camera_pyramid.h"
#include "cuda_SE3_aligner.h"

#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"

#include <functional>
namespace msc
{
    template<typename Scalar, int CS>
    class SlamSystem
    {
    public:
        using NetworkConfiguration = msc::DecoderNetwork::NetworkConfiguration;
        using SE3T = Sophus::SE3<Scalar>;
        using GradT = Eigen::Matrix<Scalar, 1, 2>;
        using CameraT = msc::PinholeCamera<Scalar>;
        using CameraTrackerT = msc::CameraTracker<Scalar>;
        using MapperT = msc::Mapper<Scalar, CS>;
        using CameraPyramidT = msc::CameraPyramid<Scalar>;
        using LoopDetectorT = msc::LoopDetector<Scalar>;
        using MapPtr = typename msc::Map<Scalar>::Ptr;
        using SE3AlignerT = msc::SE3Aligner<Scalar>;
        using SE3AlignerPtr = typename SE3AlignerT::Ptr;
        using KeyframeId = typename msc::Keyframe<Scalar>::IdType;
        using ImagePyramidT = vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA>;
        using GradientPyramidT = vc::RuntimeBufferPyramidManaged<GradT, vc::TargetDeviceCUDA>;

        using MapCallback = std::function<void(MapPtr)>;
        using PoseCallback = std::function<void (const SE3T&)>;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SlamSystem();
        virtual ~SlamSystem();

        SlamSystem(const SlamSystem& other) = delete;

        void initializeSystem(CameraT& camera, SlamSystemOptions& options);
        void resetSystem();
        void processFrame(double timestamp, const cv::Mat& frame);

        void bootstrapOneFrame(double timestamp, const cv::Mat& image);
        void bootstrapTwoFrames(double timestamp0, double timestamp1, const cv::Mat& image0, const cv::Mat& image1);

        void forceKeyframe() { forceKeyframe_ = true; }
        void forceFrame() {forceFrame_ = true; }

        MapPtr getMap() { return mapper_->getMap(); };
        SE3T getCameraPose() { return currentPose_; }
        NetworkConfiguration getNetworkConfiguration() { return networkConfiguration_; }
        CameraT getNetworkCamera();

        void setMapCallback(MapCallback callback)
        {
            mapCallback_ = callback;
            mapper_->setMapCallback(mapCallback_);
        }
        void setPoseCallback(PoseCallback callback)
        {
            poseCallback_ = callback;
        }
        //void setOptions(SlamSystemOptions options);

        void notifyPoseObservers();
        void notifyMapObservers();

    private:
        void initializeGPU(std::size_t deviceId);
        void uploadLiveFrame(const cv::Mat& frame);
        cv::Mat preprocessImage(const cv::Mat& frame, cv::Mat& colorImageOut, Features& features);

        SE3T trackFrame();
        SE3T relocalize();

        bool newKeyframeRequired();
        bool newFrameRequired();
        KeyframeId selectKeyframe();
        bool checkTrackingLost(const SE3T& pose);

        bool forceKeyframe_;
        bool forceFrame_;
        bool bootstrapped_;
        bool trackingLost_;

        KeyframeId currentKeyframe_;
        KeyframeId previousKeyframe_;
        SE3T currentPose_;

        SlamSystemOptions options_;

        std::shared_ptr<DecoderNetwork> network_;
        std::shared_ptr<CameraTrackerT> cameraTracker_;
        std::shared_ptr<MapperT> mapper_;
        std::shared_ptr<LoopDetectorT> loopDetector_;
        std::shared_ptr<FeatureDetector> featureDetector_;

        CameraPyramidT cameraPyramid_;
        CameraT camera_;
        NetworkConfiguration networkConfiguration_;
        cv::Mat map1_, map2_;

        std::shared_ptr<ImagePyramidT> liveImagePyramid_;
        std::shared_ptr<GradientPyramidT> liveGradientPyramid_;

        MapCallback mapCallback_;
        PoseCallback poseCallback_;

        SE3AlignerPtr se3Aligner_;

        std::vector<std::pair<int, int>> loopLinks_;
    };
}; // namespace msc


#endif //MASTERS_SLAM_SYSTEM_H
