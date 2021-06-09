//
// Created by madaeu on 2/19/21.
//

#ifndef MASTERS_CAMERA_TRACKER_H
#define MASTERS_CAMERA_TRACKER_H

#include "camera_pyramid.h"
#include "keyframe.h"
#include "cuda_SE3_aligner.h"
#include "cuda_image_proc.h"

#include "sophus/se3.hpp"
#include "VisionCore/Buffers/BufferPyramid.hpp"
#include "VisionCore/Buffers/Image2D.hpp"

#include <memory>

namespace msc
{
    template <typename T>
    class CameraTracker
    {
    public:
        using GradT = Eigen::Matrix<T, 1,2>;
        using KeyframeT = Keyframe<T>;
        using ImageBufferPyramid = vc::RuntimeBufferPyramidManaged<T, vc::TargetDeviceCUDA>;
        using GradientBufferPyramid = vc::RuntimeBufferPyramidManaged<GradT, vc::TargetDeviceCUDA>;
        using SE3T = Sophus::SE3<T>;

        struct TrackerConfig
        {
            std::size_t pyramidLevels;
            std::vector<int> iterationsPerLevel = {10, 5, 4};
            double huberDelta;
        };

        CameraTracker() = delete;
        CameraTracker(const CameraPyramid<T>& cameraPyramid, const TrackerConfig& configuration);

        virtual ~CameraTracker() = default;

        void trackFrame(const ImageBufferPyramid& image, const GradientBufferPyramid& gradient);
        void reset();

        void setKeyframe(std::shared_ptr<const KeyframeT> keyframe);
        void setPose(const SE3T& poseWorldCoords);
        void setConfiguration( const TrackerConfig& newConfiguration);

        float getInliers() { return inliers_; }
        float getError() { return error_; }
        SE3T getPoseEstimate();

    private:
        float inliers_;
        float error_;
        SE3T poseCurrentKeyframe_;
        TrackerConfig configuration_;
        CameraPyramid<T> cameraPyramid_;
        std::shared_ptr<const KeyframeT> keyframe_;
        cv::Mat residualImage_;

        SE3Aligner<T> se3Aligner_;
    };

    template<typename T>
    CameraTracker<T>::CameraTracker(const CameraPyramid<T>& cameraPyramid, const TrackerConfig& configuration)
    : configuration_(configuration), cameraPyramid_(cameraPyramid)
    {
        se3Aligner_.setHuberDelta(configuration_.huberDelta);
    }

    template<typename T>
    void CameraTracker<T>::trackFrame(const ImageBufferPyramid &imagePyramid, const GradientBufferPyramid &gradientPyramid)
    {
        if ( !keyframe_)
        {
            throw std::runtime_error("Calling CameraTracker::trackFrame before a keyframe is set!");
        }


        for (int level = configuration_.pyramidLevels-1; level >= 0; --level)
        {
            for( int iterations = 0; iterations < configuration_.iterationsPerLevel[level]; ++iterations)
            {
                auto result = se3Aligner_.runStep(poseCurrentKeyframe_, cameraPyramid_[level],
                                                  keyframe_->imagePyramid_.getLevelGPU(level),
                                                  imagePyramid[level],
                                                  keyframe_->depthPyramid_.getLevelGPU(level),
                                                  gradientPyramid[level]);
                //std::cout << "JtJ: \n" << result.JtJ << "\n";
                //std::cout << "Jtr: \n" << result.Jtr << "\n";
                Eigen::Matrix<float, 6,1> update = -result.JtJ.toDenseMatrix().ldlt().solve(result.Jtr);
                Eigen::Matrix<float, 3,1> translationUpdate = update.head<3>();
                Eigen::Matrix<float, 3,1> rotationUpdate = update.tail<3>();

                poseCurrentKeyframe_.translation() += translationUpdate;
                poseCurrentKeyframe_.so3() = Sophus::SO3<float>::exp(rotationUpdate) * poseCurrentKeyframe_.so3();

                if ( level == 0 && iterations == configuration_.iterationsPerLevel[level]-1)
                {
                    inliers_ = result.inliers / (float)imagePyramid[level].area();
                    error_ = result.inliers != 0 ? result.residual / result.inliers : std::numeric_limits<T>::infinity();
                }
            }
        }

        int level = 0;
        vc::Image2DManaged<T, vc::TargetDeviceCUDA> warpedDevice(imagePyramid[level].width(), imagePyramid[level].height());
        vc::Image2DManaged<T, vc::TargetHost> warpedHost(warpedDevice.width(), warpedDevice.height());
        vc::Image2DManaged<T, vc::TargetHost> keyframeImageHost(warpedDevice.width(), warpedDevice.height());
        se3Aligner_.warp(poseCurrentKeyframe_, cameraPyramid_[level], keyframe_->imagePyramid_.getLevelGPU(level),
                         imagePyramid[level], keyframe_->depthPyramid_.getLevelGPU(level), warpedDevice);
        warpedHost.copyFrom(warpedDevice);
        keyframeImageHost.copyFrom(keyframe_->imagePyramid_.getLevelGPU(level));
        cv::Mat absDiff = cv::abs(warpedHost.getOpenCV() - keyframeImageHost.getOpenCV());
        residualImage_ = absDiff.clone();
    }

    template< typename T>
    void CameraTracker<T>::reset()
    {
        poseCurrentKeyframe_ = SE3T();
    }

    template <typename T>
    typename CameraTracker<T>::SE3T CameraTracker<T>::getPoseEstimate()
    {
        return keyframe_->pose_ * poseCurrentKeyframe_.inverse();
    }

    template<typename T>
    void CameraTracker<T>::setKeyframe(std::shared_ptr<const KeyframeT> keyframe)
    {
        if (keyframe_)
        {
            auto worldCoords = keyframe_->pose_ * poseCurrentKeyframe_.inverse();

            poseCurrentKeyframe_ = worldCoords.inverse() * keyframe_->pose_;
        }

        keyframe_ = keyframe;
    }

    template<typename T>
    void CameraTracker<T>::setPose(const SE3T &poseWorldCoords)
    {
        poseCurrentKeyframe_ = poseWorldCoords.inverse() * keyframe_->pose_;
    }

    template<typename T>
    void CameraTracker<T>::setConfiguration(const TrackerConfig &newConfiguration)
    {
        configuration_ = newConfiguration;
    }

} //namespace msc


#endif //MASTERS_CAMERA_TRACKER_H
