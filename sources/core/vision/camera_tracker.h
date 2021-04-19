//
// Created by madaeu on 2/19/21.
//

#ifndef MASTERS_CAMERA_TRACKER_H
#define MASTERS_CAMERA_TRACKER_H

#include "camera_pyramid.h"
#include "keyframe.h"

#include "sophus/se3.hpp"
#include "VisionCore/Buffers/BufferPyramid.hpp"

#include <memory>

template <typename Scalar>
class CameraTracker {
public:
    using GradientT = Eigen::Matrix<float, 1,2>;
    using CameraPyramidT = msc::CameraPyramid<Scalar>;
    using Keyframef = Keyframe<float>;
    using SE3f = Sophus::SE3<float>;
    using ImageBufferPyramid = vc::RuntimeBufferPyramidManaged<float, vc::TargetDeviceCUDA>;
    using GradientBufferPyramid = vc::RuntimeBufferPyramidManaged<GradientT, vc::TargetDeviceCUDA>;

    CameraTracker() = delete;
    CameraTracker(const CameraPyramidT& cameraPyr);

    virtual ~CameraTracker();

    void trackFrame(const ImageBufferPyramid& imagePyr, const GradientBufferPyramid& gradientPyr);
    void reset();

    void setKeyframe(std::shared_ptr<const Keyframef> keyframe);
    void setPose(const SE3f poseWorking);


private:
    CameraPyramidT cameraPyr_;

    SE3f poseCurrentKeyframe_;
    std::shared_ptr<const Keyframef> currentKeyframe_;

};

template <typename Scalar>
CameraTracker<Scalar>::CameraTracker(const CameraPyramidT &cameraPyr)
: cameraPyr_(cameraPyr){

}

template <typename Scalar>
CameraTracker<Scalar>::~CameraTracker(){

}

template <typename Scalar>
void CameraTracker<Scalar>::reset() {
    poseCurrentKeyframe_ = SE3f();
}

template<typename Scalar>
void CameraTracker<Scalar>::setKeyframe(std::shared_ptr<const Keyframef> keyframe) {
    if (currentKeyframe_)
    {
        auto worldCoords = currentKeyframe_->pose_ * poseCurrentKeyframe_.inverse();
    }

    currentKeyframe_ = keyframe;
}

template<typename Scalar>
void CameraTracker<Scalar>::setPose(const CameraTracker::SE3f poseWorking) {

}

template<typename Scalar>
void CameraTracker<Scalar>::trackFrame(const ImageBufferPyramid& imagePyr, const GradientBufferPyramid& gradientPyr) {

}


#endif //MASTERS_CAMERA_TRACKER_H
