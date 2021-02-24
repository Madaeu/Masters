//
// Created by madaeu on 2/19/21.
//

#ifndef MASTERS_CAMERA_TRACKER_H
#define MASTERS_CAMERA_TRACKER_H

#include "camera_pyramid.h"

template <typename Scalar>
class CameraTracker {
public:
    using CameraPyramidT = CameraPyramid<Scalar>;

    CameraTracker() = delete;
    CameraTracker(const CameraPyramidT& cameraPyr);

    virtual ~CameraTracker();

private:
    CameraPyramidT cameraPyr_;
};

template <typename Scalar>
CameraTracker<Scalar>::CameraTracker(const CameraPyramidT &cameraPyr)
: cameraPyr_(cameraPyr){

}

template <typename Scalar>
CameraTracker<Scalar>::~CameraTracker(){

}
#endif //MASTERS_CAMERA_TRACKER_H
