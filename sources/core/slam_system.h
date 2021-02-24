//
// Created by madaeu on 2/15/21.
//

#ifndef MASTERS_SLAM_SYSTEM_H
#define MASTERS_SLAM_SYSTEM_H

#include "pinhole_camera.h"
#include "camera_pyramid.h"

#include <iostream>

template <typename Scalar>
class SlamSystem
{
public:
    SlamSystem() = default;
    virtual ~SlamSystem();

    void initializeSystem(PinholeCamera<Scalar>& cam);

private:
    PinholeCamera<Scalar> origCam_;
    CameraPyramid<Scalar> camera_pyr_;


};

template<typename Scalar>
SlamSystem<Scalar>::~SlamSystem() {

}

template<typename Scalar>
void SlamSystem<Scalar>::initializeSystem(PinholeCamera<Scalar> &cam) {
    origCam_ = cam;
    camera_pyr_ = CameraPyramid<Scalar>(cam, 4);
}

#endif //MASTERS_SLAM_SYSTEM_H
