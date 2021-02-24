//
// Created by madaeu on 2/24/21.
//

#ifndef MASTERS_CAMERA_INTERFACE_H
#define MASTERS_CAMERA_INTERFACE_H

#include "pinhole_camera.h"
namespace cv {class Mat; }
class CameraInterface
{
public:
    CameraInterface(){}
    virtual ~CameraInterface(){}

    virtual bool supportsDepth() { return false; }
    virtual bool hasIntrinsics() { return false; }
    virtual bool hasMore() = 0;

    virtual PinholeCamera<float> getIntrinsics() {return PinholeCamera<float>{}; }

    virtual void grabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) = 0;
};
#endif //MASTERS_CAMERA_INTERFACE_H
