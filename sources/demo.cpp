//
// Created by madaeu on 2/17/21.
//
#include "pinhole_camera.h"
#include "camera_pyramid.h"
#include "camera_interface_factory.h"
#include "feature_detector.h"
#include "slam_system.h"

#include <iostream>

int main()
{
    std::unique_ptr<CameraInterface> camInterface;
    camInterface = CameraInterfaceFactory::get()->getInterfaceFromUrl("kitti:///home/madaeu/Masters/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03");
    msc::PinholeCamera<float> camera = camInterface->getIntrinsics();

    double timestamp;
    cv::Mat image;

    std::unique_ptr<SlamSystem<float, 32>> slamSystem = std::make_unique<SlamSystem<float, 32>>();
    slamSystem->initializeSystem(camera);

    camInterface->grabFrames(timestamp, &image);

    slamSystem->bootstrapOneFrame(timestamp, image);



    return 0;
}

