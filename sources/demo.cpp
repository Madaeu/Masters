//
// Created by madaeu on 2/17/21.
//
#include "pinhole_camera.h"
#include "camera_pyramid.h"
#include "camera_interface_factory.h"
#include "feature_detector.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "keyframe.h"
#include "decoder_network.h"
#include "cuda_context.h"
#include "photometric_factor.h"
#include "reprojection_factor.h"
#include "depth_prior_factor.h"
#include "factor_graph.h"
#include "work_implementation.h"
#include "work_manager.h"
#include "mapper.h"
#include "loop_detector.h"
#include "slam_system_options.h"
#include "slam_system.h"

#include "VisionCore/Image/BufferOps.hpp"


int main()
{
    std::unique_ptr<CameraInterface> camInterface;
    camInterface = CameraInterfaceFactory::get()->getInterfaceFromUrl("kitti:///home/madaeu/Masters/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03");
    msc::PinholeCamera<float> camera = camInterface->getIntrinsics();

    double timestamp;
    cv::Mat image;

    camInterface->grabFrames(timestamp, &image);




    //slamSystem->bootstrapOneFrame(timestamp, image);

    return 0;
}

