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
    //std::string imageSequencePath = "kitti:///home/madaeu/Masters/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03";
    std::string imageSequencePath = "scannet:///home/madaeu/Masters/data/ScanNet/scans/scene0000_00";
    std::string networkPath = "/home/madaeu/Masters/data/network/scannet256_32.cfg";
    std::string vocabularyPath = "/home/madaeu/Masters/data/BoW/small_voc.yml.gz";

    std::unique_ptr<CameraInterface> camInterface;
    camInterface = CameraInterfaceFactory::get()->getInterfaceFromUrl(imageSequencePath);
    msc::PinholeCamera<float> camera = camInterface->getIntrinsics();

    msc::SlamSystemOptions options;
    options.networkPath = networkPath;
    options.vocabularyPath = vocabularyPath;

    std::unique_ptr<msc::SlamSystem<float, 32>> slamSystem = std::make_unique<msc::SlamSystem<float,32>>();
    slamSystem->initializeSystem(camera, options);

    auto netConfiguration = slamSystem->getNetworkConfiguration();
    camera.resizeViewport(netConfiguration.inputWidth, netConfiguration.inputHeight);

    auto netCamera = slamSystem->getNetworkCamera();

    double timestamp;
    cv::Mat image;

    camInterface->grabFrames(timestamp, &image);

    slamSystem->bootstrapOneFrame(timestamp, image);

    std::mutex slam_mutex;

    int i = 0;
    while(camInterface->hasMore())
    {
        static int retries = 4;
        try
        {
            camInterface->grabFrames(timestamp, &image);
            retries = 4;
        }
        catch (std::exception& e)
        {
            std::cout << "Grab Frame Error " << retries-- << ": " << e.what();
            if( retries <= 0)
            {
                throw std::runtime_error("Failed to grab frame too many times");
            }
            continue;
        }

        std::lock_guard<std::mutex> guard(slam_mutex);
        try
        {
            slamSystem->processFrame(timestamp, image);
        }
        catch (std::exception& e)
        {
            std::cout << "Exception in processing frame: " << e.what();
            break;
        }
        i++;
        std::cout << i << "\n";
    }

    std::cout << "ok \n";

    //slamSystem->bootstrapOneFrame(timestamp, image);

    return 0;
}

