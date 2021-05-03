//
// Created by madaeu on 2/17/21.
//
#include "pinhole_camera.h"
#include "camera_pyramid.h"
#include "camera_interface_factory.h"
#include "feature_detector.h"
#include "slam_system.h"
#include <iostream>
#include <opencv2/imgproc.hpp>
#include "keyframe.h"
#include "decoder_network.h"
#include "cuda_context.h"
#include "photometric_factor.h"
#include "reprojection_factor.h"
#include "depth_prior_factor.h"
#include "factor_graph.h"

#include "VisionCore/Image/BufferOps.hpp"


int main()
{
    std::unique_ptr<CameraInterface> camInterface;
    camInterface = CameraInterfaceFactory::get()->getInterfaceFromUrl("kitti:///home/madaeu/Masters/data/kitti/2011_09_26/2011_09_26_drive_0001_sync/image_03");
    msc::PinholeCamera<float> camera = camInterface->getIntrinsics();

    msc::DecoderNetwork::NetworkConfiguration networkConfiguration = msc::loadJsonNetworkConfiguration("/home/madaeu/Masters/data/network/scannet256_32.cfg");
    cuda::init();
    cuda::createAndBindContext(0);
    auto scopedPop = cuda::ScopedContextPop();
    std::shared_ptr<msc::DecoderNetwork> network_ = std::make_shared<msc::DecoderNetwork>(networkConfiguration);



    double timestamp;
    cv::Mat image;

    //std::unique_ptr<SlamSystem<float, 32>> slamSystem = std::make_unique<SlamSystem<float, 32>>();
    //slamSystem->initializeSystem(camera);

    camInterface->grabFrames(timestamp, &image);

    cv::Mat resized, gray, grayFloat;
    cv::resize(image, resized, cv::Size(256,192));
    cv::cvtColor(resized, gray, cv::COLOR_RGB2GRAY);
    gray.convertTo(grayFloat, CV_32FC1, 1/255.0);

    auto keyframe = std::make_shared<msc::Keyframe<float>>(3, 256, 192, 32);

    keyframe->pose_ = Sophus::SE3<float>();
    keyframe->colorImage_ = resized.clone();
    keyframe->timestamp_ = timestamp;

    for (uint i = 0; i < 2; ++i)
    {
        assert(resized.type() == CV_8UC3);
        vc::image::fillBuffer(keyframe->validPyramid_.getLevelGPU(i), 1.0f);
        if( i == 0)
        {
            vc::Image2DView<float, vc::TargetHost> temp(grayFloat);
            keyframe->imagePyramid_.getLevelGPU(0).copyFrom(temp);
            msc::sobelGradients(keyframe->imagePyramid_.getLevelGPU(0), keyframe->gradientPyramid_.getLevelGPU(0));
            continue;
        }
        msc::gaussianBlurDown(keyframe->imagePyramid_.getLevelGPU(i-1), keyframe->imagePyramid_.getLevelGPU(i));
        msc::sobelGradients(keyframe->imagePyramid_.getLevelGPU(i), keyframe->gradientPyramid_.getLevelGPU(i));

    }
    cv::Mat netimage;
    cv::cvtColor(keyframe->colorImage_, netimage, cv::COLOR_RGB2GRAY);
    netimage.convertTo(netimage, CV_32FC1, 1/255.0);
    vc::Image2DView<float, vc::TargetHost> netImageView(netimage);

    auto origProxPtr = keyframe->proximityPyramid_.getCPUMutable();
    auto jacobianPtr = keyframe->jacobianPyramid_.getCPUMutable();
    auto uncertaintyPtr = keyframe->uncertaintyPyramid_.getCPUMutable();

    cuda::ScopedContextPop pop;
    const Eigen::VectorXf zeroCode = Eigen::VectorXf::Zero(32);

    network_->decode(netImageView, zeroCode, origProxPtr.get(), uncertaintyPtr.get(), jacobianPtr.get());

    for (uint i = 0; i < 3; ++i)
    {
        msc::updateDepth((Eigen::Matrix<float, 32, 1>)keyframe->code_,
                         keyframe->proximityPyramid_.getLevelGPU(i),
                         keyframe->jacobianPyramid_.getLevelGPU(i),
                         2.0f,
                         keyframe->depthPyramid_.getLevelGPU(i));
    }

    vc::Image2DView<float, vc::TargetHost> imageView(grayFloat);
    imageView.copyFrom(keyframe->proximityPyramid_.getLevelGPU(0));
    cv::Mat depthImage = imageView.getOpenCV();

    cv::imshow("proximity", depthImage);
    cv::waitKey();
    std::cout << "ok" << "\n";


    //slamSystem->bootstrapOneFrame(timestamp, image);

    return 0;
}

