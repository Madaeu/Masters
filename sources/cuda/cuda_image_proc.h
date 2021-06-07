//
// Created by madaeu on 4/6/21.
//

#ifndef MASTERS_CUDA_IMAGE_PROC_H
#define MASTERS_CUDA_IMAGE_PROC_H

#include <VisionCore/Buffers/Image2D.hpp>
#include <Eigen/Dense>

namespace msc {
    template<typename T>
    void gaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA> &input,
                          vc::Buffer2DView<T, vc::TargetDeviceCUDA> &output);

/*
template <typename T, typename GradientT>
void sobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
                    vc::Buffer2DView<Eigen::Matrix<GradientT,1,2>,vc::TargetDeviceCUDA>& grad);*/

    template<typename T, typename TG>
    void sobelGradients(const vc::Buffer2DView<T, vc::TargetDeviceCUDA> &img,
                        vc::Buffer2DView<Eigen::Matrix<TG, 1, 2>, vc::TargetDeviceCUDA> &grad);

    template <typename T, int CS, typename ImageBuffer=vc::Buffer2DView<T, vc::TargetDeviceCUDA>>
    void updateDepth(const Eigen::Matrix<T, CS, 1>& code,
                     const ImageBuffer& originalProximity,
                     const ImageBuffer& proximityJacobian,
                     T avgDepth,
                     ImageBuffer& depth);

} //namespace msc

#endif //MASTERS_CUDA_IMAGE_PROC_H
