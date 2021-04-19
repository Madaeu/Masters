//
// Created by madaeu on 4/19/21.
//

#ifndef MASTERS_CUDA_SE3_ALIGNER_H
#define MASTERS_CUDA_SE3_ALIGNER_H

#include "reduction_items.h"

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "VisionCore/Buffers/Image2D.hpp"
#include "VisionCore/Buffers/Buffer1D.hpp"
#include "VisionCore/CUDAGenerics.hpp"

namespace msc
{
    template <typename Scalar>
    class PinholeCamera;

    template <typename Scalar>
    class SE3Aligner
    {
    public:
        using Ptr = std::shared_ptr<SE3Aligner<Scalar>>;
        using UpdateT = Sophus::SE3<Scalar>;
        using ImageGrad = Eigen::Matrix<Scalar,1,2>;
        using ReductionItem = JTJJrReductionItem<Scalar,6>;
        using CorrespondenceItem = CorrespondenceReductionItem<Scalar>;
        using ImageBuffer = vc::Image2DView<Scalar, vc::TargetDeviceCUDA>;
        using GradBuffer = vc::Image2DView<ImageGrad, vc::TargetDeviceCUDA>;
        using CorrespondenceItemBuffer = vc::Buffer1DManaged<CorrespondenceItem, vc::TargetDeviceCUDA>;
        using StepReductionBuffer = vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA>;

        SE3Aligner();
        virtual ~SE3Aligner();

        CorrespondenceItem warp(const Sophus::SE3<Scalar>& pose,
                                const msc::PinholeCamera<Scalar>& camera,
                                const ImageBuffer& image0,
                                const ImageBuffer& image1,
                                const ImageBuffer& depth0,
                                ImageBuffer& warpedImage);

        ReductionItem runStep(const Sophus::SE3<Scalar>& pose,
                              const msc::PinholeCamera<Scalar>& camera,
                              const ImageBuffer& image0,
                              const ImageBuffer& image1,
                              const ImageBuffer& depth0,
                              const GradBuffer& gradient1);

        void setHuberDelta(float value) { huberDelta_ = value; }

    private:
        int max_blocks = 1024;

        CorrespondenceItemBuffer bscratch1_;
        StepReductionBuffer bscratch2_;
        float huberDelta_ = 0.1f;
    };

} // namespace msc


#endif //MASTERS_CUDA_SE3_ALIGNER_H
