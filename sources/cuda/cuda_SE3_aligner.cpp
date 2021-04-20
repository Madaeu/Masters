//
// Created by madaeu on 4/19/21.
//

#include "cuda_SE3_aligner.h"
#include "pinhole_camera.h"
#include "lucas_kanade_dense.h"
#include "launch_utilities.h"
#include "kernel_utilities.h"

#include "VisionCore/LaunchUtils.hpp"
#include "VisionCore/Buffers/Reductions.hpp"
#include "VisionCore/Buffers/Image2D.hpp"

namespace msc
{
    template <typename Scalar, typename BaseT=SE3Aligner<Scalar>>
    __global__ void kernel_step_calculate(const Sophus::SE3<Scalar> pose,
                                          const msc::PinholeCamera<Scalar> camera,
                                          const typename BaseT::ImageBuffer image0,
                                          const typename BaseT::ImageBuffer image1,
                                          const typename BaseT::ImageBuffer depth0,
                                          const typename BaseT::GradBuffer gradient1,
                                          const Scalar huberDelta,
                                          vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename SE3Aligner<Scalar>::ReductionItem;

        Item sum;

        vc::runReductions(image0.area(), [&] __device__ (unsigned int i)
        {
            const unsigned int y = i / image0.width();
            const unsigned int x = i - (y * image0.width());

            sum += lkDense(x, y, pose, camera, image0, image1, depth0, gradient1, huberDelta);
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template <typename Scalar, typename BaseT=SE3Aligner<Scalar>>
    __global__ void kernel_warp_calculate(const Sophus::SE3<Scalar> pose,
                                          const msc::PinholeCamera<Scalar> camera,
                                          const typename BaseT::ImageBuffer image0,
                                          const typename BaseT::ImageBuffer image1,
                                          const typename BaseT::ImageBuffer depth0,
                                          typename BaseT::ImageBuffer warpedImage,
                                          vc::Buffer1DView<typename BaseT::CorrespondenceItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename BaseT::CorrespondenceItem;
        using PixelT = typename msc::PinholeCamera<Scalar>::PixelT;
        using PointT = typename msc::PinholeCamera<Scalar>::PointT;

        Item sum;

        vc::runReductions(image0.area(), [&] __device__ (unsigned int i){
            const unsigned int y = i / image0.width();
            const unsigned int x = i - (y * image0.width());

            warpedImage(x,y) = 0;

            const PixelT pixel0(x,y);
            const PointT point = camera.backwardProjection(pixel0, depth0(x,y));
            const PointT transformedPoint = pose*point;

            const Scalar depth = transformedPoint[2];

            if (depth <= 0)
            {
                return;
            }

            const PixelT pixel1 = camera.forwardProjection(transformedPoint);
            if (camera.isPixelValid(pixel1, 1))
            {
                const Scalar sampled = image1.template getBilinear<Scalar>(pixel1);

                warpedImage(x,y) = sampled;

                sum.inliers += 1;
                sum.residual += static_cast<Scalar>(image0(x,y)) - sampled;
            }
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template <typename Scalar>
    SE3Aligner<Scalar>::SE3Aligner() : max_blocks(1024), bscratch1_(max_blocks), bscratch2_(max_blocks) {}

    template <typename Scalar>
    SE3Aligner<Scalar>::~SE3Aligner() {}

    template <typename Scalar>
    typename SE3Aligner<Scalar>::CorrespondenceItem
    SE3Aligner<Scalar>::warp(const Sophus::SE3<Scalar>& pose,
                             const msc::PinholeCamera<Scalar>& camera,
                             const ImageBuffer& image0,
                             const ImageBuffer& image1,
                             const ImageBuffer& depth0,
                             ImageBuffer& warpedImage)
    {
        CorrespondenceItem result;

        int threads = 256;
        int blocks = std::min(static_cast<int>((image0.area()+threads - 1) /threads), max_blocks);
        const int smemSize = (threads /32 + 1) * sizeof(SE3Aligner::CorrespondenceItem);

        kernel_warp_calculate<<<blocks, threads, smemSize>>>(pose, camera, image0, image1, depth0, warpedImage, bscratch1_);
        cudaCheckLastError("[SE3Aligner::warp] Kernel launch failed (kernel_warp_calculate");

        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch1_, bscratch1_, blocks);
        cudaCheckLastError("[SE3Aligner::warp] Kernel launch failed (kernel_finalize_reduction)");

        auto tempBuffer = vc::Buffer1DView<CorrespondenceItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch1_);
        return result;
    }

    template <typename Scalar>
    typename SE3Aligner<Scalar>::ReductionItem
    SE3Aligner<Scalar>::runStep(const Sophus::SE3<Scalar> &pose,
                                const msc::PinholeCamera<Scalar> &camera,
                                const ImageBuffer &image0,
                                const ImageBuffer &image1,
                                const ImageBuffer &depth0,
                                const GradBuffer &gradient1)
    {
        ReductionItem result;
        int threads = 256;
        int blocks = std::min(static_cast<int>((image0.area() + threads - 1) / threads), max_blocks);
        const int smemSize = (threads / 32 +1 ) * sizeof(SE3Aligner::ReductionItem);

        kernel_step_calculate<<<blocks, threads, smemSize>>>(pose, camera, image0, image1, depth0, gradient1, huberDelta_, bscratch2_);
        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch2_, bscratch2_, blocks);
        cudaCheckLastError("[SE3Aligner::runStep] Kernel launch failed");

        auto tempBuffer = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch2_);
        return result;
    }

    template class SE3Aligner<float>;
} // namespace msc