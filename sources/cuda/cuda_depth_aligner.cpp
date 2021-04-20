//
// Created by madaeu on 4/20/21.
//

#include "cuda_depth_aligner.h"
#include "launch_utilities.h"
#include "kernel_utilities.h"
#include "transformations.h"

namespace msc
{
    template <typename Scalar, int CS, typename BaseT=DepthAligner<Scalar, CS>>
    __global__  void kernel_depth_aligner_run_step(const typename BaseT::CodeT code,
                                                   const typename BaseT::ImageBuffer targetDepth,
                                                   const typename BaseT::ImageBuffer originalProximity,
                                                   const typename BaseT::ImageBuffer proximityJac,
                                                   vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename DepthAligner<Scalar,CS>::ReductionItem;
        Item sum;

        Scalar avgDepth = Scalar(2);

        vc::runReductions(targetDepth.area(), [&] __device__ (unsigned int i){
           const unsigned int y = i / targetDepth.width();
           const unsigned int x = i - ( y * targetDepth.width());

           Eigen::Map<const Eigen::Matrix<Scalar, 1, CS>> temp(&proximityJac(x*CS, y));
           const Eigen::Matrix<Scalar, 1, CS> proximityJacCode(temp);
           Scalar depth = msc::depthFromCode(code, proximityJacCode, originalProximity(x,y), avgDepth);

           Scalar diff = targetDepth(x,y) - depth;

           Eigen::Matrix<Scalar, 1, CS> jacobian = -2 * abs(diff) * msc::depthJacobianProximity(depth, avgDepth) * proximityJacCode;

           sum.inliers += 1;
           sum.residual += diff * diff;
           sum.Jtr += jacobian.transpose() * diff;
           sum.JtJ += Item::HessianT(jacobian.transpose());
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template <typename Scalar, int CS>
    DepthAligner<Scalar, CS>::DepthAligner()
    :bscratch_(1024) {}

    template <typename Scalar, int CS>
    DepthAligner<Scalar, CS>::~DepthAligner() {}

    template <typename Scalar, int CS>
    typename DepthAligner<Scalar, CS>::ReductionItem
    DepthAligner<Scalar, CS>::runStep(const CodeT &code,
                                      const ImageBuffer &targetDepth,
                                      const ImageBuffer &originalProximity,
                                      const ImageBuffer &proximityJac)
    {
        int codeSize = proximityJac.width() / targetDepth.width();

        int threads = 32;
        int blocks = 40;

        blocks = std::min(blocks, max_blocks);
        const int smemSize = (threads/ 32) * sizeof(ReductionItem);

        kernel_depth_aligner_run_step<Scalar, CS><<<threads, blocks, smemSize>>>(code, targetDepth, originalProximity,
                                                                                 proximityJac, bscratch_);

        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch_, bscratch_, blocks);
        cudaCheckLastError("[DepthAligner::runStep] kernel launch failed");

        ReductionItem result;
        auto tempBuffer = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch_);
        return result;
    }

    template class DepthAligner<float, 32>;
} //namespace msc