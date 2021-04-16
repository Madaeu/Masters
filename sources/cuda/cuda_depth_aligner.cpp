//
// Created by madaeu on 4/15/21.
//

#include "cuda_depth_aligner.h"
#include "launch_utilities.h"
#include "kernel_utilities.h"
#include "transformations.h"

namespace msc
{
    // kernel
    template <typename Scalar, int CS, typename BaseT=DepthAligner<Scalar,CS>>
    __global__ /*__launch_bounds__(64, 6)*/
    void kernel_depthaligner_run_step(const typename BaseT::CodeT code,
                                      const typename BaseT::ImageBufferT targetDepth,
                                      const typename BaseT::ImageBufferT originalProximity,
                                      const typename BaseT::ImageBufferT proximityJacobian,
                                      vc::Buffer1DView<typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename BaseT::ReductionItem;
        Item sum;

        Scalar avgDepth = Scalar(2);

        vc::runReductions(targetDepth.area(), [&] __device__ (unsigned int i){
            const unsigned int y = i / targetDepth.width();
            const unsigned int x = i - (y * targetDepth.width());

            Eigen::Map<const Eigen::Matrix<Scalar, 1, CS>> temp(&proximityJacobian(x*CS,y));
            const Eigen::Matrix<Scalar,1,CS> proximityJacCode(temp);
            Scalar depth = msc::depthFromCode(code, proximityJacCode, originalProximity(x,y), avgDepth);

            Scalar diff = targetDepth(x,y) - depth;

            Eigen::Matrix<Scalar,1,CS> jacobian = -2*abs(diff)*msc::depthJacobianProximity(depth, avgDepth) * proximityJacCode;

            sum.inliers += 1;
            sum.residual += diff*diff;
            sum.Jtr += jacobian.transpose*diff;
            sum.JtJ += Item::HessianType(jacobian.transpose());
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template <typename Scalar, int CS>
    DepthAligner<Scalar, CS>::DepthAligner(): bscratch_(1024) {}

    template <typename Scalar, int CS>
    DepthAligner<Scalar, CS>::~DepthAligner() {}

    template <typename Scalar, int CS>
    typename DepthAligner<Scalar, CS>::ReductionItem
    DepthAligner<Scalar, CS>::runStep(const CodeT& code,
                                      const ImageBufferT& targetDepth,
                                      const ImageBufferT& originalProximity,
                                      const ImageBufferT& proximityJac)
    {
        int codeSize = proximityJac.width() / targetDepth.width();

        int threads = 32;
        int blocks = 40;

        const int max_blocks = 1024;
        blocks = min(blocks, max_blocks);
        const int smemSize = (threads/32)*sizeof(ReductionItem);
        kernel_depthaligner_run_step<Scalar, CS><<<blocks, threads, smemSize>>>(code, targetDepth, originalProximity, proximityJac, bscratch_);
        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch_, bscratch_, blocks);
        cudaCheckLastError("[DepthAligner::runStep] kernel launch failed");

        ReductionItem result;
        auto tempBuffer = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch_);
        return result;
    }

    template class DepthAligner<float, 32>;
} //namespace msc