//
// Created by madaeu on 4/19/21.
//

#include "cuda_SFM_aligner.h"
#include "pinhole_camera.h"
#include "launch_utilities.h"
#include "kernel_utilities.h"
#include "cuda_context.h"

#include "VisionCore/LaunchUtils.hpp"
#include "VisionCore/Buffers/Reductions.hpp"
#include "VisionCore/Buffers/Image2D.hpp"

namespace msc
{
    __constant__ DenseSFMParameters SFMParameters;

    template <typename Scalar, int CS, typename BaseT=SFMAligner<Scalar, CS>>
    __global__ void kernel_step_calculate(const typename BaseT::SE3T relativePose,
                                          const typename BaseT::RelativePoseJac relPoseJacPose0,
                                          const typename BaseT::RelativePoseJac relPoseJacPose1,
                                          const typename BaseT::CodeT code,
                                          const msc::PinholeCamera<Scalar> camera,
                                          const typename BaseT::ImageBuffer image0,
                                          const typename BaseT::ImageBuffer image1,
                                          const typename BaseT::ImageBuffer depth0,
                                          const typename BaseT::ImageBuffer uncertainty0,
                                          typename BaseT::ImageBuffer valid0,
                                          const typename BaseT::ImageBuffer proximity0Jac,
                                          const typename BaseT::GradBuffer gradient1,
                                          vc::Buffer1DView< typename BaseT::ReductionItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename SFMAligner<Scalar,CS>::ReductionItem;
        Item sum;

        vc::runReductions(image0.area(), [&] __device__ (unsigned int i){
           const unsigned int y = i / image0.width();
           const unsigned int x = i - (y * image0.width());
           denseSFM<Scalar,CS>(x,y,relativePose,relPoseJacPose0,relPoseJacPose1, code, camera, image0, image1,
                               depth0, uncertainty0, valid0, proximity0Jac, gradient1, SFMParameters, sum);
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template <typename Scalar, int CS, typename BaseT=SFMAligner<Scalar, CS>>
    __global__ void kernel_evaluate_error(const typename BaseT::SE3T relativePose,
                                          const msc::PinholeCamera<Scalar> camera,
                                          const typename BaseT::ImageBuffer image0,
                                          const typename BaseT::ImageBuffer image1,
                                          const typename BaseT::ImageBuffer depth0,
                                          const typename BaseT::ImageBuffer uncertainty0,
                                          const typename BaseT::GradBuffer gradient1,
                                          vc::Buffer1DView<typename BaseT::ErrorReductionItem, vc::TargetDeviceCUDA> bscratch)
    {
        using Item = typename SFMAligner<Scalar,CS>::ErrorReductionItem;
        Item sum;

        vc::runReductions(image0.area(), [&] __device__ (unsigned int i)
        {
            const unsigned int y = i / image0.width();
            const unsigned int x = i - ( y * image0.width());

            denseSFMEvaluateError<Scalar, CS>(x, y, relativePose, camera, image0, image1, depth0,
                                              uncertainty0, gradient1, SFMParameters, sum);
        });

        vc::finalizeReduction(bscratch.ptr(), &sum, Item::warpReduceSum, Item());
    }

    template < typename Scalar, int CS>
    SFMAligner<Scalar,CS>::SFMAligner(SFMAlignerParameters parameters)
    : parameters_(parameters), bscratch1_(1024), bscratch2_(1024)
    {
        setEvalThreadsBlocks(parameters.eval_threads, parameters.eval_blocks);
        setStepThreadsBlocks(parameters.step_threads, parameters.step_blocks);

        cudaMemcpyToSymbol(SFMParameters, &parameters.SFMParameters, sizeof(DenseSFMParameters));
        cudaCheckLastError("Copying SFM parameters to GPU failed");

        deviceInfo_ = cuda::getCurrentDeviceInfo();
    }

    template< typename Scalar, int CS>
    SFMAligner<Scalar,CS>::~SFMAligner() {}

    template <typename Scalar, int CS>
    typename SFMAligner<Scalar,CS>::ErrorReductionItem
    SFMAligner<Scalar,CS>::evaluateError(const SE3T &pose0,
                                         const SE3T &pose1,
                                         const msc::PinholeCamera<Scalar> &camera,
                                         const ImageBuffer &image0,
                                         const ImageBuffer &image1,
                                         const ImageBuffer &depth0,
                                         const ImageBuffer &uncertainty0,
                                         const GradBuffer &gradient1)
    {
        const SE3T relativePose = msc::relativePose(pose1, pose0);

        int max_blocks = 1024;
        int threads = parameters_.eval_threads;
        int blocks = std::min(parameters_.step_blocks, max_blocks);
        const int smemSize = (threads / deviceInfo_.WarpSize) * sizeof(ErrorReductionItem);

        kernel_evaluate_error<Scalar, CS><<<blocks, threads, smemSize>>>(relativePose, camera, image0,
                                                                         image1, depth0, uncertainty0,
                                                                         gradient1, bscratch2_);

        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch2_, bscratch2_, blocks);
        cudaCheckLastError("[SfmAligner::EvaluateError] kernel launch failed");

        ErrorReductionItem result;
        auto tempBuffer = vc::Buffer1DView<ErrorReductionItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch2_);
        return result;
    }

    template <typename Scalar, int CS>
    typename SFMAligner<Scalar, CS>::ReductionItem
    SFMAligner<Scalar, CS>::runStep(const SE3T &pose0,
                                    const SE3T &pose1,
                                    const CodeT &code0,
                                    const msc::PinholeCamera<Scalar> &camera,
                                    const ImageBuffer &image0,
                                    const ImageBuffer &image1,
                                    const ImageBuffer &depth0,
                                    const ImageBuffer &uncertainty0,
                                    ImageBuffer &valid0,
                                    const ImageBuffer &proximity0Jac,
                                    const GradBuffer &gradient1)
    {
        RelativePoseJac relativePoseJacPose0;
        RelativePoseJac relativePoseJacPose1;
        const SE3T relativePose = msc::relativePose(pose1, pose0, relativePoseJacPose1, relativePoseJacPose0);

        int max_blocks = 1024;
        int threads = parameters_.step_threads;
        int blocks = std::min(parameters_.step_blocks, max_blocks);
        const int smemSize = (threads / 32) * sizeof(ReductionItem);
        if (smemSize > deviceInfo_.SharedMemPerBlock)
            throw std::runtime_error("Too much shared memory per block requested (" + std::to_string(smemSize)
                                     + " vs " + std::to_string(deviceInfo_.SharedMemPerBlock) + ")");

        kernel_step_calculate<Scalar, CS><<<threads, blocks, smemSize>>>(relativePose, relativePoseJacPose0,
                                                                         relativePoseJacPose1,code0,
                                                                         camera, image0, image1,
                                                                         depth0, uncertainty0,
                                                                         valid0, proximity0Jac,
                                                                         gradient1, bscratch1_);

        kernel_finalize_reduction<<<1, threads, smemSize>>>(bscratch1_, bscratch1_, blocks);
        cudaCheckLastError("[SfmAligner::RunStep] kernel launch failed");

        ReductionItem result;
        auto tempBuffer = vc::Buffer1DView<ReductionItem, vc::TargetHost>(&result, 1);
        tempBuffer.copyFrom(bscratch1_);
        return result;
    }

    template <typename Scalar, int CS>
    void SFMAligner<Scalar, CS>::setEvalThreadsBlocks(int threads, int blocks)
    {
        parameters_.eval_threads = threads;
        parameters_.eval_blocks = blocks;
    }

    template <typename Scalar, int CS>
    void SFMAligner<Scalar, CS>::setStepThreadsBlocks(int threads, int blocks)
    {
        parameters_.step_threads = threads;
        parameters_.step_blocks = blocks;
    }

    template class SFMAligner<float, 32>;
} // namespace msc
