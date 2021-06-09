//
// Created by madaeu on 4/19/21.
//

#ifndef MASTERS_CUDA_SFM_ALIGNER_H
#define MASTERS_CUDA_SFM_ALIGNER_H

#include "device_info.h"
#include "reduction_items.h"
#include "dense_SFM.h"

#include "Eigen/Core"
#include "sophus/se3.hpp"
#include "VisionCore/Buffers/Image2D.hpp"
#include "VisionCore/Buffers/Buffer1D.hpp"
#include "VisionCore/CUDAGenerics.hpp"

namespace msc
{
    template <typename Scalar>
    class PinholeCamera;

    struct SFMAlignerParameters
    {
        DenseSFMParameters SFMParameters;
        int step_threads = 32;
        int step_blocks = 11;
        int eval_threads = 224;
        int eval_blocks = 66;
    };

    template <typename Scalar, int CS>
    class SFMAligner
    {
    public:
        using Ptr = std::shared_ptr<SFMAligner<Scalar,CS>>;
        using CodeT = Eigen::Matrix<Scalar, CS, 1>;
        using ImageGrad = Eigen::Matrix<Scalar, 1,2>;
        using SE3T = Sophus::SE3<Scalar>;
        using ImageBuffer = vc::Image2DView<Scalar, vc::TargetDeviceCUDA>;
        using GradBuffer = vc::Image2DView<ImageGrad, vc::TargetDeviceCUDA>;
        using ReductionItem = JTJJrReductionItem<Scalar, 2*SE3T::DoF+CS>;
        using ErrorReductionItem = CorrespondenceReductionItem<Scalar>;
        using RelativePoseJac = Eigen::Matrix<Scalar, SE3T::DoF, SE3T::DoF>;

        SFMAligner(SFMAlignerParameters parameters = SFMAlignerParameters());
        virtual ~SFMAligner();

        ErrorReductionItem evaluateError(const SE3T& pose0,
                                         const SE3T& pose1,
                                         const msc::PinholeCamera<Scalar>& camera,
                                         const ImageBuffer& image0,
                                         const ImageBuffer& image1,
                                         const ImageBuffer& depth0,
                                         const ImageBuffer& uncertainty0,
                                         const GradBuffer& gradient1);

        ReductionItem runStep(const SE3T& pose0,
                              const SE3T& pose1,
                              const CodeT& code0,
                              const msc::PinholeCamera<Scalar>& camera,
                              const ImageBuffer& image0,
                              const ImageBuffer& image1,
                              const ImageBuffer& depth0,
                              const ImageBuffer& uncertainty0,
                              ImageBuffer& valid0,
                              const ImageBuffer& proximity0Jac,
                              const GradBuffer& gradient1);

        void setEvalThreadsBlocks(int threads, int blocks);

        void setStepThreadsBlocks(int threads, int blocks);

    private:
        //static const int max_blocks = 1024;
        SFMAlignerParameters parameters_;
        cuda::DeviceInfo deviceInfo_;
        vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch1_;
        vc::Buffer1DManaged<ErrorReductionItem, vc::TargetDeviceCUDA> bscratch2_;
    };

} // namespace msc

#endif //MASTERS_CUDA_SFM_ALIGNER_H
