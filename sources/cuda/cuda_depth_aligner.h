//
// Created by madaeu on 4/15/21.
//

#ifndef MASTERS_CUDA_DEPTH_ALIGNER_H
#define MASTERS_CUDA_DEPTH_ALIGNER_H

#include "reduction_items.h"

#include "Eigen/Core"
#include "VisionCore/Buffers/Image2D.hpp"
#include "VisionCore/Buffers/Buffer1D.hpp"

namespace msc
{
    template< typename Scalar, int CS>
    class DepthAligner
    {
        using Ptr = std::shared_ptr<DepthAligner<Scalar, CS>>;
        using CodeT = Eigen::Matrix<Scalar, CS, 1>;
        using ImageBufferT = vc::Image2DView<Scalar,vc::TargetDeviceCUDA>;
        using ReductionItem = JTJJrReductionItem<Scalar,CS>;

    public:
        DepthAligner();
        virtual ~DepthAligner();

        ReductionItem runStep(const CodeT& code,
                              const ImageBufferT& targetDepth,
                              const ImageBufferT& originalProximity,
                              const ImageBufferT& proximityJacobian);

    private:
        vc::Buffer1DManaged<ReductionItem, vc::TargetDeviceCUDA> bscratch_;
    };

} //namespace msc


#endif //MASTERS_CUDA_DEPTH_ALIGNER_H
