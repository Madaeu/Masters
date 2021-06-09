//
// Created by madaeu on 5/3/21.
//

#ifndef MASTERS_DEPTH_PRIOR_FACTOR_H
#define MASTERS_DEPTH_PRIOR_FACTOR_H

#include "keyframe.h"
#include "cuda_depth_aligner.h"
#include "cuda_image_proc.h"

#include "gtsam/nonlinear/NonlinearFactor.h"
#include "gtsam/linear/HessianFactor.h"
#include "VisionCore/Buffers/BufferPyramid.hpp"
#include "opencv2/opencv.hpp"

namespace msc
{
    template<typename Scalar, int CS>
    class DepthPriorFactor: public gtsam::NonlinearFactor
    {
        using CodeT = gtsam::Vector;
        using Base = gtsam::NonlinearFactor;
        using KeyframeT = msc::Keyframe<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using AlignerT = msc::DepthAligner<Scalar, CS>;
        using AlignerPtr = typename AlignerT::Ptr;
        using StepResult = typename AlignerT::ReductionItem;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        DepthPriorFactor(cv::Mat& depth,
                         const KeyframePtr& keyframe,
                         const gtsam::Key& codeKey,
                         Scalar sigma,
                         std::size_t pyramidLevels,
                         Scalar avgDepth,
                         AlignerPtr aligner);

        virtual ~DepthPriorFactor() = default;

        double error(const gtsam::Values& c) const override;

        boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

        std::size_t dim() const override { return CS; }
    private:

        StepResult runAlignment(const CodeT& code) const;

        void updateKeyframeDepth(const CodeT& code) const;

        gtsam::Key codeKey_;
        Scalar avgDepth_;
        Scalar sigma_;
        std::size_t pyramidLevels_;
        KeyframePtr keyframe_;
        AlignerPtr aligner_;
        vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA> targetDepthPyramid_;

    };

    template<typename Scalar, int CS>
    DepthPriorFactor<Scalar, CS>::DepthPriorFactor(cv::Mat &depth,
                                                   const KeyframePtr &keyframe,
                                                   const gtsam::Key &codeKey,
                                                   Scalar sigma,
                                                   std::size_t pyramidLevels,
                                                   Scalar avgDepth,
                                                   AlignerPtr aligner)
                                                   : Base(gtsam::cref_list_of<1>(codeKey)),
                                                   codeKey_(codeKey),
                                                   avgDepth_(avgDepth),
                                                   sigma_(sigma),
                                                   pyramidLevels_(pyramidLevels),
                                                   keyframe_(keyframe),
                                                   aligner_(aligner),
                                                   targetDepthPyramid_(pyramidLevels, depth.cols, depth.rows)
    {
        vc::Image2DView<Scalar, vc::TargetHost> depthBuffer(depth);
        targetDepthPyramid_[0].copyFrom(depthBuffer);

        for (std::size_t i = 1; i < pyramidLevels_; ++i)
        {
            msc::gaussianBlurDown(targetDepthPyramid_[i-1], targetDepthPyramid_[i]);
        }
    }

    template<typename Scalar, int CS>
    double DepthPriorFactor<Scalar, CS>::error(const gtsam::Values &c) const
    {
        if(this->active(c))
        {
            CodeT code = c.template at<CodeT>(codeKey_);

            auto result = runAlignment(code);

            return 0.5 * result.residual /sigma_ / sigma_;
        }
        else
        {
            return 0.0;
        }
    }

    template<typename Scalar, int CS>
    boost::shared_ptr<gtsam::GaussianFactor>
    DepthPriorFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
    {
        if(!this->active(c))
        {
            return boost::shared_ptr<gtsam::HessianFactor>();
        }

        CodeT code = c.at<CodeT>(codeKey_);

        auto result = runAlignment(code);

        gtsam::Matrix JtJ = result.JtJ.toDenseMatrix().template cast<double>();
        gtsam::Vector Jtr = result.Jtr.template cast<double>();

        JtJ /= sigma_ * sigma_;
        Jtr /= sigma_ * sigma_;

        return boost::make_shared<gtsam::HessianFactor>(codeKey_, JtJ, -Jtr, (double)result.residual);
    }

    template<typename Scalar, int CS>
    typename DepthPriorFactor<Scalar, CS>::StepResult
    DepthPriorFactor<Scalar, CS>::runAlignment(const CodeT &code) const
    {
        updateKeyframeDepth(code);

        StepResult result;

        const Eigen::Matrix<Scalar, CS, 1> codeEigen = code.template cast<Scalar>();
        for (std::size_t i = 0; i < pyramidLevels_; ++i)
        {
            result += aligner_->runStep(codeEigen,
                                        targetDepthPyramid_[i],
                                        keyframe_->proximityPyramid.getLevelGPU(i),
                                        keyframe_->jacobianPyramid.getLevelGPU(i));
        }
        return result;
    }

    template<typename Scalar, int CS>
    void DepthPriorFactor<Scalar, CS>::updateKeyframeDepth(const CodeT &code) const
    {
        const Eigen::Matrix<Scalar, CS, 1> codeEigen = code.template cast<Scalar>();
        for (std::size_t i = 0; i < pyramidLevels_; ++i)
        {
            msc::updateDepth(codeEigen,
                             keyframe_->proximityPyramid.getLevelGPU(i),
                             keyframe_->jacobianPyramid.getLevelGPU(i),
                             avgDepth_,
                             keyframe_->depthPyramid.getLevelGPU(i));
        }
    }

}
#endif //MASTERS_DEPTH_PRIOR_FACTOR_H
