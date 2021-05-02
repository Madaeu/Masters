//
// Created by madaeu on 5/2/21.
//

#ifndef MASTERS_PHOTOMETRIC_FACTOR_H
#define MASTERS_PHOTOMETRIC_FACTOR_H

#include "pinhole_camera.h"
#include "keyframe.h"
#include "cuda_SFM_aligner.h"

#include "gtsam/nonlinear/NonlinearFactor.h"
#include "gtsam/linear/HessianFactor.h"
#include "sophus/se3.hpp"
#include "VisionCore/Buffers/Buffer2D.hpp"
#include "VisionCore/Image/BufferOps.hpp"
#include "Eigen/Dense"

namespace msc
{
    template<typename Scalar, int CS>
    class PhotometricFactor: public gtsam::NonlinearFactor
    {
        using This = PhotometricFactor<Scalar,CS>;
        using Base = gtsam::NonlinearFactor;
        using SE3T = Sophus::SE3<Scalar>;
        using CodeT = gtsam::Vector;
        using KeyframeT = msc::Keyframe<Scalar>;
        using FrameT = msc::Frame<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using FramePtr = typename FrameT::Ptr;
        using AlignerT = msc::SFMAligner<Scalar,CS>;
        using AlignerPtr = typename AlignerT::Ptr;
        using JacobianResult = typename AlignerT::ReductionItem;
        using ErrorResult = typename AlignerT::ErrorReductionItem;

        static const int NP = 2*SE3T::DoF + CS;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        PhotometricFactor(const msc::PinholeCamera<Scalar> camera,
                      const KeyframePtr& keyframe,
                      const FramePtr& frame,
                      const gtsam::Key& pose0Key,
                      const gtsam::Key& pose1Key,
                      const gtsam::Key& code0Key,
                      int pyramidLevels,
                      AlignerPtr aligner,
                      bool updateValid = true);

        virtual ~PhotometricFactor() = default;

        double error(const gtsam::Values& c) const override;

        boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

        size_t dim() const override { return NP; }

        virtual shared_ptr clone() const {
            return shared_ptr(new This(camera_, keyframe_, frame_, pose0Key_, pose1Key_, code0Key_, pyramidLevels_, aligner_));
        }

        std::string name() const;

        int pyramidLevels() const { return pyramidLevels; }


    private:
        ErrorResult runWarping(const SE3T& pose0, const SE3T& pose1, const CodeT& code0) const;
        JacobianResult runAlignmentStep(const SE3T& pose0, const SE3T& pose1, const CodeT& code0) const;
        JacobianResult getJacobianIfNeeded(const SE3T pose0, const SE3T& pose1, const CodeT& code0) const;

        void updateDepthMaps(const CodeT& initialCode) const;

        gtsam::Key pose0Key_;
        gtsam::Key pose1Key_;
        gtsam::Key code0Key_;
        int pyramidLevels_;
        AlignerPtr aligner_;
        msc::PinholeCamera<Scalar> camera_;

        mutable KeyframePtr keyframe_;
        FramePtr frame_;

        mutable JacobianResult linearSystem_;
        mutable CodeT linearCode0_;
        mutable SE3T linearPose0_;
        mutable SE3T linearPose1_;

        mutable bool first_;
        bool updateValid_;
    };

    template <typename Scalar, int CS>
    PhotometricFactor<Scalar,CS>::PhotometricFactor(const msc::PinholeCamera<Scalar> camera,
                                                    const KeyframePtr &keyframe,
                                                    const FramePtr &frame,
                                                    const gtsam::Key &pose0Key,
                                                    const gtsam::Key &pose1Key,
                                                    const gtsam::Key &code0Key,
                                                    int pyramidLevels, AlignerPtr aligner,
                                                    bool updateValid)
                                                    : Base(gtsam::cref_list_of<3>(pose0Key)(pose1Key)(code0Key)),
                                                    pose0Key_(pose0Key),
                                                    pose1Key_(pose1Key),
                                                    code0Key_(code0Key),
                                                    pyramidLevels_(pyramidLevels),
                                                    aligner_(aligner),
                                                    camera_(camera),
                                                    keyframe_(keyframe),
                                                    frame_(frame),
                                                    first_(true),
                                                    updateValid_(updateValid)
                                                    {}

    template<typename Scalar, int CS>
    double PhotometricFactor<Scalar,CS>::error(const gtsam::Values &c) const
    {
        if(this->active(c))
        {
            SE3T pose0 = c.at<SE3T>(pose0Key_);
            SE3T pose1 = c.at<SE3T>(pose1Key_);
            CodeT code0 = c.at<SE3T>(code0Key_);

            updateDepthMaps(code0);

            auto result = runWarping(pose0, pose1, code0);
            return 0.5 * result.residual;
        }
        else
        {
            return 0.0;
        }

    }

    template<typename Scalar, int CS>
    boost::shared_ptr<gtsam::GaussianFactor>
    PhotometricFactor<Scalar,CS>::linearize(const gtsam::Values &c) const
    {
        if(!this->active(c))
        {
            return boost::shared_ptr<gtsam::HessianFactor>();
        }

        SE3T pose0 = c.template at<SE3T>(pose0Key_);
        SE3T pose1 = c.template at<SE3T>(pose1Key_);
        CodeT code0 = c.template at<CodeT>(code0Key_);

        auto sys = getJacobianIfNeeded(pose0, pose1, code0);

        Eigen::MatrixXd JtJ = sys.JtJ.toDenseMatrix().template cast<double>();
        Eigen::MatrixXd Jtr = -sys.Jtr.template cast<double>();

        if (sys.inliers > 0)
        {
            return boost::shared_ptr<gtsam::HessianFactor>();
        }

        const std::vector<gtsam::Key> keys = {pose0Key_, pose1Key_, code0Key_};
        std::vector<gtsam::Matrix> Gs;
        std::vector<gtsam::Vector> gs;

        const Eigen::MatrixXd G11 = JtJ.template block<6,6>(0,0);
        const Eigen::MatrixXd G12 = JtJ.template block<6,6>(0, 6);
        const Eigen::MatrixXd G13 = JtJ.template block<6,CS>(0, 12);
        const Eigen::MatrixXd G22 = JtJ.template block<6,6>(6,6);
        const Eigen::MatrixXd G23 = JtJ.template block<6,CS>(6, 12);
        const Eigen::MatrixXd G33 = JtJ.template block<CS,CS>(12, 12);
        Gs.push_back(G11);
        Gs.push_back(G12);
        Gs.push_back(G13);
        Gs.push_back(G22);
        Gs.push_back(G23);
        Gs.push_back(G33);

        const Eigen::MatrixXd g1 = Jtr.template block<6,1>(0, 0);
        const Eigen::MatrixXd g2 = Jtr.template block<6,1>(6, 0);
        const Eigen::MatrixXd g3 = Jtr.template block<CS,1>(12, 0);
        gs.push_back(g1);
        gs.push_back(g2);
        gs.push_back(g3);

        return boost::make_shared<gtsam::HessianFactor>(keys, Gs, gs, (double)sys.residual);

    }

    template<typename Scalar, int CS>
    std::string PhotometricFactor<Scalar, CS>::name() const
    {
        std::stringstream ss;
        auto format = gtsam::DefaultKeyFormatter;
        ss << "Photometric Factor " << format(pose0Key_) << " -> " << format(pose1Key_) << ", Pyramid level = " << pyramidLevels_ << "\n";
        return ss.str();
    }

    template<typename Scalar, int CS>
    typename PhotometricFactor<Scalar,CS>::JacobianResult
    PhotometricFactor<Scalar, CS>::getJacobianIfNeeded(const SE3T pose0, const SE3T &pose1, const CodeT &code0) const
    {
        double eps = 1e-6;
        if ( !gtsam::traits<SE3T>::Equals(pose0, linearPose0_, eps) ||
             !gtsam::traits<SE3T>::Equals(pose1, linearPose1_, eps) ||
             !gtsam::traits<CodeT>::Equals(code0, linearCode0_, eps))
        {
            first_ = false;

            linearSystem_ = runAlignmentStep(pose0, pose1, code0);
            linearPose0_ = pose0;
            linearPose1_ = pose1;
            linearCode0_ = code0;
        }

        return linearSystem_;
    }

    template<typename Scalar, int CS>
    typename PhotometricFactor<Scalar, CS>::JacobianResult
    PhotometricFactor<Scalar,CS>::runAlignmentStep(const SE3T &pose0, const SE3T &pose1, const CodeT &code0) const
    {
        updateDepthMaps(code0);

        int i = pyramidLevels_;

        Eigen::Matrix<Scalar, CS, 1> code = code0.template cast<Scalar>();

        vc::Image2DView<Scalar, vc::TargetDeviceCUDA> valid = keyframe_->validPyramid.getLevelGPU(i);
        auto result = aligner_->runStep(pose0, pose1, code, camera_,
                                        keyframe_->imagePyramid.getLevelGPU(i),
                                        frame_->imagePyramid.getLevelGPU(i),
                                        keyframe_->depthPyramid.getLevelGPU(i),
                                        keyframe_->uncertaintyPyramid.getLevelGPU(i),
                                        valid,
                                        keyframe_->jacobianPyramid.getLevelGPU(i),
                                        frame_->gradientPyramid.getLevelGPU(i));

        if (result.inliers > 0 )
        {
            result.residual = result.residual / result.inliers * camera_.width() * camera_.height();
        }
        else
        {
            result.residual = std::numeric_limits<float>::infinity();
        }

        return result;
    }

    template<typename Scalar, int CS>
    typename PhotometricFactor<Scalar,CS>::ErrorResult
    PhotometricFactor<Scalar,CS>::runWarping(const SE3T &pose0, const SE3T &pose1, const CodeT &code0) const
    {
        int i = pyramidLevels_;

        auto result = aligner_->evaluateError(pose0, pose1, camera_,
                                              keyframe_->imagePyramid.getLevelGPU(i),
                                              frame_->imagePyramid.getLevelGPU(i),
                                              keyframe_->depthPyramid.getLevelGPU(i),
                                              keyframe_->uncertaintyPyramid.getLevelGPU(i),
                                              frame_->gradientPyramid.getLevelGPU(i));
        if(result.inliers > 0)
        {
            result.residual = result.residual / result.inliers * camera_.width() * camera_.height();
        }
        else
        {
            result.residual = std::numeric_limits<float>::infinity();
        }
        return result;
    }

    template<typename Scalar, int CS>
    void PhotometricFactor<Scalar, CS>::updateDepthMaps(const CodeT &initialCode) const
    {
        int i = pyramidLevels_;
        Eigen::Matrix<Scalar, CS, 1> code0 = initialCode.template cast<Scalar>();
        msc::updateDepth(code0, keyframe_->proximityPyramid.getLevelGPU(i),
                         keyframe_->jacobianPyramid.getLevelGPU(i),
                         2.0f, //average depth
                         keyframe_->depthPyramid.getLevelGPU(i) );
    }

} // namespace msc
#endif //MASTERS_PHOTOMETRIC_FACTOR_H
