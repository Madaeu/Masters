//
// Created by madaeu on 5/2/21.
//

#ifndef MASTERS_SPARSE_GEOMETRIC_FACTOR_H
#define MASTERS_SPARSE_GEOMETRIC_FACTOR_H

#include "uniform_sampler.h"
#include "pinhole_camera.h"
#include "keyframe.h"
#include "transformations.h"
#include "correspondence.h"

#include <Eigen/Dense>
#include "gtsam/nonlinear/NonlinearFactor.h"
#include "gtsam/linear/JacobianFactor.h"
#include "sophus/se3.hpp"
#include "cuda_image_proc.h"
#include "gtsam_traits.h"
#include "dense_SFM.h"



namespace msc
{
    template <typename Scalar, int CS>
    class SparseGeometricFactor : public gtsam::NonlinearFactor
    {
        using This = SparseGeometricFactor<Scalar,CS>;
        using Base = gtsam::NonlinearFactor;
        using PoseT = Sophus::SE3<Scalar>;
        using CodeT = gtsam::Vector;
        using KeyframeT = msc::Keyframe<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        SparseGeometricFactor(const msc::PinholeCamera<Scalar>& camera,
                              const KeyframePtr& keyframe0,
                              const KeyframePtr& keyframe1,
                              const gtsam::Key& pose0Key,
                              const gtsam::Key& pose1Key,
                              const gtsam::Key& code0Key,
                              const gtsam::Key& code1Key,
                              int numberOfPoints,
                              Scalar huberDelta,
                              bool stochastic);

        SparseGeometricFactor(const msc::PinholeCamera<Scalar>& camera,
                              const std::vector<Point>& points,
                              const KeyframePtr& keyframe0,
                              const KeyframePtr& keyframe1,
                              const gtsam::Key& pose0Key,
                              const gtsam::Key& pose1Key,
                              const gtsam::Key& code0Key,
                              const gtsam::Key& code1Key,
                              Scalar huberDelta,
                              bool stochastic);

        virtual ~SparseGeometricFactor();

        double error(const gtsam::Values& c) const override;

        boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

        size_t dim() const override {return points_.size();}

        std::string name() const;

        virtual shared_ptr clone() const;

    private:
        gtsam::Key pose0Key_;
        gtsam::Key pose1Key_;
        gtsam::Key code0Key_;
        gtsam::Key code1Key_;

        msc::PinholeCamera<Scalar> camera_;
        mutable std::vector<Point> points_;
        Scalar huberDelta_;

        mutable KeyframePtr keyframe0_;
        mutable KeyframePtr keyframe1_;

        bool stochastic_;

    };

    template<typename Scalar, int CS>
    SparseGeometricFactor<Scalar, CS>::SparseGeometricFactor(
            const msc::PinholeCamera<Scalar> &camera, const KeyframePtr &keyframe0, const KeyframePtr &keyframe1,
            const gtsam::Key &pose0Key, const gtsam::Key &pose1Key, const gtsam::Key &code0Key,
            const gtsam::Key &code1Key, int numberOfPoints, Scalar huberDelta, bool stochastic)
            : Base(gtsam::cref_list_of<4>(pose0Key)(pose1Key)(code0Key)(code1Key)),
              pose0Key_(pose0Key),
              pose1Key_(pose1Key),
              code0Key_(code0Key),
              code1Key_(code1Key),
              camera_(camera),
              huberDelta_(huberDelta),
              keyframe0_(keyframe0),
              keyframe1_(keyframe1),
              stochastic_(stochastic)
    {
        UniformSampler sampler(camera.width(), camera.height());
        points_ = sampler.samplePoints(numberOfPoints);
    }

    template<typename Scalar, int CS>
    SparseGeometricFactor<Scalar, CS>::SparseGeometricFactor(
            const msc::PinholeCamera<Scalar> &camera, const std::vector<Point> &points, const KeyframePtr &keyframe0,
            const KeyframePtr &keyframe1, const gtsam::Key &pose0Key, const gtsam::Key &pose1Key,
            const gtsam::Key &code0Key, const gtsam::Key &code1Key, Scalar huberDelta, bool stochastic)
            : Base(gtsam::cref_list_of<4>(pose0Key)(pose1Key)(code0Key)(code1Key)),
              pose0Key_(pose0Key),
              pose1Key_(pose1Key),
              code0Key_(code0Key),
              code1Key_(code1Key),
              camera_(camera),
              points_(points),
              huberDelta_(huberDelta),
              keyframe0_(keyframe0),
              keyframe1_(keyframe1),
              stochastic_(stochastic)
    {  }

    template<typename Scalar, int CS>
    SparseGeometricFactor<Scalar, CS>::~SparseGeometricFactor()
    {}

    template<typename Scalar, int CS>
    double SparseGeometricFactor<Scalar,CS>::error(const gtsam::Values &c) const
    {
        if (this->active(c))
        {
            PoseT pose0 = c.at<PoseT>(pose0Key_);
            PoseT pose1 = c.at<PoseT>(pose1Key_);
            Eigen::Matrix<Scalar, CS, 1> code0 = c.at<CodeT>(code0Key_).template cast<Scalar>();
            Eigen::Matrix<Scalar, CS, 1> code1 = c.at<CodeT>(code1Key_).template cast<Scalar>();

            Scalar avgDepth = 2.0f;

            Scalar total_sqerr = 0;
            for ( uint i = 0; i < points_.size(); ++i)
            {
                auto point = points_[i];

                PoseT pose10 = msc::relativePose(pose1, pose0);

                auto proximity0JacImage = keyframe0_->jacobianPyramid_.getLevelCPU(0);
                Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> proximity0JacCode(&proximity0JacImage((int)point.x*CS, (int)point.y));
                Scalar proximity0ZeroCode = keyframe0_->proximityPyramid_.getLevelCPU(0)(point.x, point.y);

                Scalar depth0 = msc::depthFromCode(code0, proximity0JacCode, proximity0ZeroCode, avgDepth);
                msc::Correspondence<Scalar> correspondence = findCorrespondence(point.x, point.y, depth0, camera_, pose10);

                Scalar depth1Point = correspondence.transformedPoint(2);

                Eigen::Matrix<int, 2, 1> pixel1NearestN = correspondence.pixel1.template cast<int>();

                auto proximity1JacImage = keyframe1_->jacobianPyramid_.getLevelCPU(0);
                Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> proximity1JacCode(&proximity1JacImage(pixel1NearestN(0)*CS, pixel1NearestN(1)));
                Scalar proximity1ZeroCode = keyframe1_->proximityPyramid_.getLevelCPU(0)(pixel1NearestN(0), pixel1NearestN(1));

                Scalar depth1 = msc::depthFromCode(code1, proximity1JacCode, proximity1ZeroCode, avgDepth);

                Scalar error = depth1Point - depth1;

                Scalar huberW = msc::denseSFMRobustLoss(error, huberDelta_);
                error *= huberW;

                total_sqerr += error * error;
            }
            return 0.5 * total_sqerr;
        }
        else
        {
            return 0.0;
        }
    }

    template <typename Scalar, int CS>
    boost::shared_ptr<gtsam::GaussianFactor>
    SparseGeometricFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
    {
        if (!this->active(c))
        {
            return boost::shared_ptr<gtsam::JacobianFactor>();
        }

        if (stochastic_)
        {
            UniformSampler sampler(camera_.width(), camera_.height());
            points_ = sampler.samplePoints(points_.size());
        }

        PoseT pose0 = c.at<PoseT>(pose0Key_);
        PoseT pose1 = c.at<PoseT>(pose1Key_);
        Eigen::Matrix<Scalar, CS, 1> code0 = c.at<CodeT>(code0Key_).template cast<Scalar>();
        Eigen::Matrix<Scalar, CS, 1> code1 = c.at<CodeT>(code1Key_).template cast<Scalar>();

        Scalar avgDepth = 2.0f;

        std::vector<int> dimensions = {PoseT::DoF, PoseT::DoF, CS, CS, 1};
        gtsam::VerticalBlockMatrix ab(dimensions, points_.size());

        for(uint i = 0; i < points_.size(); ++i)
        {
            auto point = points_[i];

            Eigen::Matrix<Scalar, 6, 6> pose10JacPose0;
            Eigen::Matrix<Scalar, 6, 6> pose10JacPose1;
            PoseT pose10 = msc::relativePose(pose1, pose0, pose10JacPose1, pose10JacPose0);

            auto proximity0JacImage = keyframe0_->jacobianPyramid_.getLevelCPU(0);
            Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> proximity0JacCode(&proximity0JacImage((int)point.x*CS, (int)point.y));
            Scalar proximity0ZeroCode = keyframe0_->proximityPyramid_.getLevelCPU(0)(point.x, point.y);

            Scalar depth0 = msc::depthFromCode(code0, proximity0JacCode, proximity0ZeroCode, avgDepth);
            msc::Correspondence<Scalar> correspondence = findCorrespondence(point.x, point.y, depth0, camera_, pose10);

            if(!camera_.template isPixelValid(correspondence.pixel1) || !correspondence.valid)
            {
                ab(0).template block<1, PoseT::DoF>(i,0).setZero();
                ab(1).template block<1, PoseT::DoF>(i,0).setZero();
                ab(2).template block<1, CS>(i,0).setZero();
                ab(3).template block<1, CS>(i,0).setZero();
                ab(4)(i,0) = 0;
                continue;
            }

            Scalar depth1Point = correspondence.transformedPoint;

            Eigen::Matrix<int, 2, 1> pixel1NearestN = correspondence.pixel.template cast<int>();

            auto proximity1JacImage = keyframe1_->jacobianPyramid_.getLevelCPU(0);
            Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> proximity1JacCode(&proximity1JacImage(pixel1NearestN(0)*CS, pixel1NearestN(1)));
            Scalar proximity1ZeroCode = keyframe1_->proximityPyramid_.getLevelCPU(0)(pixel1NearestN(0), pixel1NearestN(1));
            Scalar depth1 = msc::depthFromCode(code1, proximity1JacCode, proximity1ZeroCode, avgDepth);

            Scalar error = depth1 - depth1Point;

            Eigen::Matrix<Scalar, 1, PoseT::DoF> errorJacPose0;
            Eigen::Matrix<Scalar, 1, PoseT::DoF> errorJacPose1;
            Eigen::Matrix<Scalar, 1, CS> errorJacCode0;
            Eigen::Matrix<Scalar, 1, CS> errorJacCode1;

            Eigen::Matrix<Scalar, 1, 2> depthGradient = keyframe1_->depthGradients_(pixel1NearestN(0), pixel1NearestN(1));

            Eigen::Matrix<Scalar, 2, PoseT::DoF> correspondenceJacPose10 = msc::findCorrespondenceJacPose(correspondence, depth0, camera_, pose10);

            Eigen::Matrix<Scalar, 3, PoseT::DoF> transformedPointJacPose0 = msc::transformJacobianPose(correspondence.point, pose10)*pose10JacPose0;
            Eigen::Matrix<Scalar, 1, PoseT::DoF> depth1PointJacPose0 = transformedPointJacPose0.template block<1,PoseT::DoF>(2,0);
            errorJacPose0 = depth1PointJacPose0 -depthGradient * correspondenceJacPose10*pose10JacPose0;

            Eigen::Matrix<Scalar, 3, PoseT::DoF> transformedPointJacPose1 = msc::transformJacobianPose(correspondence.point, pose10)*pose10JacPose1;
            Eigen::Matrix<Scalar, 1, PoseT::DoF> depth1PointJacPose1 = transformedPointJacPose1.template block<1,PoseT::DoF>(2,0);
            errorJacPose1 = depth1PointJacPose1 -depthGradient * correspondenceJacPose10*pose10JacPose1;

            Eigen::Matrix<Scalar, 2, CS> correspondenceJacCode0;
            msc::findCorrespondenceJacCode(correspondence, depth0, camera_, pose10, proximity0JacCode, avgDepth, correspondenceJacCode0);
            Eigen::Matrix<Scalar, 3, 3> translationJacPoint = msc::transformJacobianPoint(correspondence.point, pose10);
            Eigen::Matrix<Scalar, 3, CS> translationJacCode = translationJacPoint*camera_.backwardProjectionDepthJac(correspondence.pixel0, depth0)*msc::depthJacobianProximity(depth0, avgDepth)*proximity0JacCode;
            errorJacCode0 = translationJacCode.template block<1, CS>(2,0) - depthGradient * correspondenceJacCode0;

            errorJacCode1 = -msc::depthJacobianProximity(depth1, avgDepth)*proximity1JacCode;

            Scalar huberW = msc::denseSFMRobustLoss(error, huberDelta_);

            error *= huberW;
            errorJacPose0 *= huberW;
            errorJacPose1 *= huberW;
            errorJacCode0 *= huberW;
            errorJacCode1 *= huberW;

            ab(0).template block<1, PoseT::DoF>(i,0) = errorJacPose0.template cast<double>();
            ab(1).template block<1, PoseT::DoF>(i,0) = errorJacPose1.template cast<double>();
            ab(2).template block<1, CS>(i,0) = errorJacCode0.template cast<double>();
            ab(3).template block<1, CS>(i,0) = errorJacCode1.template cast<double>();
            ab(4)(i,0) = error;
        }

        const std::vector<gtsam::Key> keys = {pose0Key_, pose1Key_, code0Key_, code1Key_};
        return boost::make_shared<gtsam::JacobianFactor>(keys, ab);
    }

    template<typename Scalar, int CS>
    std::string SparseGeometricFactor<Scalar, CS>::name() const
    {
        std::stringstream  ss;
        auto format = gtsam::DefaultKeyFormatter;
        ss << "SpareGeometricFactor" << keyframe0_->id_ << " -> " << keyframe1_->id_ << " keys = {"
           << format(pose0Key_) << ", " << format(pose1Key_) << ", " << format(code0Key_) << "}";
        return ss.str();
    }

    template<typename Scalar, int CS>
    typename SparseGeometricFactor<Scalar, CS>::shared_ptr
    SparseGeometricFactor<Scalar, CS>::clone() const
    {
        return boost::make_shared<This>(camera_, points_, keyframe0_, keyframe1_, pose0Key_, pose1Key_, code0Key_, code1Key_, huberDelta_, stochastic_);
    }

} //namespace msc
#endif //MASTERS_SPARSE_GEOMETRIC_FACTOR_H
