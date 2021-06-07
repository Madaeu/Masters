//
// Created by madaeu on 5/2/21.
//

#include "pinhole_camera.h"
#include "keyframe.h"
#include "M_estimators.h"
#include "transformations.h"

#include "gtsam/nonlinear/NonlinearFactor.h"
#include "gtsam/linear/JacobianFactor.h"
#include "sophus/se3.hpp"
#include "correspondence.h"
#include "Eigen/Dense"
#include "opencv2/core/eigen.hpp"

#ifndef MASTERS_REPROJECTION_FACTOR_H
#define MASTERS_REPROJECTION_FACTOR_H

namespace msc
{
    template<typename Scalar, int CS>
    class ReprojectionFactor: public gtsam::NonlinearFactor
    {
        using This = ReprojectionFactor<Scalar,CS>;
        using Base = gtsam::NonlinearFactor;
        using SE3T = Sophus::SE3<Scalar>;
        using CodeT = gtsam::Vector;
        using KeyframeT = msc::Keyframe<Scalar>;
        using FrameT = msc::Frame<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using FramePtr = typename FrameT::Ptr;
        using Correspondence = msc::Correspondence<Scalar>;

    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ReprojectionFactor(const msc::PinholeCamera<Scalar>& camera,
                           const KeyframePtr& keyframe,
                           const FramePtr& frame,
                           const gtsam::Key& pose0Key,
                           const gtsam::Key& pose1Key,
                           const gtsam::Key& code0Key,
                           const Scalar maxFeatureDistance,
                           const Scalar huberDelta,
                           const Scalar sigma,
                           const int maxIterations,
                           const Scalar threshold);

        ReprojectionFactor(const msc::PinholeCamera<Scalar>& camera,
                           const std::vector<cv::DMatch>& matches,
                           const KeyframePtr& keyframe,
                           const FramePtr& frame,
                           const gtsam::Key& pose0Key,
                           const gtsam::Key& pose1Key,
                           const gtsam::Key& code0Key,
                           const Scalar huberDelta,
                           const Scalar sigma);

        virtual ~ReprojectionFactor() = default;

        double error(const gtsam::Values& c) const override;

        boost::shared_ptr<gtsam::GaussianFactor> linearize(const gtsam::Values& c) const override;

        size_t dim() const override { return 2 * matches_.size(); }

        std::string name() const;

        virtual shared_ptr clone() const;

        cv::Mat drawMatches() const;

        cv::Mat errorImage();

        Scalar ReprojectionError() const { return totalError_; }

        FramePtr frame() { return frame_; }

        KeyframePtr keyframe() { return keyframe_; }

        std::vector<cv::DMatch> matches() { return matches_; }

    private:
        gtsam::Key pose0Key_;
        gtsam::Key pose1Key_;
        gtsam::Key code0Key_;

        msc::PinholeCamera<Scalar> camera_;
        std::vector<cv::DMatch> matches_;
        Scalar huberDelta_;

        mutable KeyframePtr keyframe_;
        FramePtr frame_;

        mutable std::vector<std::pair<Eigen::Matrix<Scalar,2,1>, Eigen::Matrix<Scalar,2,1>>> correspondences_;
        mutable Scalar totalError_;

        Scalar sigma_;
    };

    template<typename Scalar, int CS>
    ReprojectionFactor<Scalar, CS>::ReprojectionFactor(const msc::PinholeCamera<Scalar> &camera, const KeyframePtr &keyframe, const FramePtr &frame,
            const gtsam::Key &pose0Key, const gtsam::Key &pose1Key, const gtsam::Key &code0Key,
            const Scalar maxFeatureDistance, const Scalar huberDelta, const Scalar sigma, const int maxIterations,
            const Scalar threshold)
            : Base(gtsam::cref_list_of<3>(pose0Key)(pose1Key)(code0Key)),
            pose0Key_(pose0Key),
            pose1Key_(pose1Key),
            code0Key_(code0Key),
            camera_(camera),
            huberDelta_(huberDelta),
            keyframe_(keyframe),
            frame_(frame),
            sigma_(sigma)
    {
                cv::BFMatcher matcher(cv::NORM_HAMMING);
                matcher.match(keyframe_->features_.descriptors, frame_->features_.descriptors, matches_);

                cv::Mat cameraMatrix;
                cv::eigen2cv(camera_.template Matrix<double>(), cameraMatrix);

                //TODO: Match pruning

    }

    template<typename Scalar, int CS>
    ReprojectionFactor<Scalar, CS>::ReprojectionFactor(const msc::PinholeCamera<Scalar> &camera, const std::vector<cv::DMatch> &matches,
            const KeyframePtr &keyframe, const FramePtr &frame, const gtsam::Key &pose0Key, const gtsam::Key &pose1Key,
            const gtsam::Key &code0Key, const Scalar huberDelta, const Scalar sigma)
            : Base(gtsam::cref_list_of<3>(pose0Key)(pose1Key)(code0Key)),
              pose0Key_(pose0Key),
              pose1Key_(pose1Key),
              camera_(camera),
              matches_(matches),
              huberDelta_(huberDelta),
              keyframe_(keyframe),
              frame_(frame),
              sigma_(sigma) { }

    template<typename Scalar, int CS>
    double ReprojectionFactor<Scalar,CS>::error(const gtsam::Values &c) const
    {
        if(this->active(c))
        {
            SE3T pose0 = c.at<SE3T>(pose0Key_);
            SE3T pose1 = c.at<SE3T>(pose1Key_);
            Eigen::Matrix<Scalar, CS, 1> code0 = c.at<CodeT>(code0Key_).template cast<Scalar>();

            Scalar avgDepth = 2.0f;

            Scalar totalSquaredError = 0;
            correspondences_.clear();
            for (unsigned int i = 0; i < matches_.size(); ++i)
            {
                cv::Point2f query = keyframe_->features_.keypoints[matches_[i].queryIdx].pt;
                cv::Point2f train = frame_->features_.keypoints[matches_[i].trainIdx].pt;
                Eigen::Matrix<Scalar, 2,1> pixel0(query.x, query.y);
                Eigen::Matrix<Scalar, 2,1> pixel1(train.x, train.y);

                SE3T pose10 = msc::relativePose(pose0, pose1);

                auto proximity0JacobianImage =keyframe_->jacobianPyramid_.getLevelCPU(0);
                Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> temp(&proximity0JacobianImage((int)query.x*CS, (int)query.y));
                Eigen::Matrix<Scalar, 1, CS> proximityJacobianCode(temp);
                Scalar proximity0Code = keyframe_->proximityPyramid_.getLevelCPU(0)(query.x, query.y);

                Scalar depth0 = msc::depthFromCode(code0, proximityJacobianCode, proximity0Code, avgDepth);
                msc::Correspondence<Scalar> correspondence = msc::findCorrespondence(query.x, query.y, depth0, camera_, pose10, 1.0f, 0.0f, false);

                correspondences_.template emplace_back(pixel1, correspondence.pixel1);

                Eigen::Matrix<Scalar, 2, 1> difference = pixel1 - correspondence.pixel1;

                Scalar error = difference.norm();

                Scalar huberWeight = msc::cauchyWeight(error, huberDelta_);
                error *= huberWeight;

                totalSquaredError += error * error;
            }
            totalError_ = totalSquaredError;
            return 0.5 * totalSquaredError / sigma_ / sigma_;
        }
        else
        {
            return 0.0;
        }
    }

    template<typename Scalar, int CS>
    boost::shared_ptr<gtsam::GaussianFactor>
    ReprojectionFactor<Scalar, CS>::linearize(const gtsam::Values &c) const
    {
        if (!this->active(c))
        {
            return boost::shared_ptr<gtsam::JacobianFactor>();
        }

        SE3T pose0 = c.template at<SE3T>(pose0Key_);
        SE3T pose1 = c.template at<SE3T>(pose1Key_);
        Eigen::Matrix<Scalar, CS, 1> code0 = c.template at<CodeT>(code0Key_).template cast<Scalar>();

        Scalar avgDepth = 2.0f;

        std::vector<int> dimensions = {SE3T::DoF, SE3T::DoF, CS, 1};
        gtsam::VerticalBlockMatrix ab(dimensions, 2*matches_.size());

        Scalar totalError = 0;
        correspondences_.clear();

        for (unsigned int i = 0; i < matches_.size(); ++i)
        {
            cv::Point2f query = keyframe_->features_.keypoints[matches_[i].queryIdx].pt;
            cv::Point2f train = frame_->features_.keypoints[matches_[i].trainIdx].pt;
            Eigen::Matrix<Scalar, 2, 1> pixel1(query.x, query.y);
            Eigen::Matrix<Scalar, 2, 1> pixel0(train.x, train.y);

            Eigen::Matrix<Scalar, 6, 6> pose10JacobianPose0;
            Eigen::Matrix<Scalar, 6, 6> pose10JacobianPose1;
            SE3T pose10 = msc::relativePose(pose1, pose0, pose10JacobianPose1, pose10JacobianPose0);

            auto proximityJacobianImage = keyframe_->jacobianPyramid_.getLevelCPU(0);
            Eigen::Map<Eigen::Matrix<Scalar, 1, CS>> proximityJacobianCode(&proximityJacobianImage((int)query.x * CS, (int)query.y));
            Scalar proximity0Code = keyframe_->proximityPyramid_.getLevelCPU(0)(query.x, query.y);

            Scalar depth0 = msc::depthFromCode(code0, proximityJacobianCode, proximity0Code, avgDepth);
            msc::Correspondence<Scalar> correspondence = msc::findCorrespondence(query.x, query.y, depth0, camera_, pose10, 1.0f, 0.0f, false);

            Scalar depth2 = correspondence.transformedPoint[2];

            if ( !correspondence.valid)
            {
                ab(0).template block<2, SE3T::DoF>(i, 0).setZero();
                ab(1).template block<2, SE3T::DoF>(i, 0).setZero();
                ab(2).template block<2, CS>(i, 0).setZero();
                ab(3).template block<2, 1>(2 * i, 0).setZero();
                continue;
            }

            correspondences_.template emplace_back(pixel1, correspondence.pixel1);

            Eigen::Matrix<Scalar, 2, CS> correspondenceJacCode;
            msc::findCorrespondenceJacCode(correspondence, depth0, camera_, pose10, proximityJacobianCode, avgDepth, correspondenceJacCode);

            Eigen::Matrix<Scalar, 2, 6> correspondenceJacPose10;
            correspondenceJacPose10 = msc::findCorrespondenceJacPose(correspondence, depth0, camera_, pose10);

            Eigen::Matrix<Scalar, 2, SE3T::DoF> errorJacobianPose0 = correspondenceJacPose10 * pose10JacobianPose0;
            Eigen::Matrix<Scalar, 2, SE3T::DoF> errorJacobianPose1 = correspondenceJacPose10 * pose10JacobianPose1;
            Eigen::Matrix<Scalar, 2, CS> errorJacobianCode = correspondenceJacCode;

            Eigen::Matrix<Scalar, 2, 1> difference = pixel1 - correspondence.pixel1;

            Scalar error = difference.norm();

            Scalar huberWeight = msc::cauchyWeight(error, huberDelta_);

            errorJacobianPose0 *= huberWeight;
            errorJacobianPose1 *= huberWeight;
            errorJacobianCode *= huberWeight;

            difference *= huberWeight;

            totalError += error * error;

            errorJacobianPose0 /= sigma_;
            errorJacobianPose1 /= sigma_;
            errorJacobianCode /= sigma_;
            difference /= sigma_;

            ab(0).template block<2, SE3T::DoF>(2 * i, 0) = errorJacobianPose0.template cast<double>();
            ab(1).template block<2, SE3T::DoF>(2 * i, 0) = errorJacobianPose1.template cast<double>();
            ab(2).template block<2, CS>(2 * i, 0) = errorJacobianCode.template cast<double>();
            ab(3).template block<2, 1>(2 * i, 0) = difference.template cast<double>();
        }

        totalError_ = totalError;

        const std::vector<gtsam::Key> keys = {pose0Key_, pose1Key_, code0Key_};
        return boost::make_shared<gtsam::JacobianFactor>(keys, ab);

    }

    template<typename Scalar, int CS>
    std::string ReprojectionFactor<Scalar, CS>::name() const
    {
        std::stringstream ss;
        auto format = gtsam::DefaultKeyFormatter;
        ss << "Reprojection Factor " << format(pose0Key_) << " -> " << format(pose1Key_) << " {" << format(code0Key_) << "} \n";
        return ss.str();
    }

    template<typename Scalar, int CS>
    typename ReprojectionFactor<Scalar, CS>::shared_ptr
    ReprojectionFactor<Scalar, CS>::clone() const
    {
        auto ptr = boost::make_shared<This>(camera_, matches_, keyframe_, frame_, pose0Key_, pose1Key_, code0Key_, huberDelta_, sigma_);
        ptr->correspondences_ = correspondences_;
        ptr->totalError_ = totalError_;
        return ptr;
    }

    template<typename Scalar, int CS>
    cv::Mat ReprojectionFactor<Scalar, CS>::drawMatches() const
    {
        cv::Mat imageMatches;
        cv::drawMatches(keyframe_->colorImage_, keyframe_->features_.keypoints, frame_->colorImage_,
                        frame_->features_.keypoints, matches_, imageMatches);
        return imageMatches;
    }

    template<typename Scalar, int CS>
    cv::Mat ReprojectionFactor<Scalar, CS>::errorImage()
    {
        cv::Mat image = frame_->colorImage_.clone();
        for (auto& correspondence: correspondences_)
        {
            cv::Point point0(correspondence.first(0), correspondence.first(1));
            cv::Point point1(correspondence.second(0), correspondence.second(1));

            cv::arrowedLine(image, point1, point0, cv::Scalar(0,0,255));
        }
        return image;
    }




} // namespace msc

#endif //MASTERS_REPROJECTION_FACTOR_H
