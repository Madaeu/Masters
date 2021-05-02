//
// Created by madaeu on 5/1/21.
//

#ifndef MASTERS_GTSAM_TRAITS_H
#define MASTERS_GTSAM_TRAITS_H

#include "sophus/se3.hpp"

namespace gtsam
{

    template<typename Scalar>
    struct traits<Sophus::SE3<Scalar>>
    {
        using SE3T = Sophus::SE3<Scalar>;

        static size_t getDimension(const SE3T& pose)
        {
            return SE3T::DoF;
        }

        static void print(const SE3T& pose, const std::string& string)
        {
            std::cout << string << pose.log().transpose();
        }

        static SE3T retract(const SE3T& pose, const gtsam::Vector& delta)
        {
            Eigen::Matrix<Scalar, 6,1> update = delta.cast<Scalar>();
            Eigen::Matrix<Scalar, 3,1> translationUpdate = update.template head<3>();
            Eigen::Matrix<Scalar, 3,1> rotationUpdate = update.template tail<3>();

            SE3T newPose;
            newPose.translation() = pose.translation() + translationUpdate;
            newPose.so3() = SE3T::SO3Type ::exp(rotationUpdate)*pose.so3();
            return newPose;
        }

        static gtsam::Vector local(const SE3T first, const SE3T& second)
        {
            using tangent = SE3T::Tangent;
            tangent.template head<3>() = second.translation() - first.translation();
            tangent.template tail<3>() (second.so3() * first.so3().inverse()).log();
            return tangent.template cast<double>();
        }

        static bool Equals(const SE3T& first, const SE3T& second, Scalar tolerance)
        {
            return local(first, second).norm() < tolerance;
        }
    };

    template struct traits<Sophus::SE3<float>>;
} // namespace msc

#endif //MASTERS_GTSAM_TRAITS_H
