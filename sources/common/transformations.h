//
// Created by madaeu on 4/13/21.
//

#ifndef MASTERS_TRANSFORMATIONS_H
#define MASTERS_TRANSFORMATIONS_H

#include "Eigen/Core"
#include "sophus/se3.hpp"

namespace msc
{
    /**
     * Converts proximity value to depth
     * @tparam T
     * @tparam Scalar
     * @param proximity
     * @param avgDepth
     * @return
     */
    template< typename T, typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    T proximityToDepth(T proximity, Scalar avgDepth){
        return avgDepth/ proximity - avgDepth;
    }
    /**
     * Converts depth to proximity value
     * @tparam T
     * @tparam Scalar
     * @param depth
     * @param avgDepth
     * @return
     */
    template< typename T, typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    T depthToProximity(T depth, Scalar avgDepth){
        return avgDepth /(depth + avgDepth);
    }

    /**
     * Calculates the derivative of depth w.r.t. proximity
     * @tparam Scalar
     * @param depth
     * @param avgDepth
     * @return
     */
    template< typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Scalar depthJacobianProximity(Scalar depth, Scalar avgDepth){
        Scalar proximity = depthToProximity(depth, avgDepth);
        return -avgDepth / (proximity*proximity);
    }
    /**
     * Calculate proximity from zero code, depth image jacbian and code using equation
     * D = D^0+J(I)*c. Eq (1) Deepfactor
     * @tparam CodeT
     * @tparam JacT
     * @tparam T
     * @param code
     * @param jacobianProximityCode
     * @param proximityZeroCode
     * @return
     */
    template<typename CodeT, typename JacT, typename T>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    T proximityFromCode(const Eigen::MatrixBase<CodeT>& code,
                        const Eigen::MatrixBase<JacT>& proximityJacCode,
                        T proximityZeroCode){
        return proximityZeroCode+(proximityJacCode*code)(0);

    }

    /**
     * Calculates the depth image from code based on proximity from code.
     * @tparam CodeT
     * @tparam JacT
     * @tparam T
     * @param code
     * @param ProximityJacCode
     * @param proximityZeroCode
     * @param avgDepth
     * @return
     */
    template<typename CodeT, typename JacT, typename T>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    T depthFromCode(const Eigen::MatrixBase<CodeT>& code,
                    const Eigen::MatrixBase<JacT>& proximityJacCode,
                    T proximityZeroCode, T avgDepth){
        T proximity = proximityFromCode(code, proximityJacCode, proximityZeroCode);
        return proximityToDepth(proximity, avgDepth);
    }

    /**
     * Returns the relative pose between pose A and pose B expressed in pose B, where both poses
     * are expressed in the same reference system.
     * @tparam T
     * @tparam JacT
     * @param poseA
     * @param poseB
     * @return
     */
    template<typename T>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Sophus::SE3<T> relativePose(const Sophus::SE3<T>& poseA, const Sophus::SE3<T>& poseB){
        return poseA.inverse()*poseB;
    }

    /**
     * Returns the relative pose between pose A and pose B expressed in pose B, where both poses
     * are expressed in the same reference system. Also finds the jacobians of the relative pose w.r.t. pose A and B.
     * @tparam T
     * @tparam JacT
     * @param poseA
     * @param poseB
     * @param jacobianA
     * @param jacobianB
     * @return
     */
    template< typename T, typename JacT = Eigen::Matrix<T, 6,6>>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Sophus::SE3<T> relativePose(const Sophus::SE3<T>& poseA,
                                const Sophus::SE3<T>& poseB,
                                const JacT& jacobianA,
                                const JacT& jacobianB){
        using SE3 = Sophus::SE3<T>;
        using Vec3 = Eigen::Matrix<T,3,1>;

        SE3 poseAB = relativePose(poseA, poseB);
        Eigen::Matrix<T,3,3> rotationA = poseA.so3().matrix();
        Vec3 translationA = poseA.translation();
        Vec3 translationB = poseB.translation();

        //Calculate the Jacobian of the relative pose w.r.t. pose A
        //Determine the Jacobian of the relative pose w.r.t. the translation of A
        jacobianA.template block<3,3>(0,0) = -rotationA.transpose();
        jacobianA.template block<3,3>(3,0).setZero();
        //Determine the Jacobian of the relative pose w.r.t. the rotation of A
        jacobianA.template block<3,3>(0,3) = -SE3::SO3Type::hat(rotationA.transpose()*(translationA-translationB))*rotationA.transpose();
        jacobianA.template block<3,3>(3,3) = -rotationA.transpose();

        //Calculate the Jacobian of the relative pose w.r.t. pose B
        //Determine the Jacobian of the relative pose w.r.t. the translation of B
        jacobianB.template block<3,3>(0,0) = rotationA.transpose();
        jacobianB.template block<3,3>(3,0).setZero();
        //Determine the Jacobian of the relative pose w.r.t. the rotation of B
        jacobianB.template block<3,3>(3,0).setZero();
        jacobianB.template block<3,3>(3,3) = rotationA.transpose();

        return poseAB;
    }

    /**
     * Calculate the weighted distance between two poses (neglecting roll)
     * @tparam T
     * @param poseA
     * @param poseB
     * @param translationWeight
     * @param rotationWeight
     * @return
     */
    template<typename T>
    T poseDistance(const Sophus::SE3<T>& poseA,
                   const Sophus::SE3<T>& poseB,
                   T translationWeight = 8.0,
                   T rotationWeight=3.0){
        auto relPose = relativePose(poseA, poseB);
        T dRotation = relPose.so3().log().template head(2).norm();
        T dTranslation = relPose.translation().norm();
        return dTranslation*translationWeight+dRotation*rotationWeight;
    }

    /**
     * transform jacobian w.r.t. the transform
     * T(x) = X = Rx + t
     * we use R3 x SO3 instead of SE3 here to decouple (t, R)
     * [dX/dt, dX/dR] = (3 x 6)
     * [I, -(Rx)^]
     */
    template<typename T>
    Eigen::Matrix<T, 3, 6> transformJacobianPose(const Eigen::Matrix<T,3,1>& pt, const Sophus::SE3<T>& pose){
        Eigen::Matrix<T,3,6> dXdt;
        dXdt.template block<3,3>(0,0) = Eigen::Matrix<T,3,3>::Identity();
        dXdt.template block<3,3>(3,0) = -Sophus::SO3<T>::hat(pose.so3()*pt);
        return dXdt;
    }

    /**
    * transform jacobian w.r.t. input point
    * T(x) = X = Rx + t
    * dX/dx = (3 x 3)
    * [R]
    */
    template <typename T>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Eigen::Matrix<T,3,3> transformJacobianPoint(const Eigen::Matrix<T,3,1>& pt, const Sophus::SE3<T>& pose)
    {
        return pose.so3().matrix();
    }

} //namespace msc

#endif //MASTERS_TRANSFORMATIONS_H
