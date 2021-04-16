//
// Created by madaeu on 4/13/21.
//

#ifndef MASTERS_CORRESPONDENCES_H
#define MASTERS_CORRESPONDENCES_H

#include "pinhole_camera.h"
#include "transformations.h"

#include "Eigen/Core"
#include "sophus/se3.hpp"

namespace msc
{
    template< typename T>
    struct Correspondence
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        using PixelT = Eigen::Matrix<T,2,1>;
        using PointT = Eigen::Matrix<T,3,1>;

        PixelT pixel0;
        PixelT pixel1;
        PointT point;
        PointT transformedPoint;
        bool valid{false};
    };

    template<typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    msc::Correspondence<Scalar> findCorrespondence(std::size_t x, std::size_t y, Scalar depth,
                                               const msc::PinholeCamera<Scalar>& camera,
                                               const Sophus::SE3<Scalar>& cameraPose,
                                               int border = 1, Scalar minDepth = 0, bool checkBounds = true)
    {
        using PixelT = Eigen::Matrix<T,2,1>;
        using PointT = Eigen::Matrix<T,3,1>;

        const PixelT pixel0(x,y);

        msc::Correspondence<Scalar> correspondence;

        const PointT point = camera.backwardProjection(pixel0, depth);
        const PointT transformedPoint = cameraPose*point;

        const Scalar transformedDepth = transformedPoint[2];
        if (transformedDepth > minDepth)
        {
            const Pixelt pixel1 = camera.forwardProjection(transformedPoint);

            if (cam.isPixelValid(pixel1, border) || checkBounds)
            {
                correspondence.pixel0 = pixel0;
                correspondence.pixel1 = pixel1;
                correspondence.transformedPoint = transformedPoint;
                correspondence.point = point;
                correspondence.valid = true;
            }
        }
        return correspondence;
    }

    template<typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    Eigen::Matrix<Scalar, 3, 6> findCorrespondenceJacPose(const msc::Correspondence<Scalar>& correspondence,
                                                     const Scalar depth,
                                                     const msc::PinholeCamera<Scalar>& camera,
                                                     const Sophus::SE3<Scalar>& cameraPose)
    {
        const Eigen::Matrix<Scalar, 3,6> dXdt = transformJacobianPose(correspondence.point, cameraPose);
        const Eigen::Matrix<Scalar,2,3> dCamera = camera.forwardProjectionJacobian(correspondence.transformedPoint);
        return dCamera*dXdt;
    }

    template<typename Scalar>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    void findCorrespondenceJacDepth(const msc::Correspondence<Scalar>& correspondence,
                                    const Scalar depth,
                                    const msc::PinholeCamera<Scalar>& camera,
                                    const Sophus::SE3<Scalar>& cameraPose,
                                    Eigen::MatrixBase<Scalar, 2, 1>& jacobian)
    {
        const Eigen::Matrix<Scalar, 2, 3> pixel1JacTransformedPoint = camera.forwardProjectionJacobian(correspondence.transformedPoint);
        const Eigen::Matrix<Scalar, 3, 3> transformedPointJacPoint = transformJacobianPoint(correspondence.point, cameraPose);
        const Eigen::Matrix<Scalar, 3, 1> pointJacDepth = camera.backwardProjectionDepthJac(correspondence.pixel0, depth);
        jacobian.noalias() = pixel1JacTransformedPoint*transformedPointJacPoint*pointJacDepth;
    }

    template<typename Scalar, typename CorrespJacT>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    void findCorrespondenceJacProximity(const msc::Correspondence<Scalar>& correspondence,
                                        const Scalar depth,
                                        const msc::PinholeCamera<Scalar>& camera,
                                        const Sophus::SE3<Scalar>& cameraPose,
                                        const Scalar avgDepth,
                                        Eigen::MatrixBase<CorrespJacT>& jacobian)
    {
        Eigen::Matrix<Scalar, 2, 1> pixel1JacDepth;
        msc::findCorrespondenceJacDepth(correspondence, depth, camera, cameraPose, pixel1JacDepth);
        const Scalar depthJacProximity = msc::depthJacobianProximity(depth, avgDepth);
        jacobian.noalias() = pixel1JacDepth*depthJacProximity;
    }

    template<typename Scalar, typename ProximityJacT, typename CorrespJacT>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    void findCorrespondenceJacCode(const msc::Correspondence<Scalar>& correspondence,
                                   const Scalar depth,
                                   const msc::PinholeCamera<Scalar>& camera,
                                   const Sophus::SE3<Scalar>& cameraPose,
                                   const Eigen::MatrixBase<ProximityJacT>& proximityJacCode,
                                   const Scalar avgDepth,
                                   Eigen::MatrixBase<CorrespJacT>& jacobian)
    {
        Eigen::Matrix<Scalar, 2, 1> pixel1JacDepth;
        msc::findCorrespondenceJacDepth(correspondence, depth, camera, cameraPose, pixel1JacDepth);
        const Scalar depthJacProximity = msc::depthJacobianProximity(depth, avgDepth);
        jacobian.noalias() = pixel1JacDepth*depthJacProximity*proximityJacCode;
    }

} //namespace msc


#endif //MASTERS_CORRESPONDENCES_H
