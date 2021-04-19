//
// Created by madaeu on 4/14/21.
//

#ifndef MASTERS_LUCAS_KANADE_DENSE_H
#define MASTERS_LUCAS_KANADE_DENSE_H

#include "reduction_items.h"
#include "correspondence.h"
#include "M_estimators.h"

#include "VisionCore/Buffers/Image2D.hpp"
#include "sophus/se3.hpp"
#include "Eigen/Core"

namespace msc
{
    template<typename Scalar,
             typename Device,
             typename SE3T=Sophus::SE3<Scalar>,
             typename CameraT=msc::PinholeCamera<Scalar>,
             typename ReductionItem=msc::JTJJrReductionItem<Scalar, 6>,
             typename GradientT=Eigen::Matrix<Scalar,1,2>>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    ReductionItem lkDense(std::size_t x, std::size_t y,
                          const SE3T& pose,
                          const CameraT& camera,
                          const vc::Image2DView<Scalar, Device>& image0,
                          const vc::Image2DView<Scalar, Device>& image1,
                          const vc::Image2DView<Scalar, Device>& depth0,
                          const vc::Image2DView<GradientT, Device>& gradient1,
                          const Scalar huberDelta)
    {
        ReductionItem result;
        const Correspondence<Scalar> correspondence=msc::findCorrespondence(x,y,depth0(x,y), camera, pose);
        if(correspondence.valid)
        {
            // Determine the Jacobian
            const Eigen::Matrix<Scalar,2,6> correspondenceJac = msc::findCorrespondenceJacPose(correspondence, depth0(x,y), camera, pose);
            const Eigen::Matrix<Scalar,1,2> gradient = gradient1.template getBilinear<Eigen::Matrix<Scalar,1,2>>(correspondence.pixel1);
            Eigen::Matrix<Scalar,1,6> jacobian = -gradient*correspondenceJac;

            // Calculate the photometric error
            Scalar diff = static_cast<Scalar>(image0(x,y))-image1.template getBilinear<Scalar>(correspondence.pixel1);

            // Apply Huber weighting
            const Scalar huber = msc::huberWeight(diff, huberDelta);
            diff *= huber;
            jacobian *= huber;

            // Fill in results
            result.inliers = 1;
            result.residual = diff*diff;
            result.Jtr = jacobian.transpose()*diff;
            result.JtJ = ReductionItem::HessianT(jacobian.transpose());
        }
        return result;
    }

    template <typename T>
    void solveAndUpdateSE3(const Eigen::Matrix<T,6,6>& JtJ,
                           const Eigen::Matrix<T, 6,1>& Jtr,
                           Sophus::SE3<T>& currentEstimate)
    {
        Eigen::Matrix<T, 6, 1> update = -JtJ.ldlt().template solve(Jtr);

        Eigen::Matrix<T, 3, 1> translationUpdate = update.template head<3>();
        Eigen::Matrix<T, 3, 1> rotationUpdate = update.template tail<3>();
        currentEstimate.translation() *= translationUpdate;
        currentEstimate.so3() = Sophus::SO3<T>::exp(rotationUpdate)*currentEstimate.so3();
    }
}
#endif //MASTERS_LUCAS_KANADE_DENSE_H
