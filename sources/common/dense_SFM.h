//
// Created by madaeu on 4/15/21.
//

#ifndef MASTERS_DENSE_SFM_H
#define MASTERS_DENSE_SFM_H

#include "correspondence.h"
#include "reduction_items.h"
#include "M_estimators.h"

#include "Eigen/Dense"
#include "sophus/se3.hpp"
#include "VisionCore/Buffers/Image2D.hpp"

namespace msc
{
    struct DenseSFMParameters
    {
        float huberDelta{0};
        float avgDepth{2};
        float minDepth{0};
        int validBorder{2};
    };

    template<typename Scalar>
    EIGEN_DEVICE_FUNC Scalar denseSFMRobustLoss(Scalar x, Scalar delta)
    {
        return huberWeight(x,delta);
    }

    template<typename Scalar>
    EIGEN_DEVICE_FUNC Scalar denseSFMUncertaintyWeight(Scalar logb, Scalar errorJacProximity)
    {
        // Calculate uncertainty of the photometric error
        const Scalar sigmaProximity = sqrtf(2)* expf(logb);
        const Scalar sigmaError = fabsf(errorJacProximity)*sigmaProximity;
        const Scalar sigmaErrorNorm = fmaxf(sigmaError*Scalar(1000), Scalar(1));
        return Scalar(1.0);
    }

    template<typename Scalar,
             int CS,
             typename Device=vc::TargetDeviceCUDA,
             typename SE3T=Sophus::SE3<Scalar>,
             typename ImageGradT=Eigen::Matrix<Scalar, 1,2>,
             typename GradientBuffer = vc::Image2DView<ImageGradT, Device>,
             typename ReductionItem=msc::CorrespondenceReductionItem<Scalar>>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    void denseSFMEvaluateError(std::size_t x,
                               std::size_t y,
                               const SE3T& relativePose,
                               const msc::PinholeCamera<Scalar>& camera,
                               const vc::Image2DView<Scalar, Device>& image0,
                               const vc::Image2DView<Scalar, Device>& image1,
                               const vc::Image2DView<Scalar, Device>& depth0,
                               const vc::Image2DView<Scalar, Device>& uncertainty0,
                               const GradientBuffer& gradient1,
                               const DenseSFMParameters& parameters,
                               ReductionItem& result)
    {
        // Find the pixel correspondences between images
        const Correspondence<Scalar> correspondence = msc::findCorrespondence(x,y, depth0(x,y), camera, relativePose);
        if(correspondence.valid)
        {
            // Determine the photometric error
            Scalar diff = static_cast<Scalar>(image0(x,y))-image1.template getBilinear<Scalar>(correspondence.pixel1);

            // Calculate the Huber weight based on the photometric error
            const Scalar huberW = denseSFMRobustLoss(diff, parameters.huberDelta);

            // Calculate Jacobian of the photometric error w.r.t. the proximity
            Eigen::Matrix<Scalar,2,1> pixel1JacProximity;
            findCorrespondenceJacProximity(correspondence, depth0(x,y), camera, relativePose, parameters.avgDepth, pixel1JacProximity);
            const ImageGradT gradient = gradient1.template getBilinear<ImageGradT>(correspondence.pixel1);
            const Scalar errorJacProximity = -(gradient*pixel1JacProximity)(0,0);

            // Calculate the uncertainty of the photometric error
            const Scalar sigmaW = denseSFMUncertaintyWeight(uncertainty0(x,y), errorJacProximity);

            // Determine the total pixel weighting based on huber weight and uncertainty
            const Scalar totalWeight = huberW*sigmaW;

            // Weight the photometric error
            diff *= totalWeight;

            // Fill in the reduction item
            result.inliers +=1;
            result.residual += diff*diff;
        }
    }

    template<typename Scalar,
            int CS,
            typename Device=vc::TargetDeviceCUDA,
            typename SE3T=Sophus::SE3<Scalar>,
            typename RelPoseJacobian=Eigen::Matrix<Scalar, SE3T::DoF, SE3T::DoF>,
            typename ImageBuffer=vc::Image2DView<Scalar, Device>,
            typename ImageGradT=Eigen::Matrix<Scalar,1,2>,
            typename GradBuffer=vc::Image2DView<ImageGradT, Device>,
            typename ReductionItem=msc::JTJJrReductionItem<Scalar, 2*SE3T::DoF+CS>>
    EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    void denseSFM(std::size_t x,
                  std::size_t y,
                  const SE3T& pose10,
                  const RelPoseJacobian& relPose10JacPose0,
                  const RelPoseJacobian& relPose10JacPose1,
                  const Eigen::Matrix<Scalar, CS, 1>& code,
                  const msc::PinholeCamera<Scalar>& camera,
                  const vc::Image2DView<Scalar,Device>& image0,
                  const vc::Image2DView<Scalar,Device>& image1,
                  const vc::Image2DView<Scalar,Device>& depth0,
                  const vc::Image2DView<Scalar,Device>& uncertainty0,
                  vc::Image2DView<Scalar,Device>& valid0,
                  const vc::Image2DView<Scalar,Device>& proximity0Jac,
                  const GradBuffer& gradient1,
                  const DenseSFMParameters& parameters,
                  ReductionItem& result)
    {
        // Get the jacobian of the pixel w.r.t. the code.
        Eigen::Map<const Eigen::Matrix<Scalar,1,CS>> temp(&proximity0Jac(x*CS,y));
        const Eigen::Matrix<Scalar,1,CS> proximityJacCode(temp);

        // Determine correspondences between the images
        const Correspondence<Scalar> correspondence = findCorrespondence(x,y,depth0(x,y), camera, pose10, parameters.validBorder, parameters.minDepth);

        const Eigen::Matrix<Scalar, 2,6> correspondenceJacPose10 = findCorrespondenceJacPose(correspondence, depth0(x,y), camera, pose10);

        if(correspondence.valid)
        {
            valid0(x,y) = 1.0f;

            // Calculate the jacobians w.r.t. the two poses and the zero code
            // [dErr/dPose0, dErr/dPose1, dErr/dCode0] = 1x(6+6+CS)
            Eigen::Matrix<Scalar, 1, 12+CS> J;

            // Jacobians w.r.t. pose 1 and 0
            const ImageGradT gradient = gradient1.template getBilinear<ImageGradT>(correspondence.pixel1);
            J.template block<1,6>(0,0) = -gradient*correspondenceJacPose10*relPose10JacPose0;
            J.template block<1,6>(0,6) = -gradient*correspondenceJacPose10*relPose10JacPose1;

            // Jacobian w.r.t depth( Proximity)
            Eigen::Matrix<Scalar,2,1> pixel1JacProximity;
            msc::findCorrespondenceJacProximity(correspondence, depth0(x,y), camera, pose10, parameters.avgDepth, pixel1JacProximity);
            const Scalar errorJacProximity = -(gradient*pixel1JacProximity)(0,0);

            //Jacobian depth(proximity) w.r.t. code
            J.template block<1, CS>(0,12) = errorJacProximity*proximityJacCode;

            // Calculate photometric error
            Scalar diff = static_cast<Scalar>(image0(x,y))-image1.template getBilinear<Scalar>(correspondence.pixel1);

            // Determine Huber weight
            const Scalar huberW = denseSFMRobustLoss(diff, parameters.huberDelta);

            // Calculate uncertainty of photometric error
            const Scalar sigmaW = denseSFMUncertaintyWeight(uncertainty0(x,y), errorJacProximity);

            // Determine the total pixel weighting based on huber weight and uncertainty
            const Scalar totalWeight = huberW*sigmaW;

            // Apply weighting to Jacobian and photometric error
            J *= totalWeight;
            diff *= totalWeight;

            // Fill in results
            result.inliers += 1;
            result.residual += diff*diff;
            result.Jtr += J.transpose()*diff;
            result.JtJ += typename ReductionItem::HessianT(J.transpose());
        }
    }

} //namespace msc
#endif //MASTERS_DENSE_SFM_H
