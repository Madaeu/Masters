//
// Created by madaeu on 4/14/21.
//

#ifndef MASTERS_REDUCTION_ITEMS_H
#define MASTERS_REDUCTION_ITEMS_H

#include "kernel_utilities.h"

#include "Eigen/Core"
#include "VisionCore/Platform.hpp"
#include "VisionCore/CUDAGenerics.hpp"
#include "VisionCore/Types/SquareUpperTriangularMatrix.hpp"

namespace msc {
    template<typename Scalar>
    struct CorrespondenceReductionItem {
        EIGEN_DEVICE_FUNC CorrespondenceReductionItem() : residual(0), inliers(0) {}

#ifdef  __CUDACC__

        EIGEN_PURE_DEVICE_FUNC static inline void warpReduceSum(CorrespondenceReductionItem<Scalar> &value)
        {
            #pragma unroll
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2)
            {
                value.residual += vc::shfl_down(value.residual, offset);
                value.inliers += vc::shfl_down(value.inliers, offset);
            }
        }

#endif

        EIGEN_DEVICE_FUNC inline CorrespondenceReductionItem<Scalar> operator+(const
        CorrespondenceReductionItem<Scalar> &rhs) const
        {
            CorrespondenceReductionItem<Scalar> result;
            result += rhs;
            return result;
        }

        EIGEN_DEVICE_FUNC inline CorrespondenceReductionItem<Scalar>&
                operator+=(const CorrespondenceReductionItem<Scalar> &rhs)
        {
            residual += rhs.residual;
            inliers += rhs.inliers;
            return *this;
        }

        Scalar residual;
        std::size_t inliers;
    };

    template<typename Scalar, int NP>
    struct JTJJrReductionItem {
        using HessianT = vc::types::SquareUpperTriangularMatrix<Scalar, NP>;
        using JacobianT = Eigen::Matrix<Scalar, NP, 1>;

        EIGEN_DEVICE_FUNC JTJJrReductionItem()
        {
            JtJ = HessianT::Zero();
            Jtr = JacobianT::Zero();
            residual = Scalar(0);
            inliers = 0;
        }

#ifdef __CUDACC__

        EIGEN_PURE_DEVICE_FUNC static inline void warpReduceSum(JTJJrReductionItem<Scalar, NP> &value) {
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                value.residual += vc::shfl_down(value.residual, offset);
                value.inliers += vc::shfl_down(value.inliers, offset);

                #pragma unroll
                for (std::size_t i = 0; i < JacobianT::RowsAtCompileTime; ++i) {
                    value.Jtr(i) += vc::shfl_down(value.Jtr(i), offset);
                }

                for (std::size_t i = 0; i < HessianT::CoeffType::RowsAtCompileTime; ++i) {
                    value.JtJ.coeff()(i) += vc::shfl_down(value.JtJ.coeff()(i), offset);
                }
            }
        }

#endif

        EIGEN_DEVICE_FUNC inline JTJJrReductionItem<Scalar, NP>
        operator+(const JTJJrReductionItem<Scalar, NP> &rhs) const
        {
            JTJJrReductionItem<Scalar, NP> result;
            result += rhs;
            return result;
        }

        EIGEN_DEVICE_FUNC inline JTJJrReductionItem<Scalar, NP>&
        operator+=(const JTJJrReductionItem<Scalar, NP> &rhs)
        {
            JtJ += rhs.JtJ;
            Jtr += rhs.Jtr;
            residual += rhs.residual;
            inliers += rhs.inliers;
            return *this;
        }

        HessianT JtJ;
        JacobianT Jtr;
        Scalar residual;
        std::size_t inliers;
    };

} //namespace msc

#if __CUDACC__
namespace vc
{

#define SPECIALIZE_JTJJrReductionItem(TYPE, CS) \
  template<> \
  struct SharedMemory<msc::JTJJrReductionItem<TYPE, CS>> { \
    typedef msc::JTJJrReductionItem<TYPE, CS> ReductionItem; \
    \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr() const { \
      extern __shared__ ReductionItem smem_JTJJrReductionItem_##TYPE_##CS[]; \
      return smem_JTJJrReductionItem_##TYPE_##CS; \
    } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr(std::size_t idx) { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem* ptr(std::size_t idx) const { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator()(std::size_t s) { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator()(std::size_t s) const { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator[](std::size_t ix) { return operator()(ix); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator[](std::size_t ix) const { return operator()(ix); } \
  };

#define SPECIALIZE_CorrespondenceReductionItem(TYPE) \
  template<> \
  struct SharedMemory<msc::CorrespondenceReductionItem<TYPE>> \
  { \
    typedef msc::CorrespondenceReductionItem<TYPE> ReductionItem; \
    \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr() const { \
      extern __shared__ ReductionItem smem_CorrespondenceReductionItem_##TYPE[]; \
      return smem_CorrespondenceReductionItem_##TYPE; \
    } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem* ptr(std::size_t idx) { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem* ptr(std::size_t idx) const { return (ptr() + idx); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator()(std::size_t s) { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator()(std::size_t s) const { return *ptr(s); } \
    EIGEN_PURE_DEVICE_FUNC inline ReductionItem& operator[](std::size_t ix) { return operator()(ix); } \
    EIGEN_PURE_DEVICE_FUNC inline const ReductionItem& operator[](std::size_t ix) const { return operator()(ix); } \
  };

SPECIALIZE_JTJJrReductionItem(float, 6)    // 6DoF tracking
SPECIALIZE_JTJJrReductionItem(float, 32)   // Depth alignment (against gt)
SPECIALIZE_JTJJrReductionItem(float, 44)   // SfM alignment 2*6DoF + 32 code
SPECIALIZE_CorrespondenceReductionItem(float)

// SPECIALIZE_JTJJrReductionItem(double)
// SPECIALIZE_CorrespondenceReductionItem(double)

#undef SPECIALIZE_JTJJrReductionItem
#undef SPECIALIZE_CorrespondenceReductionItem

} // namespace vc

namespace msc
{
    template <typename Scalar>
    struct reduction_traits<msc::CorrespondenceReductionItem<Scalar>>
    {
        using OurT = msc::CorrespondenceReductionItem<Scalar>;

        static inline EIGEN_PURE_DEVICE_FUNC void warpReduceSum(OurT& item)
        {
            OurT::warpReduceSum(item);
        }

        static inline EIGEN_DEVICE_FUNC OurT zero()
        {
            return OurT();
        }
    };

    template <typename Scalar, int NP>
    struct reduction_traits<msc::JTJJrReductionItem<Scalar, NP>>
    {
        using OurT = msc::JTJJrReductionItem<Scalar,NP>;

        static inline EIGEN_PURE_DEVICE_FUNC void warpReduceSum(OurT& item)
        {
            OurT::warpReduceSum(item);
        }

        static inline EIGEN_DEVICE_FUNC OurT zero()
        {
            return OurT();
        }
    };

} //namespace msc

#endif

#endif //MASTERS_REDUCTION_ITEMS_H
