//
// Created by madaeu on 4/6/21.
//

#ifndef MASTERS_KERNEL_UTILITIES_H
#define MASTERS_KERNEL_UTILITIES_H

#include "VisionCore/Platform.hpp"
#include "VisionCore/Buffers/Reductions.hpp"
#include "VisionCore/CUDAGenerics.hpp"
#include "Eigen/Core"

namespace msc
{

#ifdef __CUDACC__
    template<typename T>
    struct reduction_traits{
        static inline EIGEN_PURE_DEVICE_FUNC void warpReduceSum(T& item)
        {
            for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
                item += vc::shfl_down(item, offset);
            }
        }

        static inline EIGEN_DEVICE_FUNC T zero() {
            return T(0);
        }
    };

    template<typename T>
    __global__ void kernel_finalize_reduction(const vc::Buffer1DView<T, vc::TargetDeviceCUDA> in,
                                              vc::Buffer1DView<T, vc::TargetDeviceCUDA> out,
                                              const int nblocks) {
        T sum;

        // this uses grid-stride loop running your lambda
        // each thread sums up multiples of data
        vc::runReductions(nblocks, [&] __device__(unsigned int i) {
            sum += in[i];
        });

        // reduce blocks and store to memory
        // we assume here this kernel is always run with one block
        // so this will be the final sum
        vc::finalizeReduction(out.ptr(), &sum, &reduction_traits<T>::warpReduceSum, reduction_traits<T>::zero());
    }

#endif

} // namespace msc

template <typename Derived>
EIGEN_DEVICE_FUNC void print_mat(const Eigen::MatrixBase<Derived>& mat)
{
    for (int i = 0; i < Derived::RowsAtCompileTime; ++i)
    {
        for (int j = 0; j < Derived::ColsAtCompileTime; ++j)
            printf("%f ", mat(i,j));
        printf("\n");
    }
}

#endif //MASTERS_KERNEL_UTILITIES_H
