//
// Created by madaeu on 4/6/21.
//

#ifndef MASTERS_LAUNCH_UTILITIES_H
#define MASTERS_LAUNCH_UTILITIES_H

#include "VisionCore/CUDAException.hpp"

inline void cudaCheckLastError(const std::string& msg)
{
    cudaDeviceSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if(cudaSuccess != err)
    {
        throw vc::CUDAException(err, msg);
    }
}
#endif //MASTERS_LAUNCH_UTILITIES_H
