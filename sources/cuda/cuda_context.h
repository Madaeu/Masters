//
// Created by madaeu on 4/19/21.
//

#ifndef MASTERS_CUDA_CONTEXT_H
#define MASTERS_CUDA_CONTEXT_H

#include <cuda.h>
#include "device_info.h"

namespace cuda
{
    DeviceInfo init(uint gpuId = 0);
    DeviceInfo getDeviceInfo(uint gpuId);
    DeviceInfo getCurrentDeviceInfo();

    CUcontext createRuntimeContext(std::size_t deviceId);
    CUcontext createAndBindContext(std::size_t deviceId);

    CUcontext getCurrentContext();

    void setCurrentContext(const CUcontext& context);

    void pushContext(const CUcontext& context);

    CUcontext popContext();

    class ScopedContextPop
    {
    public:
        ScopedContextPop()
        {
            context_ = popContext();
        }

        ~ScopedContextPop()
        {
            pushContext(context_);
        }

    private:
        CUcontext context_;
    };

} //namespace cuda


#endif //MASTERS_CUDA_CONTEXT_H
