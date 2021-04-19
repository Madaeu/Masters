//
// Created by madaeu on 4/19/21.
//

#include "cuda_context.h"

#include "VisionCore/CUDAException.hpp"
#include <cuda_runtime.h>

namespace cuda
{
    inline void throwOnErrorRuntime(const cudaError_t& err, const std::string& message)
    {
        if (err != cudaSuccess)
        {
            throw vc::CUDAException(err, message);
        }
    }

    inline void throwOnErrorDriver(const CUresult& result, const std::string& message) {
        if (result != CUDA_SUCCESS) {
            const char *errorName, *messageName;

            cuGetErrorName(result, &errorName);
            cuGetErrorString(result, &messageName);

            std::string err(errorName);
            std::string errMessage(messageName);

            std::stringstream ss;
            ss << message << ": " << err << ", " << errMessage << "\n";
            throw std::runtime_error(ss.str());
        }
    }

    DeviceInfo getDeviceInfo(uint gpuId)
    {
        cudaDeviceProp cudaDeviceProp;
        throwOnErrorRuntime(cudaGetDeviceProperties(&cudaDeviceProp, gpuId), "cudaGetDeviceProperties failed");

        // Fill in the DeviceInfo structure
        DeviceInfo deviceInfo;
        deviceInfo.Name = cudaDeviceProp.name;
        deviceInfo.SharedMemPerBlock = cudaDeviceProp.sharedMemPerBlock;
        deviceInfo.RegistersPerBlock = cudaDeviceProp.regsPerBlock;
        deviceInfo.WarpSize = cudaDeviceProp.warpSize;
        deviceInfo.MemPitch = cudaDeviceProp.memPitch;
        deviceInfo.MaxThreadsPerBlock = cudaDeviceProp.maxThreadsPerBlock;
        deviceInfo.MaxThreadsDim[0] = cudaDeviceProp.maxThreadsDim[0];
        deviceInfo.MaxThreadsDim[1] = cudaDeviceProp.maxThreadsDim[1];
        deviceInfo.MaxThreadsDim[2] = cudaDeviceProp.maxThreadsDim[2];
        deviceInfo.MaxGridSize[0] = cudaDeviceProp.maxGridSize[0];
        deviceInfo.MaxGridSize[1] = cudaDeviceProp.maxGridSize[1];
        deviceInfo.MaxGridSize[2] = cudaDeviceProp.maxGridSize[2];
        deviceInfo.TotalConstMem = cudaDeviceProp.totalConstMem;
        deviceInfo.Major = cudaDeviceProp.major;
        deviceInfo.Minor = cudaDeviceProp.minor;
        deviceInfo.ClockRate = cudaDeviceProp.clockRate;
        deviceInfo.TextureAlignment = cudaDeviceProp.textureAlignment;
        deviceInfo.DeviceOverlap = cudaDeviceProp.deviceOverlap;
        deviceInfo.KernelExecTimeoutEnabled = cudaDeviceProp.kernelExecTimeoutEnabled;
        deviceInfo.Integrated = cudaDeviceProp.integrated;
        deviceInfo.CanMapHostMemory = cudaDeviceProp.canMapHostMemory;
        deviceInfo.ComputeMode = cudaDeviceProp.computeMode;
        deviceInfo.ConcurrentKernels = cudaDeviceProp.concurrentKernels;
        deviceInfo.ECCEnabled = cudaDeviceProp.ECCEnabled;
        deviceInfo.PCIBusID = cudaDeviceProp.pciBusID;
        deviceInfo.PCIDeviceID = cudaDeviceProp.pciDeviceID;
        deviceInfo.MemoryClockRate = cudaDeviceProp.memoryClockRate;
        deviceInfo.MemoryBusWidth = cudaDeviceProp.memoryBusWidth;
        deviceInfo.MaxThreadsPerMultiprocessor = cudaDeviceProp.maxThreadsPerMultiProcessor;

        throwOnErrorRuntime(cudaMemGetInfo(&deviceInfo.FreeGlobalMem, &deviceInfo.TotalGlobalMem), "cudaMemGetInfo failed");

        return deviceInfo;
    }

    DeviceInfo getCurrentDeviceInfo()
    {
        int device;
        throwOnErrorRuntime(cudaGetDevice(&device), "cudaGetDevice failed");
        return getDeviceInfo(device);
    }

    DeviceInfo init(uint gpuId)
    {
        throwOnErrorDriver(cuInit(0), "cuInit failed");

        throwOnErrorRuntime(cudaSetDevice(gpuId), "cudaSetDevice failed");

        DeviceInfo deviceInfo = getDeviceInfo(gpuId);

        return deviceInfo;
    }

    CUcontext createRuntimeContext(std::size_t deviceId)
    {
        throwOnErrorRuntime(cudaSetDevice(deviceId), "cudaSetDevice failed");
        throwOnErrorRuntime(cudaFree(0), "cudaFree failed");

        CUcontext context;
        throwOnErrorDriver(cuCtxGetCurrent(&context), "cuCtxGetCurrent failed");

        return context;
    }

    CUcontext createAndBindContext(std::size_t deviceId)
    {
        CUcontext context;
        throwOnErrorDriver(cuCtxCreate(&context, 0, deviceId), "cuCtxCreate failed");

        return context;
    }

    CUcontext getCurrentContext()
    {
        CUcontext context;
        throwOnErrorDriver(cuCtxGetCurrent(&context), "cuCtxGetCurrent failed");
        return context;
    }

    void setCurrentContext(const CUcontext& context)
    {
        throwOnErrorDriver(cuCtxSetCurrent(context), "cuCtxSetCurrent failed");
    }

    void pushContext(const CUcontext& context)
    {
        throwOnErrorDriver(cuCtxPushCurrent(context), "cuCtxPushCurrent failed");
    }

    CUcontext popContext()
    {
        CUcontext context;
        throwOnErrorDriver(cuCtxPopCurrent(&context), "cuCtxPopCurrent failed");
        return context;
    }

} //namespace cuda
