//
// Created by madaeu on 4/19/21.
//

#ifndef MASTERS_DEVICE_INFO_H
#define MASTERS_DEVICE_INFO_H

namespace cuda
{
    struct DeviceInfo
    {
        std::string Name;
        std::size_t TotalGlobalMem;
        std::size_t FreeGlobalMem;
        std::size_t SharedMemPerBlock;
        int RegistersPerBlock;
        int WarpSize;
        std::size_t MemPitch;
        int MaxThreadsPerBlock;
        int MaxThreadsDim[3];
        int MaxGridSize[3];
        std::size_t TotalConstMem;
        int Major;
        int Minor;
        int ClockRate;
        std::size_t TextureAlignment;
        int DeviceOverlap;
        int MultiProcessorCount;
        int KernelExecTimeoutEnabled;
        int Integrated;
        int CanMapHostMemory;
        int ComputeMode;
        int ConcurrentKernels;
        int ECCEnabled;
        int PCIBusID;
        int PCIDeviceID;
        int MemoryClockRate;
        int MemoryBusWidth;
        int MaxThreadsPerMultiprocessor;
    };
} //namespace cuda

#endif //MASTERS_DEVICE_INFO_H
