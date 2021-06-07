//
// Created by madaeu on 4/20/21.
//

#ifndef MASTERS_SYNCED_PYRAMID_H
#define MASTERS_SYNCED_PYRAMID_H

#include "VisionCore/Buffers/BufferPyramid.hpp"

namespace msc
{
    template< typename Scalar>
    class SynchronizedBufferPyramid
    {
    public:
        using CPUBuffer = vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetHost>;
        using GPUBuffer = vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA>;
        using Ptr = std::shared_ptr<SynchronizedBufferPyramid<Scalar>>;

        SynchronizedBufferPyramid() = delete;

        SynchronizedBufferPyramid(std::size_t pyramidLevels, std::size_t width, std::size_t height)
        : cpuModified_(false), gpuModified_(false), pyramidLevels_(pyramidLevels), width_(width), height_(height)
        {}

        virtual ~SynchronizedBufferPyramid() {}

        SynchronizedBufferPyramid(const SynchronizedBufferPyramid<Scalar>& other)
        {
            pyramidLevels_ = other.pyramidLevels_;
            width_ = other.width_;
            height_ = other.height_;
            codeSize_ = other.codeSize_;
            cpuModified_ = other.cpuModified_;
            gpuModified_ = other.gpuModified_;

            if (other.cpuBuffer_)
            {
                cpuBuffer_ = std::make_shared<CPUBuffer>(pyramidLevels_, width_, height_);
                cpuBuffer_->copyFrom(*other.cpuBuffer_);
            }

            if (other.gpuBuffer_)
            {
                gpuBuffer_ = std::make_shared<GPUBuffer>(pyramidLevels_, width_, height_);
                gpuBuffer_->copyFrom(*other.gpuBuffer_);
            }
        }

        std::shared_ptr<const GPUBuffer> getGPU() const
        {
            synchronizeGPU();
            return gpuBuffer_;
        }

        std::shared_ptr<const CPUBuffer> getCPU() const
        {
            synchronizeCPU();
            return cpuBuffer_;
        }

        std::shared_ptr<CPUBuffer> getCPUMutable()
        {
            synchronizeCPU();
            flagNewDataCPU();
            return cpuBuffer_;
        }

        std::shared_ptr<GPUBuffer> getGPUMutable()
        {
            synchronizeGPU();
            flagNewDataGPU();
            return gpuBuffer_;
        }

        const typename CPUBuffer::ViewType& getLevelCPU(int level) const
        {
            return getCPU()->operator[](level);
        }

        typename CPUBuffer::ViewType& getLevelCPU(int level)
        {
            return getCPUMutable()->operator[](level);
        }

        const typename GPUBuffer::ViewType& getLevelGPU(int level) const
        {
            return getGPU()->operator[](level);
        }

        typename GPUBuffer::ViewType& getLevelGPU(int level)
        {
            return getGPUMutable()->operator[](level);
        }

        void flagNewDataCPU()
        {
            cpuModified_ = true;
            checkDivergence();
        }

        void flagNewDataGPU()
        {
            gpuModified_ = true;
            checkDivergence();
        }

        bool isSynchronized()
        {
            return !gpuModified_ && !cpuModified_;
        }

        void unloadCPU()
        {
            cpuBuffer_.reset();
            flagNewDataCPU();
        }

        void unloadGPU()
        {
            gpuBuffer_.reset();
            flagNewDataGPU();
        }

        void checkDivergence() const
        {
            if (gpuModified_ && cpuModified_)
                std::cout << "CPU and GPU are not synchronized! \n";
        }

        void synchronizeCPU() const
        {
            if(!cpuBuffer_)
            {
                cpuBuffer_ = std::make_shared<CPUBuffer>(pyramidLevels_, width_, height_);
            }

            checkDivergence();

            if(gpuModified_)
            {
                cpuBuffer_->copyFrom(*gpuBuffer_);
            }

            gpuModified_ = false;
        }

        void synchronizeGPU() const
        {
            if(!gpuBuffer_)
            {
                gpuBuffer_ = std::make_shared<GPUBuffer>(pyramidLevels_, width_, height_);
            }

            checkDivergence();

            if(cpuModified_)
            {
                gpuBuffer_->copyFrom(*cpuBuffer_);
            }

            cpuModified_ = false;
        }

        std::size_t width() const { return width_; }
        std::size_t height() const { return height_; }
        std::size_t levels() const { return pyramidLevels_; }
        std::size_t codeSize() const { return codeSize_; }
        std::size_t area() const { return width()*height(); }

    private:
        mutable std::shared_ptr<CPUBuffer> cpuBuffer_;
        mutable std::shared_ptr<GPUBuffer> gpuBuffer_;

        mutable bool cpuModified_;
        mutable bool gpuModified_;

        std::size_t pyramidLevels_;
        std::size_t width_;
        std::size_t height_;
        std::size_t codeSize_;
    };
} //namespace msc

#endif //MASTERS_SYNCED_PYRAMID_H
