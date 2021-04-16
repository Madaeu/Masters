//
// Created by madaeu on 4/6/21.
//

#include "cuda_image_proc.h"
#include "launch_utilities.h"
#include "kernel_utilities.h"

#include <VisionCore/LaunchUtils.hpp>
#include <VisionCore/Buffers/Reductions.hpp>
#include <Eigen/Core>

// CUDA Sobel Gradients
template <typename Derived>
static inline void SetSobelCoefficients(Eigen::DenseBase<Derived>& mx,
                                        Eigen::DenseBase<Derived>& my)
{
    typedef typename Eigen::DenseBase<Derived>::Scalar Scalar;

    // NOTE: Canonical Sobel kernels
    mx << Scalar(-1.0), Scalar(0.0), Scalar(1.0),
            Scalar(-2.0), Scalar(0.0), Scalar(2.0),
            Scalar(-1.0), Scalar(0.0), Scalar(1.0);

    my << Scalar(-1.0), Scalar(-2.0), Scalar(-1.0),
            Scalar(0.0),  Scalar(0.0),  Scalar(0.0),
            Scalar(1.0),  Scalar(2.0),  Scalar(1.0);
}

struct SobelCoeffs
{
    float X[9];
    float Y[9];
};
__constant__ SobelCoeffs SC;

template<typename PixelT, typename TG>
__global__ void kernel_sobel_gradients(const vc::Buffer2DView<PixelT,vc::TargetDeviceCUDA> img,
                                       vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA> grad)
{
    typedef Eigen::Matrix<TG,1,2> DerivT;

    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    Eigen::Map<const Eigen::Matrix<float,3,3>> mx(SC.X);
    Eigen::Map<const Eigen::Matrix<float,3,3>> my(SC.Y);

    if(img.inBounds(x,y) && grad.inBounds(x,y))
    {
        DerivT& out = grad(x,y);

        TG sum_dx = 0;
        TG sum_dy = 0;

        for(int py = -1 ; py <= 1 ; ++py)
        {
            for(int px = -1 ; px <= 1 ; ++px)
            {
                const PixelT pix = img.getWithClampedRange((int)x + px,(int)y + py);
                const float& kv_dx = mx(1 + py, 1 + px);
                const float& kv_dy = my(1 + py, 1 + px);
                sum_dx += (pix * kv_dx);
                sum_dy += (pix * kv_dy);
            }
        }

        out << sum_dx , sum_dy;
        out /= 8;
    }
}

template<typename T, typename TG>
void SobelGradients(const vc::Buffer2DView<T,vc::TargetDeviceCUDA>& img,
                    vc::Buffer2DView<Eigen::Matrix<TG,1,2>,vc::TargetDeviceCUDA>& grad)
{
    // get a gaussian kernel in the gpu
    SobelCoeffs coeffs;
    Eigen::Map<Eigen::Matrix<float,3,3>> mx(coeffs.X);
    Eigen::Map<Eigen::Matrix<float,3,3>> my(coeffs.Y);
    SetSobelCoefficients(mx, my);
    cudaMemcpyToSymbol(SC, &coeffs, sizeof(SobelCoeffs));
    cudaCheckLastError("cudaMemcpyToSymbol failed");

    // calculate blocks and threads
    dim3 threads, blocks;
    vc::InitDimFromBufferOver(threads, blocks, grad);

    // run kernel
    kernel_sobel_gradients<<<blocks,threads>>>(img, grad);
    cudaCheckLastError("Kernel launch failed (kernel_sobel_gradients)");
}
/*
template <typename Derived>
static inline void setSobelCoefficients(Eigen::DenseBase<Derived>& sobelX,
                                        Eigen::DenseBase<Derived>& sobelY)
{
    using Scalar = typename Eigen::DenseBase<Derived>::Scalar;

    sobelX << Scalar(-1.0), Scalar(0.0), Scalar(1.0),
            Scalar(-2.0), Scalar(0.0), Scalar(2.0),
            Scalar(-1.0), Scalar(0.0), Scalar(1.0);

    sobelY << Scalar(-1.0), Scalar(-2.0), Scalar(-1.0),
            Scalar(0.0),  Scalar(0.0),  Scalar(0.0),
            Scalar(1.0),  Scalar(2.0),  Scalar(1.0);
}

struct SobelCoefficients
{
    float X[9];
    float Y[9];
};

__constant__ SobelCoefficients SC;

template <typename PixelT, typename GradientT>
__global__ void kernel_sobel_gradients(const vc::Buffer2DView<PixelT, vc::TargetDeviceCUDA>& image,
                                       vc::Buffer2DView<Eigen::Matrix<GradientT, 1,2>, vc::TargetDeviceCUDA>& gradients)
{
    using DerivedT = Eigen::Matrix<GradientT, 1,2>;

    const std::size_t x = blockIdx.x*blockDim.x+threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y+threadIdx.y;

    Eigen::Map<const Eigen::Matrix<float, 3,3>> mx(SC.X);
    Eigen::Map<const Eigen::Matrix<float, 3,3>> my(SC.Y);

    if( image.inBounds(x,y) && gradients.inBounds(x,y))
    {
        DerivedT& output = gradients(x,y);

        GradientT sum_dx = 0;
        GradientT sum_dy = 0;
        for (int py = -1; py <= 1; ++py)
        {
            for (int px = -1; px <= 1; ++px)
            {
                const PixelT pixel = image.getWithClampedRange((int)x +px, (int)y+py);
                const float& kv_dx = mx(1 + py, 1 + px);
                const float& kv_dy = my(1 + py, 1 + px);
                sum_dx += (pixel*kv_dx);
                sum_dy += (pixel*kv_dy);
            }
        }
        output << sum_dx, sum_dy;
        output /= 8;
    }
}

template <typename T, typename GradientT>
void sobelGradients(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& input,
                    vc::Buffer2DView<Eigen::Matrix<GradientT, 1,2>, vc::TargetDeviceCUDA>& output)
{
    SobelCoefficients coefficients;
    Eigen::Map<Eigen::Matrix<float, 3,3>> mx(coefficients.X);
    Eigen::Map<Eigen::Matrix<float, 3,3>> my(coefficients.Y);
    setSobelCoefficients(mx, my);
    cudaMemcpyToSymbol(SC, &coefficients, sizeof(SobelCoefficients));
    cudaCheckLastError("cudaMemcpyToSymbol failed");

    dim3 threads, blocks;
    vc::InitDimFromBufferOver(threads, blocks, output);

    kernel_sobel_gradients<<<blocks,threads>>>(input, output);
    cudaCheckLastError("Kernel launch failed (kernel_sobel_gradients");
}
*/

// CUDA Gaussian Blur Down
template <typename Derived>
static inline void setGaussCoefficients(Eigen::DenseBase<Derived>& gaussKernel)
{
    using  Scalar = typename Eigen::DenseBase<Derived>::Scalar;

    gaussKernel << Scalar(1.0), Scalar(4.0),  Scalar(6.0),  Scalar(4.0),  Scalar(1.0),
            Scalar(4.0), Scalar(16.0), Scalar(24.0), Scalar(16.0), Scalar(4.0),
            Scalar(6.0), Scalar(24.0), Scalar(36.0), Scalar(24.0), Scalar(6.0),
            Scalar(4.0), Scalar(16.0), Scalar(24.0), Scalar(16.0), Scalar(4.0),
            Scalar(1.0), Scalar(4.0),  Scalar(6.0),  Scalar(4.0),  Scalar(1.0);
}


__constant__ float gaussCoefficients[25];

template<typename Scalar>
__global__ void kernel_gaussian_blur_down(const vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> input,
                                          vc::Buffer2DView<Scalar, vc::TargetDeviceCUDA> output)
{
    // current point
    const std::size_t x = blockIdx.x*blockDim.x + threadIdx.x;
    const std::size_t y = blockIdx.y*blockDim.y + threadIdx.y;

    Eigen::Map<const Eigen::Matrix<float,5,5>> kernel(gaussCoefficients);
    static constexpr int kDim = 5;

    if(output.inBounds(x,y))
    {
        Scalar sum = 0.0;
        Scalar wall = 0.0;

        for(int py = 0; py < kDim; ++py)
        {
            for(int px = 0; px < kDim; ++px)
            {
                const int nx = clamp((int) (2 * x + px - kDim / 2), 0, (int)input.width() - 1);
                const int ny = clamp((int) (2 * y + py - kDim / 2), 0, (int)input.height() - 1);
                const Scalar pix = input(nx,ny);
                sum += (pix * kernel(px,py));
                wall += kernel(px,py);
            }
        }

        output(x,y) = sum / wall;
    }
}

template <typename T>
void gaussianBlurDown(const vc::Buffer2DView<T, vc::TargetDeviceCUDA>& input,
                      vc::Buffer2DView<T, vc::TargetDeviceCUDA>& output)
{
    // get a gaussian kernel in the gpu
    float coefficients[25];
    Eigen::Map<Eigen::Matrix<float,5,5>> gaussKernel(coefficients);
    setGaussCoefficients(gaussKernel);
    cudaMemcpyToSymbol(gaussCoefficients, &coefficients, sizeof(coefficients));
    cudaCheckLastError("cudaMemcpyToSymbol failed");

    // calculate blocks and threads
    dim3 threads, blocks;
    vc::InitDimFromBufferOver(threads, blocks, output);

    // run kernel
    kernel_gaussian_blur_down<<<blocks,threads>>>(input, output);
    cudaCheckLastError("Kernel launch failed (kernel_gaussian_blur_down)");
}

//Explicit instantiation
template void gaussianBlurDown(const vc::Buffer2DView<float, vc::TargetDeviceCUDA>& input,
                               vc::Buffer2DView<float, vc::TargetDeviceCUDA>& output);
/*
template void sobelGradients(const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& input,
                             vc::Buffer2DView<Eigen::Matrix<float,1,2>,vc::TargetDeviceCUDA>& output);*/

template void SobelGradients(const vc::Buffer2DView<float,vc::TargetDeviceCUDA>& img,
                             vc::Buffer2DView<Eigen::Matrix<float,1,2>,vc::TargetDeviceCUDA>& grad);