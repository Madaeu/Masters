//
// Created by madaeu on 4/28/21.
//

#ifndef MASTERS_DECODER_NETWORK_H
#define MASTERS_DECODER_NETWORK_H

#include "VisionCore/Buffers/Buffer2D.hpp"
#include "VisionCore/Buffers/Buffer1D.hpp"
#include "VisionCore/Buffers/BufferPyramid.hpp"

#include <memory>

namespace msc
{
    class DecoderNetwork
    {
    public:
        using Ptr = std::shared_ptr<DecoderNetwork>;

        struct NetworkConfiguration
        {
            struct Camera
            {
                double fx, fy, u0, v0;
            };

            Camera camera;

            std::string graphPath;
            std::size_t inputWidth;
            std::size_t inputHeight;
            std::size_t pyramidLevels;
            std::size_t codeSize;
            bool grayscale;
            double avgDepth;

            bool predictDepth;

            std::string inputImageName;
            std::string inputCodeName;

            std::string codePredictionName;
            std::vector<std::string> depthEstimateName;
            std::vector<std::string> depthJacobianName;
            std::vector<std::string> depthUncertaintyName;
            std::vector<std::string> depthPredictionName;
        };

        DecoderNetwork(const NetworkConfiguration& configuration);
        ~DecoderNetwork();

        void decode(const vc::Buffer2DView<float, vc::TargetHost>& image,
                    const Eigen::MatrixXf& code,
                    vc::RuntimeBufferPyramidView<float, vc::TargetHost>* proximityOut = nullptr,
                    vc::RuntimeBufferPyramidView<float, vc::TargetHost>* uncertaintyOut = nullptr,
                    vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jacobianOut = nullptr);

        void predictAndDecode(const vc::Buffer2DView<float, vc::TargetHost>& image,
                              const Eigen::MatrixXf& code,
                              Eigen::MatrixXf* predictedCode,
                              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* proximityOut = nullptr,
                              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* uncertaintyOut = nullptr,
                              vc::RuntimeBufferPyramidView<float, vc::TargetHost>* jacobianOut = nullptr);

    private:
        NetworkConfiguration configuration_;
        vc::Buffer2DView<float, vc::TargetHost> imageView;
        vc::Buffer1DView<float, vc::TargetHost> codeView;

        struct Impl;
        std::unique_ptr<Impl> impl_;

        vc::Buffer2DView<float, vc::TargetHost> imageView_;
        vc::Buffer1DView<float, vc::TargetHost> codeView_;
    };

    DecoderNetwork::NetworkConfiguration loadJsonNetworkConfiguration(const std::string& configurationPath);

} // namespace msc

#endif //MASTERS_DECODER_NETWORK_H
