//
// Created by madaeu on 4/28/21.
//

#include "decoder_network.h"
#include "tfwrap.h"

#include "json/json.h"

#include "fstream"
#include "cstring"

namespace msc
{
    struct DecoderNetwork::Impl
    {
        std::unique_ptr<tf::GraphEvaluator> graph_;
        tf::Tensor imageTensor_;
        tf::Tensor codeTensor_;
    };

    DecoderNetwork::DecoderNetwork(const NetworkConfiguration &configuration)
    : configuration_(configuration), impl_(std::make_unique<Impl>())
    {
        tf::SessionOptions options;
        options.gpu_memory_allow_growht(true);
        impl_->graph_ = std::make_unique<tf::GraphEvaluator>(configuration.graphPath, options);

        impl_->imageTensor_ = tf::Tensor::fromDimensions<float>({1, (long long)configuration_.inputHeight, (long long)configuration_.inputWidth, 1});
        impl_->codeTensor_ = tf::Tensor::fromDimensions<float>({1, (long long)configuration_.codeSize});

    }

    DecoderNetwork::~DecoderNetwork()
    {

    }

    void DecoderNetwork::decode(const vc::Buffer2DView<float> &image, const Eigen::MatrixXf &code,
                                vc::RuntimeBufferPyramidView<float> *proximityOut,
                                vc::RuntimeBufferPyramidView<float> *uncertaintyOut,
                                vc::RuntimeBufferPyramidView<float> *jacobianOut)
    {
        impl_->imageTensor_.copyFrom(image.ptr());
        impl_->codeTensor_.copyFrom(code.data());

        std::vector<std::pair<std::string, tf::Tensor>> feedDict = { {configuration_.inputImageName, impl_->imageTensor_},
                                                                     {configuration_.inputCodeName, impl_->codeTensor_}};

        std::vector<std::string> fetches;

        if (proximityOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthEstimateName[i]);
            }
        }

        if(uncertaintyOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthUncertaintyName[i]);
            }
        }

        if(jacobianOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthJacobianName[i]);
            }
        }

        auto outputs = impl_->graph_->run(feedDict, fetches);

        if (proximityOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = proximityOut->operator[](i).width();
                std::size_t height = proximityOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                proximityOut->operator[](i).copyFrom(view);
            }
            outputs.erase(outputs.begin(), outputs.begin() + configuration_.pyramidLevels);
        }

        if (uncertaintyOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = uncertaintyOut->operator[](i).width();
                std::size_t height = uncertaintyOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                uncertaintyOut->operator[](i).copyFrom(view);
            }
            outputs.erase(outputs.begin(), outputs.begin() + configuration_.pyramidLevels);
        }

        if (jacobianOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = jacobianOut->operator[](i).width();
                std::size_t height = jacobianOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                jacobianOut->operator[](i).copyFrom(view);
            }
        }
    }

    void DecoderNetwork::predictAndDecode(const vc::Buffer2DView<float> &image, const Eigen::MatrixXf &code,
                                          Eigen::MatrixXf *predictedCode,
                                          vc::RuntimeBufferPyramidView<float> *proximityOut,
                                          vc::RuntimeBufferPyramidView<float> *uncertaintyOut,
                                          vc::RuntimeBufferPyramidView<float> *jacobianOut)
    {
        impl_->imageTensor_.copyFrom(image.ptr());
        impl_->codeTensor_.copyFrom(code.data());

        std::vector<std::pair<std::string, tf::Tensor>> feedDict = { {configuration_.inputImageName, impl_->imageTensor_},
                                                                     {configuration_.inputCodeName, impl_->codeTensor_}};

        std::vector<std::string> fetches;

        if(predictedCode)
        {
            fetches.push_back(configuration_.codePredictionName);
        }

        if (proximityOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthEstimateName[i]);
            }
        }

        if(uncertaintyOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthUncertaintyName[i]);
            }
        }

        if(jacobianOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                fetches.push_back(configuration_.depthJacobianName[i]);
            }
        }

        auto outputs = impl_->graph_->run(feedDict, fetches);

        if (predictedCode)
        {
            std::memcpy(predictedCode->data(), outputs[0].data_ptr(), outputs[0].bytes());
            outputs.erase(outputs.begin(), outputs.begin() + 1);
        }

        if (proximityOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = proximityOut->operator[](i).width();
                std::size_t height = proximityOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                proximityOut->operator[](i).copyFrom(view);
            }
            outputs.erase(outputs.begin(), outputs.begin() + configuration_.pyramidLevels);
        }

        if (uncertaintyOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = uncertaintyOut->operator[](i).width();
                std::size_t height = uncertaintyOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                uncertaintyOut->operator[](i).copyFrom(view);
            }
            outputs.erase(outputs.begin(), outputs.begin() + configuration_.pyramidLevels);
        }

        if (jacobianOut)
        {
            for (uint i = 0; i < configuration_.pyramidLevels; ++i)
            {
                std::size_t width = jacobianOut->operator[](i).width();
                std::size_t height = jacobianOut->operator[](i).height();
                vc::Buffer2DView<float, vc::TargetHost> view(outputs[i].data_ptr(), width, height);
                jacobianOut->operator[](i).copyFrom(view);
            }
        }
    }

    DecoderNetwork::NetworkConfiguration loadJsonNetworkConfiguration(const std::string& configurationPath)
    {
        std::ifstream ifs(configurationPath);
        if(ifs.fail())
            throw std::runtime_error("Could not load configuration file: " + configurationPath);

        Json::Value root;
        Json::parseFromStream(Json::CharReaderBuilder(), ifs, &root, nullptr);
        std::string graphPath = root["graph_path"].asString();
        if(graphPath[0] != '/')
        {
            std::string configDir(configurationPath);
            configDir.erase(configDir.rfind("/"));
            graphPath = configDir + "/" + graphPath;
        }

        DecoderNetwork::NetworkConfiguration configuration;
        configuration.graphPath = graphPath;
        configuration.inputWidth = root["input_width"].asUInt();
        configuration.inputHeight = root["input_height"].asUInt();
        configuration.pyramidLevels = root["pyramid_levels"].asUInt();
        configuration.codeSize = root["code_size"].asUInt();
        configuration.grayscale = root["grayscale"].asBool();
        configuration.avgDepth = root["avg_depth"].asDouble();
        const auto& inputNames = root["input_names"];
        auto cut_after_colon = [] (const std::string& str) -> std::string {
            return str.substr(0, str.find_last_of(":"));
        };
        configuration.inputImageName = cut_after_colon(inputNames["image"].asString());
        configuration.inputCodeName = cut_after_colon(inputNames["code"].asString());

        auto node2stdvec = [cut_after_colon] (const Json::Value& val) -> std::vector<std::string> {
            std::vector<std::string> vec;
            for(Json::Value::ArrayIndex i = 0; i < val.size(); ++i){
                vec.push_back(cut_after_colon(val[i].asString()));
            }
            return vec;
        };

        const auto& outputNames = root["output_names"];
        configuration.depthEstimateName = node2stdvec(outputNames["depth_est"]);
        configuration.depthUncertaintyName = node2stdvec(outputNames["depth_stdev"]);
        configuration.depthJacobianName = node2stdvec(outputNames["depth_jac"]);

        if(root.isMember("depth_pred") && root["depth_pred"].asBool())
        {
            configuration.predictDepth = true;
            configuration.depthPredictionName = node2stdvec(outputNames["depth_pred"]);
            configuration.codePredictionName = node2stdvec(outputNames["code_pred"])[0];
        }
        else
        {
            configuration.predictDepth = false;
        }

        const auto& camera = root["camera"];
        configuration.camera.fx = camera["fx"].asDouble();
        configuration.camera.fy = camera["fy"].asDouble();
        configuration.camera.u0 = camera["u0"].asDouble();
        configuration.camera.v0 = camera["v0"].asDouble();

        return configuration;

    }

} // namespace msc
