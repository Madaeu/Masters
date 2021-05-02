//
// Created by madaeu on 4/28/21.
//

#ifndef MASTERS_TFWRAP_H
#define MASTERS_TFWRAP_H

#include <memory>
#include <vector>
#include <cstring>
#include <cassert>
#include <fstream>

#include <tensorflow/c/c_api.h>
#include <tensorflow/c/c_api_experimental.h>

namespace tf
{
    template<typename T>
    struct tf_type;

#define DEFINE_TF_TYPE(T1,T2) \
template<> \
struct tf_type<T1>{static constexpr TF_DataType value = T2; }; \

    DEFINE_TF_TYPE(float, TF_FLOAT);
    DEFINE_TF_TYPE(double, TF_DOUBLE);

    class Tensor
    {
    public:
        using TensorDims = std::vector<std::int64_t>;

        Tensor() : bytes_(0) {}

        Tensor(TF_DataType type, const TensorDims& dimensions, std::size_t bytes)
        : type_(type), dimensions_(dimensions), bytes_(bytes)
        {
            tensor_ = std::shared_ptr<TF_Tensor>(TF_AllocateTensor(type, dimensions.data(),
                                                                    static_cast<int>(dimensions.size()),
                                                                    bytes_), TF_DeleteTensor);
        }

        Tensor(TF_Tensor* ptr)
        : type_(TF_TensorType(ptr)), bytes_(TF_TensorByteSize(ptr))
        {
            tensor_ = std::shared_ptr<TF_Tensor>(ptr, TF_DeleteTensor);
            dimensions_ = TensorDims(TF_NumDims(ptr));
            for (std::size_t i = 0; i < dimensions_.size(); ++i)
            {
                dimensions_[i] = TF_Dim(ptr, i);
            }
        }

        void copyFrom(const void* data)
        {
            std::memcpy(data_ptr(), data, bytes_);
        }

        template<typename T>
        static Tensor fromData(void* data, const TensorDims& dimensions)
        {
            assert(dimensions.size());
            std::size_t size = 1;
            for(auto& d: dimensions)
            {
                size *= d;
            }
            Tensor tensor(tf_type<T>::value, dimensions, size*sizeof(T));
            tensor.copyFrom(data);
            return tensor;
        }

        template <typename T>
        static Tensor fromDimensions(const TensorDims& dimensions)
        {
            assert(dimensions.size());
            std::size_t size = 1;
            for (auto& d: dimensions)
            {
                size *= d;
            }
            return Tensor(tf_type<T>::value, dimensions, size * sizeof(T));
        }

        TF_Tensor* tensor() { return tensor_.get(); }
        std::size_t bytes() { return bytes_; }
        void* data_ptr() { return TF_TensorData(tensor_.get()); }

    private:
        TF_DataType type_;
        TensorDims dimensions_;
        std::size_t bytes_;
        std::shared_ptr<TF_Tensor> tensor_;
    };

    class Status
    {
    public:
        Status(): ptr_(TF_NewStatus(), TF_DeleteStatus) { }
        bool ok() { return TF_GetCode(ptr()) == TF_OK; }
        TF_Status* ptr() { return ptr_.get(); }

        void failOnError(std::string message)
        {
            if(!ok())
            {
                throw std::runtime_error(message + ": " + std::string(TF_Message(ptr_.get())));
            }
        }

    private:
        std::shared_ptr<TF_Status> ptr_;
    };

    static TF_Buffer* readBufferFromFile(const char* file)
    {
        std::ifstream f(file, std::ios::binary);
        if(f.fail() || !f.is_open())
        {
            return nullptr;
        }
        if(f.seekg(0, std::ios::end).fail())
        {
            return nullptr;
        }

        auto fsize = f.tellg();

        if (f.seekg(0, std::ios::beg).fail())
        {
            return nullptr;
        }

        if(fsize <= 0)
        {
            return nullptr;
        }

        auto data = static_cast<char*>(std::malloc(fsize));
        if ( f.read(data, fsize).fail())
        {
            return nullptr;
        }

        f.close();

        auto buffer = TF_NewBuffer();
        buffer->data = data;
        buffer->length = fsize;
        return buffer;
    }

    class Graph
    {
    public:
        Graph(std::string graphdefPath) : path_(graphdefPath)
        {
            graph_ = std::shared_ptr<TF_Graph>(TF_NewGraph(), TF_DeleteGraph);
            importGraphDef(graphdefPath);
        }

        void importGraphDef(std::string path)
        {
            auto buffer = readBufferFromFile(path.c_str());
            if (buffer == nullptr)
            {
                throw std::runtime_error("Can't read buffer from file: " + path);
            }
            Status status;
            auto options = TF_NewImportGraphDefOptions();
            TF_GraphImportGraphDef(graph_.get(), buffer, options, status.ptr());
            TF_DeleteImportGraphDefOptions(options);
            TF_DeleteBuffer(buffer);

            if (!status.ok())
            {
                throw std::runtime_error( "Failed importing graph from: " + path);
            }
        }

        TF_Output getOpByName(std::string opName)
        {
            TF_Output op = TF_Output{TF_GraphOperationByName(graph_.get(), opName.c_str()), 0};
            if (op.oper == nullptr)
            {
                throw std::runtime_error( "Can't find op in graph: " + opName);
            }
            return op;
        }

        TF_Graph* ptr() { return graph_.get(); }
        std::string path() { return path_; }

    private:
        std::shared_ptr<TF_Graph> graph_;
        std::string path_;
    };

    class SessionOptions
    {
    public:
        SessionOptions() : options_(TF_NewSessionOptions(), TF_DeleteSessionOptions) { }
        void enable_xla_compilation(bool value)
        {
            enable_xla_compilation_ = value;
            setOptions();
        }

        void gpu_memory_allow_growht(bool value)
        {
            gpu_memory_allow_growth_ = value;
            setOptions();
        }

        void num_cpu_devices(int n)
        {
            num_cpu_devices_ = n;
            setOptions();
        }

        void setOptions()
        {
            TF_Buffer* buffer = TF_CreateConfig(enable_xla_compilation_, gpu_memory_allow_growth_, num_cpu_devices_);
            TF_SetConfig(options_.get(), buffer->data, buffer->length, status_.ptr());
            status_.failOnError("Failed to set options!");
            TF_DeleteBuffer(buffer);
        }

        TF_SessionOptions* ptr() { return options_.get(); }

    private:
        Status status_;
        std::shared_ptr<TF_SessionOptions> options_;
        bool enable_xla_compilation_ = false;
        bool gpu_memory_allow_growth_ = false;
        int num_cpu_devices_ = 1;
    };

    class Session
    {
    public:
        Session(Graph& graph, SessionOptions& sessionOptions)
        {
            session_ = std::shared_ptr<TF_Session>(TF_NewSession(graph.ptr(), sessionOptions.ptr(),status_.ptr()), Session::closeAndDelete);
            status_.failOnError("Failed to create Session");
        }

        static void closeAndDelete(TF_Session* session)
        {
            Status status;
            TF_CloseSession(session, status.ptr());
            status.failOnError("Failed to close Session");
            TF_DeleteSession(session, status.ptr());
            status.failOnError("Failed to delete Session");
        }

        TF_Session* ptr() { return session_.get(); }

    private:
        Status status_;
        std::shared_ptr<TF_Session> session_;
    };

    class GraphEvaluator
    {
    public:
        using TensorDict = std::vector<std::pair<std::string, tf::Tensor>>;

        GraphEvaluator(std::string graphPath, tf::SessionOptions& options)
        : graph_(graphPath), session_(graph_, options) {}

        std::vector<tf::Tensor> run(std::vector<std::pair<std::string, tf::Tensor>>& feedDict, std::vector<std::string>& fetches)
        {
            std::vector<TF_Output> inputOps;
            std::vector<TF_Tensor*> inputTensors;
            for(auto& kv: feedDict)
            {
                inputOps.push_back(graph_.getOpByName(kv.first));
                inputTensors.push_back(kv.second.tensor());
            }

            std::vector<TF_Output> outputOps;
            std::vector<TF_Tensor*> outputTensors(fetches.size());
            for(auto& name: fetches)
            {
                outputOps.push_back(graph_.getOpByName(name));
            }

            TF_SessionRun(session_.ptr(),
                          nullptr,
                          &inputOps[0], &inputTensors[0], feedDict.size(),
                          &outputOps[0], &outputTensors[0], fetches.size(),
                          nullptr, 0,
                          nullptr,
                          status_.ptr());

            status_.failOnError("Running session failed!");

            std::vector<tf::Tensor> output;
            for (auto& tensor: outputTensors)
            {
                output.push_back(tf::Tensor(tensor));
            }
            return output;
        }

    private:
        tf::Status status_;
        tf::Graph graph_;
        tf::Session session_;
    };

} // namespace tf

#endif //MASTERS_TFWRAP_H
