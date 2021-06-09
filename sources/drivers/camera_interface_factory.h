//
// Created by madaeu on 2/24/21.
//

#ifndef MASTERS_CAMERA_INTERFACE_FACTORY_H
#define MASTERS_CAMERA_INTERFACE_FACTORY_H

#include "camera_interface.h"

#include "opencv2/opencv.hpp"

#include <functional>
#include <memory>
#include <map>
#include <vector>

class MalformedUrlException: public std::runtime_error
{
public:
    MalformedUrlException(const std::string& pattern, const std::string& reason)
    : std::runtime_error( "Invalid source URL:" + pattern + ": " + reason ) {}
};

class SpecificInterfaceFactory
{
public:
    virtual std::unique_ptr<CameraInterface> fromUrlParams(const std::string& url_params) = 0;
    virtual std::string getUrlPattern(const std::string& prefix_tag) = 0;
    virtual std::string getPrefix() = 0;
};

class CameraInterfaceFactory
{
public:
    using FactoryMapT = std::map<std::string, std::shared_ptr<SpecificInterfaceFactory>>;

    std::unique_ptr<CameraInterface> getInterfaceFromUrl(const std::string& url);

    template<typename T>
    void registerInterface()
    {
        auto factoryObject = std::make_shared<T>();
        typename FactoryMapT::value_type pair(factoryObject->getPrefix(), factoryObject);
        factoryMap_.insert(pair);
        urlForms_.push_back(factoryObject->getUrlPattern(prefixTag_));
        supportedInterfaces_.push_back(factoryObject->getPrefix());
    }

    std::string getUrlHelp();
    static std::shared_ptr<CameraInterfaceFactory> get();

private:
    std::vector<std::string> partitionUrl(const std::string& url);

    FactoryMapT factoryMap_;
    std::vector<std::string> supportedInterfaces_;
    std::vector<std::string> urlForms_;

    const std::string prefixTag_ = "://";
    static std::shared_ptr<CameraInterfaceFactory> ptr_;

};

template <typename T>
struct InterfaceRegistrar{
    InterfaceRegistrar(){
        CameraInterfaceFactory::get()->template registerInterface<T>();
    }
};
#endif //MASTERS_CAMERA_INTERFACE_FACTORY_H
