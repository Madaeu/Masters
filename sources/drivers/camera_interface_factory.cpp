//
// Created by madaeu on 2/24/21.
//

#include "camera_interface_factory.h"

std::shared_ptr<CameraInterfaceFactory> CameraInterfaceFactory::ptr_;

std::shared_ptr<CameraInterfaceFactory> CameraInterfaceFactory::get()
{
    if(!ptr_){
        ptr_ = std::make_shared<CameraInterfaceFactory>();
    }
    return ptr_;
}

std::unique_ptr<CameraInterface> CameraInterfaceFactory::getInterfaceFromUrl(const std::string &url)
{
    auto url_parts = partitionUrl(url);
    std::string prefix = url_parts[0];
    std::string remainder = url_parts[1];

    if (factoryMap_.find(prefix) == factoryMap_.end())
    {
        std::stringstream ss;
        ss << "Interface for prefix" << prefix << " not registered.\n";
        ss << "Supported interfaces: ";
        for(auto& pref : supportedInterfaces_){
            ss << pref << ", ";
        }
        throw MalformedUrlException(url, ss.str());
    }
    return factoryMap_[prefix]->fromUrlParams(remainder);
}

std::string CameraInterfaceFactory::getUrlHelp()
{
    std::stringstream ss;
    ss << "Supported URLs: \n";
    for (auto& url_form: urlForms_) {
        ss << url_form << "\n";
    }
    return ss.str();
}

std::vector<std::string> CameraInterfaceFactory::partitionUrl(const std::string &url)
{
    auto pos = url.find(prefixTag_);
    if(pos > url.length())
        throw MalformedUrlException(url, "Missing tag: " + prefixTag_ );
    std::string prefix = url.substr(0, pos);
    std::string remainder = url.substr(pos + prefixTag_.length());
    return {prefix, remainder};
}
