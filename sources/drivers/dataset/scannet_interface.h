//
// Created by madaeu on 5/6/21.
//

#ifndef MASTERS_SCANNET_INTERFACE_H
#define MASTERS_SCANNET_INTERFACE_H

#include "dataset_interface.h"
#include "camera_interface_factory.h"

class ScanNetInterface: public DatasetInterface
{
public:
    ScanNetInterface(std::string sequencePath);
    virtual ~ScanNetInterface();

    virtual void grabFrames(double& timestamp, cv::Mat* image, cv::Mat* depth = nullptr) override;
    virtual std::vector<DatasetFrame> getAll() override;

    virtual bool supportsDepth() override { return hasDepth_; }
    virtual bool hasIntrinsics() override { return true; }
    virtual bool hasPoses() override { return true; };
    virtual bool hasMore() override { return currentFrame_ < (int)imageFiles_.size()-1; }
    virtual std::vector<Sophus::SE3f> getPoses() override { return poses_; }

    virtual msc::PinholeCamera<float> getIntrinsics() override;

private:
    void loadFrame(int i, double& timestamp, cv::Mat* image, cv::Mat* depth, Sophus::SE3f& pose);
    msc::PinholeCamera<float> loadIntrinsics(std::string sequenceDirectory, int width, int height);
    std::vector<Sophus::SE3f> loadPoses();
    std::size_t probeNumberOfFrames(std::string directory);

    std::string sequencePath_;
    std::vector<std::string> imageFiles_;
    std::vector<std::string> depthFiles_;
    std::vector<std::string> poseFiles_;

    std::vector<Sophus::SE3f> poses_;
    msc::PinholeCamera<float> camera_;
    int currentFrame_;
    bool hasDepth_;
};

class ScanNetInterfaceFactory: public SpecificInterfaceFactory
{
public:
    virtual std::unique_ptr<CameraInterface> fromUrlParams(const std::string& urlParameters) override;
    virtual std::string getUrlPattern(const std::string& prefixTag) override;
    virtual std::string getPrefix() override;

private:
    const std::string urlPrefix_{"scannet"};
    const std::string urlParams_{"sequence_path"};
};


#endif //MASTERS_SCANNET_INTERFACE_H
