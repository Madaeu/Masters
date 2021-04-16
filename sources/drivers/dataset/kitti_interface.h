//
// Created by madaeu on 2/25/21.
//

#ifndef MASTERS_KITTI_INTERFACE_H
#define MASTERS_KITTI_INTERFACE_H

#include "dataset_interface.h"
#include "camera_interface_factory.h"


class KittiInterface : public DatasetInterface
{
public:
    KittiInterface(const std::string& seq_path, const std::string& cam_index);
    virtual ~KittiInterface();

    virtual void grabFrames(double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr) override;
    virtual std::vector<DatasetFrame> getAll() override;

    virtual bool hasIntrinsics() override { return true; }
    virtual bool hasPoses() override { return false; }
    virtual bool hasMore() override { return currentFrame_ < static_cast<int>(imageFiles_.size())-1; }

    virtual msc::PinholeCamera<float> getIntrinsics() override;

private:
    void loadFrame(int i, double& timestamp, cv::Mat* img, cv::Mat* dpt = nullptr);
    msc::PinholeCamera<float> loadIntrinsics(std::string& seq_dir, int width, int height);

    std::string sequencePath_;
    std::string cameraIndex_;
    std::vector<std::string> imageFiles_;

    msc::PinholeCamera<float> camera_;
    int currentFrame_;

};

class KittiInterfaceFactory : public SpecificInterfaceFactory
{
public:
    virtual std::unique_ptr<CameraInterface> fromUrlParams(const std::string& url_params) override;
    virtual std::string getUrlPattern(const std::string& urlPattern) override;
    virtual std::string getPrefix() override;

private:
    const std::string urlPrefix_{"kitti"};
    const std::string urlParams_{"sequence_path"};

};
#endif //MASTERS_KITTI_INTERFACE_H
