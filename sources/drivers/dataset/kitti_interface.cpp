//
// Created by madaeu on 2/25/21.
//

#include "kitti_interface.h"

#include "boost/filesystem.hpp"
#include "opencv2/opencv.hpp"

#include <iostream>

static InterfaceRegistrar<KittiInterfaceFactory> automatic;

KittiInterface::KittiInterface(const std::string &seq_path, const std::string &cam_index)
: sequencePath_(seq_path), cameraIndex_(cam_index), currentFrame_(-1)
{
    /* Interface assumes the following file structure of Kitti data:
     * calib_cam_to_cam.txt
     * calib_imu_to_velo.txt
     * calib_velo_to_cam.txt
     *      <drive_id>/
     *          image_0x/
     *              data/
     *                  *.png
     *              timestamps.txt
     *          oxts/
     *              data/
     *                  *.txt
     *              dataformat.txt
     *              timestamps.txt
     *          velodyne_points/
    std::string drive_id{dataPath.parent_path().string()};
     *              data/
     *                  *.bin
     *              timestamps.txt
     *              timestamps_end.txt
     *              timestamps_start.txt
     *
     */

    std::string dataPath{sequencePath_ + "/image_" + cameraIndex_};

    boost::filesystem::path dataDirectory{dataPath + "/data"};
    std::vector<boost::filesystem::path> frames;
    copy(boost::filesystem::directory_iterator(dataDirectory), boost::filesystem::directory_iterator(), back_inserter(frames) );
    sort(frames.begin(), frames.end());

    for (std::vector<boost::filesystem::path>::const_iterator it(frames.begin()), it_end(frames.end()); it != it_end; ++it){
        imageFiles_.push_back("/data/" + (*it).filename().string());
    }
    auto firstImage = cv::imread(dataPath + imageFiles_[0]);

    camera_ = loadIntrinsics(sequencePath_, firstImage.cols, firstImage.rows);
    camera_.resizeViewport(640,480);

}

KittiInterface::~KittiInterface() {}

void KittiInterface::grabFrames(double &timestamp, cv::Mat *img, cv::Mat *dpt) {
    if (hasMore()){
        currentFrame_++;
    }
    loadFrame(currentFrame_, timestamp, img, dpt);
}

std::vector<DatasetFrame> KittiInterface::getAll() {
    return std::vector<DatasetFrame>();
}

msc::PinholeCamera<float> KittiInterface::getIntrinsics() {
    return camera_;
}

msc::PinholeCamera<float> KittiInterface::loadIntrinsics(std::string &seq_dir, int width, int height) {
    std::string calibFile = boost::filesystem::path(seq_dir).parent_path().string() + "/calib_cam_to_cam.txt";
    std::ifstream f(calibFile);
    if(!f.is_open()){
        std::cerr << "Could not open calibration file!";
        return msc::PinholeCamera<float>();
    }
    //Iterate through lines in calibration file to find corresponding camera intrinsics
    std::string intrinsicStr = "K_" + cameraIndex_ + ": ";
    while (f){
        std::string strInput;
        std::getline(f,strInput);
        if (strInput.find(intrinsicStr) != std::string::npos) {
            std::vector<float> intrinsicValues;
            std::string values;
            std::stringstream intrinsicsStream(strInput.substr(intrinsicStr.size()));
            while (std::getline(intrinsicsStream,values,' ')){
                intrinsicValues.push_back(std::stof(values));
            }
            return msc::PinholeCamera<float>(intrinsicValues[0], intrinsicValues[4], intrinsicValues[2], intrinsicValues[5], width, height);
        }
    }
    return msc::PinholeCamera<float>();
}

void KittiInterface::loadFrame(int i, double &timestamp, cv::Mat *img, cv::Mat *dpt) {

    if(img){
        std::string imagePath = sequencePath_ + "/image_" + cameraIndex_ + imageFiles_[i];
        *img = cv::imread(imagePath);
        if(img->empty()){
            std::cout << "Problemos!";
        }
        cv::resize(*img,*img, cv::Size(640, 480));
    }
    timestamp = currentFrame_;
}

std::unique_ptr<CameraInterface> KittiInterfaceFactory::fromUrlParams(const std::string &url_params)
{
    std::string cameraStr{"image_"};
    size_t found = url_params.find(cameraStr);
    if(found == std::string::npos){
        throw MalformedUrlException(url_params, "No camera directory specified!");
    }
    std::string driveDirectory{url_params.substr(0,found-1)};
    std::string cameraIndex{url_params.substr(found+cameraStr.length(), std::string::npos)};

    return std::make_unique<KittiInterface>(driveDirectory, cameraIndex);
}

std::string KittiInterfaceFactory::getUrlPattern(const std::string &prefixTag)
{
    return urlPrefix_ + prefixTag + urlParams_;
}

std::string KittiInterfaceFactory::getPrefix()
{
    return urlPrefix_;
}