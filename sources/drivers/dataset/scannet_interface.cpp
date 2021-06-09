//
// Created by madaeu on 5/6/21.
//

#include "scannet_interface.h"

#include <fstream>
#include "opencv2/opencv.hpp"
#include "boost/filesystem.hpp"

std::vector<std::string> split(const std::string& s, char delim)
{
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elements;
    while (std::getline(ss,item, delim))
        elements.push_back(item);
    return elements;
}

static InterfaceRegistrar<ScanNetInterfaceFactory> automatic;

ScanNetInterface::ScanNetInterface(std::string sequencePath)
: sequencePath_(sequencePath), currentFrame_(-1)
{
    /* ScanNet structure
   * <scene_id>
   *    color
   *       *.jpg
   *    intrinsic
   *       extrinsic_color.txt
   *       extrinsic_depth.txt
   *       intrinsic_color.txt
   *       intrinsic_depth.txt
   *    pose
   *       *.txt
   */

    // check if the dataset has depth

    hasDepth_ = boost::filesystem::is_directory(sequencePath_ + "/depth");

    std::size_t numberOfFrames = probeNumberOfFrames(sequencePath_);

    for ( std::size_t i = 0; i < numberOfFrames; ++i)
    {
        std::string frameString = std::to_string(i);
        imageFiles_.push_back("color/" + frameString + ".jpg");
        depthFiles_.push_back("depth/" + frameString + ".png");
        poseFiles_.push_back("pose/" + frameString + ".txt");
    }

    poses_ = loadPoses();

    auto firstImage = cv::imread(sequencePath_ + "/" + imageFiles_[0]);

    camera_ = loadIntrinsics(sequencePath_, firstImage.cols, firstImage.rows);

    camera_.resizeViewport(640, 480);
}

ScanNetInterface::~ScanNetInterface() {}

void ScanNetInterface::grabFrames(double &timestamp, cv::Mat *image, cv::Mat *depth)
{
    if (hasMore())
        currentFrame_++;
    Sophus::SE3f dummy;
    loadFrame(currentFrame_, timestamp, image, depth, dummy);
}

std::vector<DatasetFrame> ScanNetInterface::getAll()
{
    std::vector<DatasetFrame> frames;
    frames.reserve(imageFiles_.size());
    for( uint i = 0; i < imageFiles_.size(); ++i)
    {
        DatasetFrame frame;
        loadFrame(i, frame.timestamp, &frame.img, &frame.dpt, frame.pose_wf);
        frame.timestamp = i;
        frames.push_back(frame);
    }
    return frames;
}

msc::PinholeCamera<float> ScanNetInterface::getIntrinsics()
{
    return camera_;
}

void ScanNetInterface::loadFrame(int i, double &timestamp, cv::Mat *image, cv::Mat *depth, Sophus::SE3f &pose)
{
    if( depth && !hasDepth_)
        throw std::runtime_error("Requesting to load depth when the dataset does not support it");

    if(image)
    {
        std::string imagePath = sequencePath_ + "/" + imageFiles_[i];
        *image = cv::imread(imagePath);
        cv::resize(*image, *image, cv::Size(640, 480));
        timestamp = static_cast<double>(i);
    }

    if(depth)
    {
        std::string depthPath = sequencePath_ + "/" + depthFiles_[i];
        *depth = cv::imread(depthPath, cv::IMREAD_ANYDEPTH);
        depth->convertTo(*depth, CV_32FC1, 0.001);
    }

    pose = poses_[i];
}

msc::PinholeCamera<float> ScanNetInterface::loadIntrinsics(std::string sequenceDirectory, int width, int height)
{
    std::string intrinsicsFile = sequenceDirectory + "/intrinsic/intrinsic_color.txt";
    std::ifstream f(intrinsicsFile);

    Eigen::Matrix4f K;
    std::string element;
    for( int x = 0; x < 4; ++x)
    {
        for (int y = 0; y < 4; ++y)
        {
            f >> element;
            K(x,y) = std::stof(element);
        }
    }

    return msc::PinholeCamera<float>(K(0,0), K(1,1), K(0,2), K(1,2), width, height);
}

std::vector<Sophus::SE3f> ScanNetInterface::loadPoses()
{
    std::vector<Sophus::SE3f> poses;
    Sophus::SE3f firstPose;
    for( uint i = 0; i < poseFiles_.size(); ++i)
    {
        std::string poseFile = sequencePath_ + "/" + poseFiles_[i];
        std::ifstream f(poseFile);

        Eigen::Matrix4f T;
        std::string elem;
        for (int x = 0; x < 4; ++x)
        {
            for (int y = 0; y < 4; ++y)
            {
                f >> elem;
                T(x,y) = std::stof(elem);
            }
        }
        if(T.allFinite())
        {
            auto pose = Sophus::SE3f(T);
            if (i == 0)
                firstPose = pose;
            poses_.push_back(firstPose.inverse() * pose);
        }
    }

    return poses_;
}

std::size_t ScanNetInterface::probeNumberOfFrames(std::string directory)
{
    using namespace boost::filesystem;
    return std::count_if(
            directory_iterator(path(directory + "/color")),
            directory_iterator(),
            [] (const path& p) { return is_regular_file(p); });
}

std::unique_ptr<CameraInterface> ScanNetInterfaceFactory::fromUrlParams(const std::string &urlParameters)
{
    return std::make_unique<ScanNetInterface>(urlParameters);
}

std::string ScanNetInterfaceFactory::getUrlPattern(const std::string &prefixTag)
{
    return urlPrefix_ + prefixTag + urlParams_;
}

std::string ScanNetInterfaceFactory::getPrefix()
{
    return urlPrefix_;
}