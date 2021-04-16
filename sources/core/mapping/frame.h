//
// Created by madaeu on 3/5/21.
//

#ifndef MASTERS_FRAME_H
#define MASTERS_FRAME_H

#include "feature_detector.h"

#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"
#include "DBoW2.h"

#include <memory>

template <typename Scalar>
class Frame
{
public:
    using This = Frame<Scalar>;
    using Ptr = std::shared_ptr<This>;
    using SE3T = Sophus::SE3<Scalar>;
    using IdType = std::size_t;

    Frame() = delete;
    Frame(std::size_t pyramidLevels, std::size_t width, std::size_t height)
    : id_(0), width_(width), height_(height) {};

    virtual ~Frame() = default;

    virtual Ptr clone()
    {
        return std::make_shared<This>(*this);
    }

    virtual std::string name() { return "frame" + std::to_string(id_); }

    virtual bool isKeyframe() { return false; }

    // TODO: Add pyramid buffers CPU/GPU

    SE3T pose_;
    cv::Mat colorImage_;
    IdType id_;

    std::size_t width_;
    std::size_t height_;

    Features features_;

    double timestamp_;

    //DBoW2::BowVector bagOfWordVector_;

    bool hasKeypoints{false};
    bool marginalized{false};
};
#endif //MASTERS_FRAME_H
