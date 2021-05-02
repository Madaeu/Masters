//
// Created by madaeu on 3/5/21.
//

#ifndef MASTERS_FRAME_H
#define MASTERS_FRAME_H

#include "feature_detector.h"
#include "cuda_image_proc.h"
#include "synced_pyramid.h"

#include "opencv2/opencv.hpp"
#include "sophus/se3.hpp"
#include "DBoW2.h"
#include "Eigen/Dense"

#include <memory>

namespace msc {

    template<typename Scalar>
    class Frame
    {
    public:
        using This = Frame<Scalar>;
        using Ptr = std::shared_ptr<This>;
        using SE3T = Sophus::SE3<Scalar>;
        using GradT = Eigen::Matrix<Scalar, 1,2>;
        using ImagePyramid = msc::SynchronizedBufferPyramid<Scalar>;
        using GradPyramid = msc::SynchronizedBufferPyramid<GradT>;
        using IdType = std::size_t;

        Frame() = delete;

        Frame(std::size_t pyramidLevels, std::size_t width, std::size_t height)
        : imagePyramid_(pyramidLevels, width, height),
          gradientPyramid_(pyramidLevels, width, height),
          id_(0), width_(width), height_(height) {};

        Frame(const Frame& other)
        : imagePyramid_(other.imagePyramid_),
        gradientPyramid_(other.gradientPyramid_),
        pose_(other.pose_),
        id_(other.id_),
        features_(other.features_),
        timestamp_(other.timestamp_),
        //bagOfWordVector_(other.bagOfWordVector_),
        hasKeypoints_(other.hasKeypoints_),
        marginalized_(other.marginalized_)
        {
            colorImage_ = other.colorImage_.clone();
        }

        virtual ~Frame() = default;

        virtual Ptr clone() {
            return std::make_shared<This>(*this);
        }

        virtual std::string name() { return "frame" + std::to_string(id_); }

        virtual bool isKeyframe() { return false; }

        void fillPyramids(const cv::Mat& image, std::size_t pyramidLevels)
        {
            for (std::size_t i = 0; i < pyramidLevels; ++i)
            {
                if (i == 0)
                {
                    vc::Image2DView<float, vc::TargetHost> temp(image);
                    imagePyramid_.getLevelGPU(0).template copyFrom(temp);
                    msc::sobelGradients(imagePyramid_.getLevelGPU(0), gradientPyramid_.getLevelGPU(0));
                    continue;
                }
                gaussianBlurDown(imagePyramid_.getLevelGPU(i-1), imagePyramid_.getLevelGPU(i));
                sobelGradients(imagePyramid_.getLevelGPU(i), gradientPyramid_.getLevelGPU(i));
            }
        }

        // TODO: Add pyramid buffers CPU/GPU
        ImagePyramid imagePyramid_;
        GradPyramid  gradientPyramid_;

        SE3T pose_;
        cv::Mat colorImage_;
        IdType id_;

        std::size_t width_;
        std::size_t height_;

        Features features_;

        double timestamp_;

        //DBoW2::BowVector bagOfWordVector_;

        bool hasKeypoints_{false};
        bool marginalized_{false};
    };
}
#endif //MASTERS_FRAME_H
