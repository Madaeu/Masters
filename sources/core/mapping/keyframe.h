//
// Created by madaeu on 2/19/21.
//

#ifndef MASTERS_KEYFRAME_H
#define MASTERS_KEYFRAME_H

#include "frame.h"
#include "synced_pyramid.h"

#include "Eigen/Dense"
#include "sophus/se3.hpp"
#include "opencv2/opencv.hpp"

namespace msc {
    template<typename Scalar>
    class Keyframe : public Frame<Scalar> {
    public:
        using This = Keyframe<Scalar>;
        using Base = Frame<Scalar>;
        using GradT = typename Base::GradT;
        using CodeT = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
        using ImagePyramid = msc::SynchronizedBufferPyramid<Scalar>;
        using CPUGradBuffer = vc::Buffer2DManaged<GradT, vc::TargetHost>;
        using Ptr = std::shared_ptr<Keyframe<Scalar>>;

        Keyframe() = delete;

        Keyframe(std::size_t pyramidLevels, std::size_t width, std::size_t height, std::size_t codeSize)
                : Base(pyramidLevels, width, height),
                 depthPyramid_(pyramidLevels, width, height),
                 validPyramid_(pyramidLevels, width, height),
                 uncertaintyPyramid_(pyramidLevels, width, height),
                 proximityPyramid_(pyramidLevels, width, height),
                 jacobianPyramid_(pyramidLevels, codeSize*width, height),
                 depthGradients_(width, height)
        {
            code_ = CodeT::Zero(codeSize, 1);
        }

        Keyframe(const Keyframe& other)
        : Base(other),
        depthPyramid_(other.depthPyramid_),
        validPyramid_(other.validPyramid_),
        uncertaintyPyramid_(other.uncertaintyPyramid_),
        proximityPyramid_(other.proximityPyramid_),
        jacobianPyramid_(other.jacobianPyramid_),
        depthGradients_(other.imagePyramid_.width(), other.imagePyramid_.height())
        {
            code_ = other.code_;
            depthGradients_.copyFrom(other.depthGradients_);
        }

        ~Keyframe() = default;

        typename Base::Ptr clone() override {
            return std::make_shared<This>(*this);
        }

        std::string name() override { return "Keyframe" + std::to_string(this->id_); }

        bool isKeyframe() override { return true; };

        ImagePyramid depthPyramid_;
        ImagePyramid validPyramid_;
        ImagePyramid uncertaintyPyramid_;
        ImagePyramid proximityPyramid_;
        ImagePyramid jacobianPyramid_;

        CPUGradBuffer depthGradients_;
        CodeT code_;

    };

} //namespace msc

#endif //MASTERS_KEYFRAME_H
