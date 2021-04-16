//
// Created by madaeu on 2/19/21.
//

#ifndef MASTERS_KEYFRAME_H
#define MASTERS_KEYFRAME_H

#include "frame.h"

template <typename Scalar>
class Keyframe : public Frame<Scalar>
{
public:
    using This = Keyframe<Scalar>;
    using Base = Frame<Scalar>;
    using Ptr = std::shared_ptr<Keyframe<Scalar>>;

    Keyframe() = delete;
    Keyframe(std::size_t pyramidLevels, std::size_t width, std::size_t height, std::size_t codeSize)
    : Base(pyramidLevels, width, height)
    { }

    ~Keyframe() = default;

    typename Base::Ptr clone() override
    {
        return std::make_shared<This>(*this);
    }

    std::string name() override { return "Keyframe" + std::to_string(this->id_); }
    bool isKeyframe() override { return true; };

    // TODO: add buffer for pyramids

    // TODO: Add code size type
};
#endif //MASTERS_KEYFRAME_H
