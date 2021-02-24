//
// Created by madaeu on 2/16/21.
//

#ifndef MASTERS_CAMERA_PYRAMID_H
#define MASTERS_CAMERA_PYRAMID_H

#include "pinhole_camera.h"

#include <vector>

template <typename Scalar>
class CameraPyramid{
public:
    using CameraT = PinholeCamera<Scalar>;
public:
    CameraPyramid() = default;
    CameraPyramid(const CameraT& cam, std::size_t levels);

    const CameraT operator[](int i) const;

    std::size_t levels() const;
    std::size_t  size() const;


private:
    std::vector<CameraT> cameras_;
    std::size_t levels_{};

};

template<typename Scalar>
CameraPyramid<Scalar>::CameraPyramid(const CameraT &cam, std::size_t levels)
: levels_(levels)
{
    for(auto i = 0; i < levels_; ++i)
    {
        cameras_.emplace_back(cam);
        if ( i != 0)
        {
            std::size_t newWidth{static_cast<std::size_t>(cameras_[i-1].width()/2)}; //Calculate width of next pyramid level from previous level
            std::size_t newHeight{static_cast<std::size_t>(cameras_[i-1].height()/2)}; //Calculate height of next pyramid level from previous level
            cameras_[i].resizeViewport(newWidth, newHeight);
        }
    }
}

template<typename Scalar>
const typename CameraPyramid<Scalar>::CameraT CameraPyramid<Scalar>::operator[](int i) const {
    return cameras_[i];
}

template<typename Scalar>
std::size_t CameraPyramid<Scalar>::levels() const {
    return levels_;
}

template<typename Scalar>
std::size_t CameraPyramid<Scalar>::size() const {
    return cameras_.size();
}


#endif //MASTERS_CAMERA_PYRAMID_H
