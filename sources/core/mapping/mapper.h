//
// Created by madaeu on 3/5/21.
//

#ifndef MASTERS_MAPPER_H
#define MASTERS_MAPPER_H

#include "camera_pyramid.h"
#include "keyframe.h"
#include "keyframe_map.h"
#include "feature_detector.h"

#include "sophus/se3.hpp"

namespace msc {
    template<typename Scalar, int CS>
    class Mapper {
    public:
        using MapT = Map<Scalar>;
        using FrameID = typename MapT::FrameID;
        using MapPtr = typename MapT::Ptr;
        using FrameT = Frame<Scalar>;
        using FramePtr = typename FrameT::Ptr;
        using KeyframeT = Keyframe<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using SE3T = Sophus::SE3<Scalar>;

        explicit Mapper(const msc::CameraPyramid<Scalar> &cameraPyramid);

        void reset();

        void initOneFrame(double timestamp, const cv::Mat &image, const cv::Mat &color, const msc::Features &features);

        void initTwoFrames(double timestamp0, double timestamp1,
                           const cv::Mat &image0, const cv::Mat &image1,
                           const cv::Mat &image0Color, const cv::Mat &image1Color,
                           const msc::Features &features0, const msc::Features &features1);

        MapPtr getMap() { return map_; }

    private:
        KeyframePtr buildKeyframe(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                  const SE3T &initialPose, const msc::Features features);

        FramePtr buildFrame(const cv::Mat image, const cv::Mat &colorImage,
                            const SE3T &initialPose, const msc::Features features);

        MapPtr map_;

        msc::CameraPyramid<Scalar> cameraPyramid_;

    };

    template<typename Scalar, int CS>
    Mapper<Scalar, CS>::Mapper(const msc::CameraPyramid<Scalar> &cameraPyramid)
            : cameraPyramid_(cameraPyramid) {
        map_ = std::make_shared<MapT>();
    }

    template<typename Scalar, int CS>
    void Mapper<Scalar, CS>::reset() {
        map_->clear();
    }

    template<typename Scalar, int CS>
    void
    Mapper<Scalar, CS>::initOneFrame(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                     const msc::Features &features) {
        reset();

        auto keyframe = buildKeyframe(timestamp, image, colorImage, SE3T{}, features);
        map_->addKeyframe(keyframe);

        //TODO: WorkManager stuff

    }

    template<typename Scalar, int CS>
    void
    Mapper<Scalar, CS>::initTwoFrames(double timestamp0, double timestamp1, const cv::Mat &image0,
                                      const cv::Mat &image1,
                                      const cv::Mat &image0Color, const cv::Mat &image1Color,
                                      const msc::Features &features0,
                                      const msc::Features &features1) {
        reset();

        auto keyframe0 = buildKeyframe(timestamp0, image0, image0Color, SE3T{}, features0);
        auto keyframe1 = buildKeyframe(timestamp1, image1, image1Color, SE3T{}, features1);
        map_->addKeyframe(keyframe0);
        map_->addKeyframe(keyframe1);

        //TODO: WorkManager stuff
    }


    template<typename Scalar, int CS>
    typename Mapper<Scalar, CS>::KeyframePtr
    Mapper<Scalar, CS>::buildKeyframe(double timestamp, const cv::Mat &image, const cv::Mat &colorImage,
                                      const Mapper::SE3T &initialPose, const msc::Features features) {
        //Create empty keyframe
        auto keyframe = std::make_shared<KeyframeT>(cameraPyramid_.levels(),
                                                    cameraPyramid_[0].width(),
                                                    cameraPyramid_[0].height(),
                                                    CS);
        //Fill in initial information i.e. pose, color image, and timestamp.
        keyframe->pose_ = initialPose;
        keyframe->colorImage_ = colorImage.clone();
        keyframe->timestamp_ = timestamp;

        // TODO: Create gaussian pyramids in buffers

        //TODO: Decode zero code

        //TODO: Fill pyramids with depth information.

        keyframe->features_ = features;
        keyframe->hasKeypoints = true;

        return keyframe;
    }

    template<typename Scalar, int CS>
    typename Mapper<Scalar, CS>::FramePtr
    Mapper<Scalar, CS>::buildFrame(const cv::Mat image, const cv::Mat &colorImage, const Mapper::SE3T &initialPose,
                                   const msc::Features features) {
        //Create empty keyframe
        auto frame = std::make_shared<Frame>(cameraPyramid_.levels(),
                                             cameraPyramid_[0].width(),
                                             cameraPyramid_[0].height(),
                                             CS);
        frame->pose_ = initialPose;
        frame->colorImage_ = colorImage.clone();

        //TODO: Create pyramids in buffer


        frame->features_ = features;
        frame->hasKeypoints_ = true;

        return frame;
    }

} //namespace



#endif //MASTERS_MAPPER_H
