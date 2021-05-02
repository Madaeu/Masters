//
// Created by madaeu on 3/4/21.
//

#ifndef MASTERS_FEATURE_DETECTOR_H
#define MASTERS_FEATURE_DETECTOR_H

#include <vector>

#include "opencv2/opencv.hpp"
#include "opencv2/features2d.hpp"
namespace msc {

    struct Features {
        enum Type {
            ORB
        };
        using Keypoints = std::vector<cv::KeyPoint>;

        Features() {}

        Features(Type type)
                : type(type) {}

        Keypoints keypoints;
        cv::Mat descriptors;
        Type type;
    };

    class FeatureDetector {
    public:
        FeatureDetector() = default;

        virtual ~FeatureDetector() = default;

        virtual Features DetectAndCompute(const cv::Mat &image) = 0;

    };

    class ORBDetector : public FeatureDetector {
    public:
        ORBDetector() {
            orb_ = cv::ORB::create();
        }

        ~ORBDetector() override {};

        Features DetectAndCompute(const cv::Mat &image) override {
            Features features{Features::ORB};
            orb_->detectAndCompute(image, cv::Mat{}, features.keypoints, features.descriptors);
            return features;
        }

    private:
        cv::Ptr<cv::ORB> orb_;
    };

} //namespace msc
#endif //MASTERS_FEATURE_DETECTOR_H
