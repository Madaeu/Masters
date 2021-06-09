//
// Created by madaeu on 5/5/21.
//

#ifndef MASTERS_LOOP_DETECTOR_H
#define MASTERS_LOOP_DETECTOR_H

#include "camera_pyramid.h"
#include "camera_tracker.h"

#include "sophus/se3.hpp"
#include "DBoW2/DBoW2.h"

#include <memory>
#include <vector>

namespace msc
{
    template<typename Scalar>
    struct LoopDetectorConfiguration
    {
        typename CameraTracker<Scalar>::TrackerConfig trackerConfiguration;
        std::vector<int> iterations;
        float minSimiliarity = 0.35f;
        float maxError = 0.5f;
        float maxDistance = 0.1f;
        int maxCandidates = 3;
        int activeWindows = 5;
    };

    template<typename Scalar>
    class LoopDetector
    {
    public:
        using SE3T = Sophus::SE3<Scalar>;
        using GradT = Eigen::Matrix<Scalar, 1,2>;
        using MapPtr = typename Map<Scalar>::Ptr;
        using CameraPyramidT = CameraPyramid<Scalar>;
        using KeyframeT = Keyframe<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using IdType = typename KeyframeT::IdType;
        using DescriptionMap = std::map<IdType, DBoW2::BowVector>;
        using ImagePyramid = vc::RuntimeBufferPyramidManaged<Scalar, vc::TargetDeviceCUDA>;
        using GradPyramid = vc::RuntimeBufferPyramidManaged<GradT, vc::TargetDeviceCUDA>;
        using TrackerPtr = std::shared_ptr<CameraTracker<Scalar>>;


        struct LoopInfo
        {
            int loopId = 1;
            SE3T pose;
            bool detected = false;
        };

        LoopDetector(std::string vocabularyPath, MapPtr map, LoopDetectorConfiguration<Scalar> configuration, CameraPyramidT cameraPyramid);

        void addKeyframe(KeyframePtr keyframe);
        int detectLoop(KeyframePtr keyframe);
        LoopInfo detectLoop(const ImagePyramid& imagePyramid, const GradPyramid& gradientPyramid,
                            const Features& features, KeyframePtr currentKeyframe, SE3T cameraPose);
        int detectLocalLoop(const ImagePyramid& imagePyramid, const GradPyramid& gradientPyramid,
                       const Features& features, KeyframePtr currentKeyframe, SE3T cameraPose);

        void reset();

    private:
        std::vector<cv::Mat> changeStructure(cv::Mat image);

        DescriptionMap descriptionMap_;
        OrbVocabulary vocabulary_;
        OrbDatabase database_;
        MapPtr map_;
        TrackerPtr cameraTracker_;
        LoopDetectorConfiguration<Scalar> configuration_;
    };

    template<typename Scalar>
    LoopDetector<Scalar>::LoopDetector(std::string vocabularyPath, MapPtr map,
                                       LoopDetectorConfiguration<Scalar> configuration, CameraPyramidT cameraPyramid)
    : vocabulary_(vocabularyPath), database_(vocabulary_, false, 0), map_(map), configuration_(configuration)
    {
        configuration.trackerConfiguration.iterationsPerLevel = configuration.iterations;
        cameraTracker_ = std::make_shared<CameraTracker<Scalar>>(cameraPyramid, configuration.trackerConfiguration);
    }

    template<typename Scalar>
    void LoopDetector<Scalar>::addKeyframe(KeyframePtr keyframe)
    {
        auto qf = changeStructure(keyframe->features_.descriptors);
        vocabulary_.transform(qf, keyframe->bagOfWordVector_);
        database_.add(keyframe->bagOfWordVector_);
        descriptionMap_[keyframe->id_] = keyframe->bagOfWordVector_;
    }

    template<typename Scalar>
    int LoopDetector<Scalar>::detectLoop(KeyframePtr keyframe)
    {
        auto qf = changeStructure(keyframe->features_.descriptors);
        vocabulary_.transform(qf, keyframe->bagOfWordVector);

        Scalar minDistance = std::numeric_limits<Scalar>::max();
        auto connections = map_->keyframes_.getConnections(keyframe->id_);
        for ( auto& c : connections)
        {
            auto distance = vocabulary_.score(descriptionMap_[c], keyframe->bagOfWordVector_);
            if (distance < minDistance)
            {
                minDistance = distance;
            }

            DBoW2::QueryResults queryResults;
            database_.query(keyframe->bagOfWordVector_, queryResults, 3);

            for (auto& result : queryResults)
            {
                auto keyframeId = result.Id + 1;
                Scalar l2Difference = (map_->keyframes_.get(keyframeId)->code_ - keyframe->code_).norm();

                if (std::find(connections.begin(), connections.end(), keyframeId) != connections.end())
                {
                    continue;
                }
                if (result.Score < minDistance)
                {
                    continue;
                }
            }

            return 0;
        }
    }

    template<typename Scalar>
    typename LoopDetector<Scalar>::LoopInfo
    LoopDetector<Scalar>::detectLoop(const ImagePyramid &imagePyramid, const GradPyramid &gradientPyramid,
                                     const Features &features, KeyframePtr currentKeyframe, SE3T cameraPose)
    {
        DBoW2::BowVector bagOfWordsVector;
        auto qf = changeStructure(features.descriptors);
        vocabulary_.transform(qf, bagOfWordsVector);

        auto minScore = vocabulary_.score(currentKeyframe->bagOfWordVector_, bagOfWordsVector);

        DBoW2::QueryResults queryResults;
        database_.query(bagOfWordsVector, queryResults, configuration_.maxCandidates);

        std::vector<IdType> candidates;
        for(auto& result : queryResults)
        {
            auto keyframeId = result.Id + 1;

            if (keyframeId == currentKeyframe->id_)
            {
                continue;
            }
            if ( keyframeId > currentKeyframe->id_ - configuration_.activeWindows)
            {
                continue;
            }

            if (result.Score < configuration_.minSimiliarity)
            {
                continue;
            }

            candidates.push_back(keyframeId);
        }

        if (candidates.empty())
        {
            return LoopInfo{};
        }

        Scalar bestDistance = std::numeric_limits<Scalar>::infinity();
        IdType bestId = candidates[0];
        SE3T bestCameraPose;
        for (auto& id : candidates)
        {
            auto keyframe = map_->keyframes_.get(id);
            cameraTracker_->setKeyframe(keyframe);
            cameraTracker_->reset();
            cameraTracker_->trackFrame(imagePyramid, gradientPyramid);

            auto cameraPose = cameraTracker_->getPoseEstimate();
            auto distance = (cameraPose.translation() - keyframe->pose_.translation()).norm();

            if (cameraTracker_->getInliers() < 0.5f)
            {
                continue;
            }
            if (distance < bestDistance)
            {
                bestDistance = distance;
                bestId = id;
                bestCameraPose = cameraPose;
            }
        }

        if (bestDistance < configuration_.maxDistance)
        {
            LoopInfo foundLoop;
            foundLoop.detected = true;
            foundLoop.pose = bestCameraPose;
            foundLoop.loopId = bestId;
            return foundLoop;
        }

        return LoopInfo{};
    }

    template<typename Scalar>
    int LoopDetector<Scalar>::detectLocalLoop(const ImagePyramid &imagePyramid, const GradPyramid &gradientPyramid,
                                              const Features &features, KeyframePtr currentKeyframe, SE3T cameraPose)
    {
        Scalar bestDistance = std::numeric_limits<Scalar>::infinity();
        IdType bestId = map_->keyframes_.lastID();

        auto ii = map_->keyframes_.ids().rbegin();
        for (int i = 0; i < configuration_.activeWindows; ++i)
        {
            if( ii == map_->keyframes_.ids().rend())
            {
                break;
            }

            auto id = *ii;
            auto keyframe = map_->keyframes_.get(id);
            auto pose = keyframe->pose_;
            auto distance = (cameraPose.translation() - pose.translation()).norm();

            if (distance < bestDistance && id != currentKeyframe->id_)
            {
                bestDistance = distance;
                bestId = id;
            }

            ii++;
        }

        if (bestId != currentKeyframe->id_ && bestDistance < configuration_.maxDistance)
        {
            return bestId;
        }
        return 0;
    }

    template<typename Scalar>
    void LoopDetector<Scalar>::reset()
    {
        database_.clear();
        descriptionMap_.clear();
    }

    template<typename Scalar>
    std::vector<cv::Mat> LoopDetector<Scalar>::changeStructure(cv::Mat image)
    {
        std::vector<cv::Mat> qf(image.rows);
        for (int i = 0; i < image.rows; ++i)
        {
            qf[i] = image.row(i);
        }
        return qf;
    }
} // namespace msc

#endif //MASTERS_LOOP_DETECTOR_H
