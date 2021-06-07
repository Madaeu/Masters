//
// Created by madaeu on 5/3/21.
//

#ifndef MASTERS_FACTOR_GRAPH_H
#define MASTERS_FACTOR_GRAPH_H

#include "photometric_factor.h"
#include "reprojection_factor.h"
#include "depth_prior_factor.h"
#include "gtsam_traits.h"
#include "gtsam_utilities.h"
#include "camera_pyramid.h"
#include "cuda_SFM_aligner.h"
#include "cuda_depth_aligner.h"
#include "keyframe.h"

#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/nonlinear/NonlinearFactor.h"
#include "gtsam/slam/PriorFactor.h"
#include "gtsam/nonlinear/Symbol.h"
#include "sophus/se3.hpp"

namespace msc
{
    template<typename Scalar, int CS>
    class FactorGraph
    {
        using PhotometricFactor = msc::PhotometricFactor<Scalar, CS>;
        using PhotometricFactorPtr = boost::shared_ptr<PhotometricFactor>;
        using ReprojectionFactor = msc::ReprojectionFactor<Scalar, CS>;
        using ReprojectionFactorPtr = boost::shared_ptr<ReprojectionFactor>;
        using DepthPriorFactor = msc::DepthPriorFactor<Scalar,CS>;
        using FactorPtr = gtsam::NonlinearFactorGraph::sharedFactor;
        using DiagonalNoise = gtsam::noiseModel::Diagonal;

        using CameraPyramidT = msc::CameraPyramid<Scalar>;
        using DepthAlignerPtr = typename msc::DepthAligner<Scalar, CS>::Ptr;
        using SFMAlignerPtr = typename msc::SFMAligner<Scalar,CS>::Ptr;
        using KeyframeT = msc::Keyframe<Scalar>;
        using KeyframePtr = typename KeyframeT::Ptr;
        using FrameT = msc::Frame<Scalar>;
        using FramePtr = typename FrameT::Ptr;
        using SE3T = Sophus::SE3<Scalar>;

    public:
        FactorGraph(const CameraPyramidT& cameraPyramid,
                    SFMAlignerPtr sfmAligner = nullptr,
                    DepthAlignerPtr depthAligner = nullptr);

        ~FactorGraph() = default;

        void addPosePrior(const KeyframePtr& keyframe, const SE3T& mean, const gtsam::Vector& sigmas);

        void addZeroPosePrior(const KeyframePtr& keyframe, const Scalar& sigma);

        template<typename Derived>
        void addCodePrior(const KeyframePtr& keyframe, const Eigen::MatrixBase<Derived>& mean,
                          const gtsam::Vector& sigmas);

        void addZeroCodePrior(const KeyframePtr& keyframe, const Scalar sigma);

        PhotometricFactorPtr addPhotometricFactor(const KeyframePtr& keyframe, const FramePtr frame, int pyramidLevel);

        PhotometricFactorPtr addPhotometricFactor(const KeyframePtr& keyframe0, const KeyframePtr& keyframe1, int pyramidLevel);

        ReprojectionFactorPtr addReprojectionFactor(const KeyframePtr& keyframe, const FramePtr& frame,
                                                    Scalar maxDistance, Scalar huberDelta);

        ReprojectionFactorPtr addReprojectionFactor(const KeyframePtr& keyframe0, const KeyframePtr keyframe1,
                                                    Scalar maxDistance, Scalar huberDelta);

        void addDepthPriorFactor(const KeyframePtr& keyframe, const cv::Mat& depth, double sigma);

        template<typename T>
        void addDiagonalNoisePrior(const gtsam::Key& key, const T& priorMean, const gtsam::Vector& priorSigmas);

        template<typename Factor, typename... Args>
        boost::shared_ptr<Factor> addFactor(Args&&... args);

        gtsam::NonlinearFactorGraph& graph() { return graph_; }

    private:
        CameraPyramidT cameraPyramid_;
        SFMAlignerPtr sfmAligner_;
        DepthAlignerPtr depthAligner_;
        gtsam::NonlinearFactorGraph graph_;
        std::vector<PhotometricFactorPtr> photometricFactors_;
    };

    template<typename Scalar, int CS>
    FactorGraph<Scalar, CS>::FactorGraph(const CameraPyramidT &cameraPyramid,
                                         SFMAlignerPtr sfmAligner,
                                         DepthAlignerPtr depthAligner)
                                         : cameraPyramid_(cameraPyramid)
    {
        sfmAligner_ = sfmAligner;
        depthAligner_ = depthAligner;
    }

    template<typename Scalar, int CS>
    void FactorGraph<Scalar,CS>::addPosePrior(const KeyframePtr &keyframe, const SE3T &mean,
                                              const gtsam::Vector &sigmas)
    {
        addDiagonalNoisePrior(poseKey(keyframe->id_), mean, sigmas);
    }

    template<typename Scalar, int CS>
    void FactorGraph<Scalar, CS>::addZeroPosePrior(const KeyframePtr &keyframe, const Scalar &sigma)
    {
        const SE3T identityPose;
        const gtsam::Vector sigmas = gtsam::Vector::Constant(SE3T::DoF, sigma);
        addDiagonalNoisePrior(poseKey(keyframe->id_), identityPose, sigmas);
    }

    template<typename Scalar, int CS>
    template<typename Derived>
    void FactorGraph<Scalar,CS>::addCodePrior(const KeyframePtr &keyframe, const Eigen::MatrixBase<Derived> &mean,
                                              const gtsam::Vector &sigmas)
    {
        addDiagonalNoisePrior(codeKey(keyframe->id_), mean, sigmas);
    }

    template<typename Scalar, int CS>
    void FactorGraph<Scalar, CS>::addZeroCodePrior(const KeyframePtr &keyframe, const Scalar sigma)
    {
        const gtsam::Vector zeroCode = gtsam::Vector::Zero(CS);
        const gtsam::Vector sigmas = gtsam::Vector::Constant(CS,sigma);
        addDiagonalNoisePrior(codeKey(keyframe->id_), zeroCode, sigmas);
    }

    template<typename Scalar, int CS>
    typename FactorGraph<Scalar, CS>::PhotometricFactorPtr
    FactorGraph<Scalar, CS>::addPhotometricFactor(const KeyframePtr &keyframe, const FramePtr frame,
                                                  int pyramidLevel)
    {
        if(!sfmAligner_)
        {
            sfmAligner_ = std::make_shared<SFMAligner<Scalar, CS>>();
        }
        auto id0 = keyframe->id_;
        auto id1 = frame->id_;

        auto factor = addFactor<PhotometricFactor>(cameraPyramid_[pyramidLevel],
                                                   keyframe,
                                                   frame,
                                                   poseKey(id0),
                                                   auxPoseKey(id1),
                                                   codeKey(id0),
                                                   pyramidLevel,
                                                   sfmAligner_);
        photometricFactors_.push_back(factor);
        return factor;
    }

    template<typename Scalar, int CS>
    typename FactorGraph<Scalar, CS>::PhotometricFactorPtr
    FactorGraph<Scalar, CS>::addPhotometricFactor(const KeyframePtr &keyframe0, const KeyframePtr &keyframe1,
                                                  int pyramidLevel)
    {
        if(!sfmAligner_)
        {
            sfmAligner_ = std::make_shared<SFMAligner<Scalar,CS>>();
        }
        auto id0 = keyframe0->id_;
        auto id1 = keyframe1->id_;

        auto factor = addFactor<PhotometricFactor>(cameraPyramid_[pyramidLevel],
                                                   keyframe0,
                                                   keyframe1,
                                                   poseKey(id0),
                                                   poseKey(id1),
                                                   codeKey(id0),
                                                   pyramidLevel,
                                                   sfmAligner_);
        photometricFactors_.push_back(factor);
        return factor;
    }

    template<typename Scalar, int CS>
    typename FactorGraph<Scalar, CS>::ReprojectionFactorPtr
    FactorGraph<Scalar, CS>::addReprojectionFactor(const KeyframePtr &keyframe, const FramePtr &frame,
                                                   Scalar maxDistance, Scalar huberDelta)
    {
        auto id0 = keyframe->id_;
        auto id1 = frame->id_;
        auto factor = addFactor<ReprojectionFactor>(cameraPyramid_[0],
                                                    keyframe,
                                                    frame,
                                                    poseKey(id0),
                                                    auxPoseKey(id1),
                                                    codeKey(id0),
                                                    maxDistance,
                                                    huberDelta);
        return factor;
    }

    template<typename Scalar, int CS>
    typename FactorGraph<Scalar, CS>::ReprojectionFactorPtr
    FactorGraph<Scalar, CS>::addReprojectionFactor(const KeyframePtr &keyframe0, const KeyframePtr keyframe1,
                                                   Scalar maxDistance, Scalar huberDelta)
    {
        auto id0 = keyframe0->id_;
        auto id1 = keyframe1->id_;
        auto factor = addFactor<ReprojectionFactor>(cameraPyramid_[0],
                                                    keyframe0,
                                                    keyframe1,
                                                    poseKey(id0),
                                                    poseKey(id1),
                                                    codeKey(id0),
                                                    maxDistance,
                                                    huberDelta);
        return factor;
    }

    template<typename Scalar, int CS>
    void FactorGraph<Scalar,CS>::addDepthPriorFactor(const KeyframePtr &keyframe, const cv::Mat &depth,
                                                     double sigma)
    {
        if(!depthAligner_)
        {
            depthAligner_ = boost::make_shared<DepthAligner<Scalar,CS>>();
        }

        const Scalar avgDepth = 2;
        addFactor<DepthPriorFactor>(depth, keyframe, codeKey(keyframe->id_), sigma,
                                    cameraPyramid_.levels(), avgDepth, depthAligner_);
    }

    template<typename Scalar, int CS>
    template<typename T>
    void FactorGraph<Scalar, CS>::addDiagonalNoisePrior(const gtsam::Key &key, const T &priorMean,
                                                        const gtsam::Vector &priorSigmas)
    {
        auto priorNoise = DiagonalNoise::Sigmas(priorSigmas);
        addFactor<gtsam::PriorFactor<T>>(key, priorMean, priorNoise);
    }

    template<typename Scalar, int CS>
    template<typename Factor, typename... Args>
    boost::shared_ptr<Factor> FactorGraph<Scalar,CS>::addFactor(Args&&...args)
    {
        auto factor = boost::make_shared<Factor>(std::forward<Args>(args)...);
        graph_.push_back(factor);
        return factor;
    }
} //namespace msc

#endif //MASTERS_FACTOR_GRAPH_H
