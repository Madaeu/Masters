//
// Created by madaeu on 5/4/21.
//

#include "work_implementation.h"
#include "gtsam_utilities.h"
#include "gtsam_traits.h"

#include <gtsam/slam/PriorFactor.h>

namespace msc
{
    namespace work
    {
        template<typename Scalar, int CS>
        InitVariables<Scalar,CS>::InitVariables(KeyframePtr keyframe, Scalar codePrior)
        : first_(true)
        {
            using DiagonalNoise = gtsam::noiseModel::Diagonal;
            using PriorFactor = gtsam::PriorFactor<gtsam::Vector>;

            gtsam::Vector zeroCode = gtsam::Vector::Zero(CS);
            initVariables_.insert(poseKey(keyframe->id_), keyframe->pose_);
            initVariables_.insert(codeKey(keyframe->id_), zeroCode);

            gtsam::Vector priorSigmas = gtsam::Vector::Constant(CS, 1, codePrior);
            auto priorNoise = DiagonalNoise::Sigmas(priorSigmas);
            priors_.template emplace_shared<PriorFactor>(codeKey(keyframe->id_), zeroCode, priorNoise);

            name_ = keyframe->name();
        }

        template<typename Scalar, int CS>
        InitVariables<Scalar, CS>::InitVariables(KeyframePtr keyframe, Scalar codePrior,
                                                 Scalar posePrior)
        : InitVariables(keyframe, codePrior)
        {
            using DiagonalNoise = gtsam::noiseModel::Diagonal;
            using PriorFactor = gtsam::PriorFactor<Sophus::SE3f>;

            gtsam::Vector priorSigmas = gtsam::Vector::Constant(6, 1, posePrior);
            auto priorNoise = DiagonalNoise::Sigmas(priorSigmas);
            priors_.emplace_shared<PriorFactor>(poseKey(keyframe->id_), Sophus::SE3f{}, priorNoise);
        }

        template<typename Scalar, int CS>
        InitVariables<Scalar,CS>::InitVariables(FramePtr frame)
        : first_(true)
        {
            initVariables_.insert(auxPoseKey(frame->id_), frame->pose_);
            name_ = frame->name();
        }

        template<typename Scalar, int CS>
        void InitVariables<Scalar, CS>::bookkeeping(gtsam::NonlinearFactorGraph &newFactors,
                                                    gtsam::FactorIndices &removeIndices, gtsam::Values &initValues)
        {
            if (first_)
            {
                newFactors += priors_;
                initValues.insert(initVariables_);
                first_ = false;
            }
        }

        template<typename Scalar, int CS>
        void InitVariables<Scalar, CS>::update()
        {

        }

        template<typename Scalar, int CS>
        std::string InitVariables<Scalar, CS>::name()
        {
            return id() + "Init Variables " + name_;
        }

        template class InitVariables<float,32>;

/* ************************************************************************* */

        template<typename Scalar>
        OptimizeWork<Scalar>::OptimizeWork(int iterator, bool removeAfter)
        : remove_(false), first_(true), iterators_({iterator}), originalIterators_(iterators_),
        activeLevel_(0), removeAfter_(removeAfter)
        { }

        template<typename Scalar>
        OptimizeWork<Scalar>::OptimizeWork(IteratorList iterators, bool removeAfter)
        : remove_(false), first_(true), iterators_(iterators),
        activeLevel_(iterators.size()-1), removeAfter_(removeAfter)
        { }

        template<typename Scalar>
        void OptimizeWork<Scalar>::bookkeeping(gtsam::NonlinearFactorGraph &newFactors,
                                               gtsam::FactorIndices &removeIndices,
                                               gtsam::Values &initVariables)
        {
            if (remove_)
            {
                removeIndices.insert(removeIndices.begin(), lastIndices_.begin(),
                                     lastIndices_.end());

                activeLevel_ = -2;
            }

            if (first_ || (activeLevel_ >= 0 && isNewLevelStart()))
            {
                first_ = false;
                newFactors += constructFactors();
                removeIndices.insert(removeIndices.begin(), lastIndices_.begin(),
                                     lastIndices_.end());
            }
        }

        template<typename Scalar>
        gtsam::NonlinearFactorGraph OptimizeWork<Scalar>::constructFactors()
        {
            return gtsam::NonlinearFactorGraph();
        }

        template<typename Scalar>
        void OptimizeWork<Scalar>::update()
        {
            if( activeLevel_ >= 0 && --iterators_[activeLevel_] < 0)
            {
                activeLevel_ -= 1;
            }

            if(removeAfter_ && activeLevel_ < 0)
            {
                signalRemove();
            }
        }

        template<typename Scalar>
        bool OptimizeWork<Scalar>::finished() const
        {
            if (removeAfter_)
            {
                return activeLevel_ == -2;
            }
            else
            {
                return activeLevel_ == -1;
            }
        }

        template<typename Scalar>
        void OptimizeWork<Scalar>::signalNoRelinearize()
        {
            if(!first_)
            {
                activeLevel_ = -1;
            }
        }

        template<typename Scalar>
        void OptimizeWork<Scalar>::signalRemove()
        {
            remove_ = true;
        }

        template<typename Scalar>
        void OptimizeWork<Scalar>::lastFactorIndices(gtsam::FactorIndices &indices)
        {
            lastIndices_ = indices;
        }

        template<typename Scalar>
        bool OptimizeWork<Scalar>::isCoarsestLevel() const
        {
            return activeLevel_ == (int)iterators_.size()-1;
        }

        template<typename Scalar>
        bool OptimizeWork<Scalar>::isNewLevelStart() const
        {
            return activeLevel_ >= 0 && iterators_[activeLevel_] == originalIterators_[activeLevel_];
        }

        template class OptimizeWork<float>;

/* ************************************************************************* */

        template<typename Scalar, int CS>
        OptimizePhotometric<Scalar,CS>::OptimizePhotometric(KeyframePtr keyframe,
                                                            FramePtr frame,
                                                            IteratorList iterators,
                                                            CameraPyramidT cameraPyramid,
                                                            AlignerPtr aligner,
                                                            bool updateValid,
                                                            bool removeAfter)
        : OptimizeWork<Scalar>(iterators, removeAfter), keyframe_(keyframe),
        frame_(frame), cameraPyramid_(cameraPyramid), aligner_(aligner),
        updateValid_(updateValid)
        { }

        template<typename Scalar, int CS>
        gtsam::NonlinearFactorGraph OptimizePhotometric<Scalar,CS>::constructFactors()
        {
            gtsam::NonlinearFactorGraph graph;
            gtsam::Key pose0Key = poseKey(keyframe_->id_);
            gtsam::Key code0Key = codeKey(keyframe_->id_);
            gtsam::Key pose1Key = frame_->isKeyframe() ? poseKey(frame_->id_) : auxPoseKey(frame_->id_);

            graph.emplace_shared<PhotometricFactorT>(cameraPyramid_[this->activeLevel_],
                                                     keyframe_, frame_,
                                                     pose0Key, pose1Key, code0Key, this->activeLevel_,
                                                     aligner_, updateValid_);
            return graph;
        }

        template<typename Scalar, int CS>
        std::string OptimizePhotometric<Scalar, CS>::name()
        {
            std::stringstream ss;
            ss << this->id() << "Optimize Photometric: " << keyframe_->name() << " -> " << frame_->name()
               << " iterators = " << (this->activeLevel_ < 0 ? 0 : this->iterators_[this->activeLevel_])
               << " active level = " << this->activeLevel_
               << " new level = " << this->isNewLevelStart()
               << " finished" << this->finished();
            ss << " factor indices = ";
            for (auto& idx : this->lastIndices_)
            {
                ss << idx << " ";
            }
            return ss.str();
        }

        template<typename Scalar, int CS>
        bool OptimizePhotometric<Scalar, CS>::involves(FramePtr frame) const
        {
            return frame_ == frame || keyframe_ == frame;
        }

        template class OptimizePhotometric<float, 32>;

/* ************************************************************************* */

        template<typename Scalar, int CS>
        OptimizeReprojection<Scalar, CS>::OptimizeReprojection(KeyframePtr keyframe,
                                                               FramePtr frame, int iterators,
                                                               CameraT camera,
                                                               float maxFeatureDistance,
                                                               float huberDelta, float sigma,
                                                               int maxIterations,
                                                               float threshold)
        : OptimizeWork<Scalar>(iterators),
        keyframe_(keyframe), frame_(frame),
        camera_(camera), maxDistance_(maxFeatureDistance),huberDelta_(huberDelta),
        sigma_(sigma), maxIterations_(maxIterations),
        threshold_(threshold)
        {}

        template<typename Scalar, int CS>
        gtsam::NonlinearFactorGraph OptimizeReprojection<Scalar, CS>::constructFactors()
        {
            gtsam::NonlinearFactorGraph graph;
            gtsam::Key pose0Key = poseKey(keyframe_->id_);
            gtsam::Key code0Key = codeKey(keyframe_->id_);
            gtsam::Key pose1Key = frame_->isKeyframe() ? poseKey(frame_->id_) : auxPoseKey(frame_->id_);

            boost::shared_ptr<ReprojectionFactorT> factor = boost::make_shared<ReprojectionFactorT>(camera_, keyframe_, frame_,
                                                                                                    pose0Key, pose1Key, code0Key,
                                                                                                    maxDistance_, huberDelta_,
                                                                                                    sigma_, maxIterations_,
                                                                                                    threshold_);
            if (factor->matches().empty())
            {
                this->finished_ = true;
            }
            else
            {
                graph.add(factor);
            }
            return graph;
        }

        template<typename Scalar, int CS>
        bool OptimizeReprojection<Scalar, CS>::finished() const
        {
            return OptimizeWork<Scalar>::finished() || finished_;
        }

        template<typename Scalar, int CS>
        bool OptimizeReprojection<Scalar, CS>::involves(FramePtr frame) const
        {
            return frame_ == frame || keyframe_ == frame;
        }

        template<typename Scalar, int CS>
        std::string OptimizeReprojection<Scalar, CS>::name()
        {
            std::stringstream ss;
            ss << this->id() << " Optimize Reprojection: " << keyframe_->name() << " -> " << frame_->name()
               << " iterators = " << this->iterators_[this->activeLevel_]
               << " finished = " << this->finished();
            return ss.str();
        }

        template class OptimizeReprojection<float, 32>;

/* ************************************************************************* */

        template<typename Scalar, int CS>
        OptimizeGeometric<Scalar, CS>::OptimizeGeometric(KeyframePtr keyframe0,
                                                         KeyframePtr keyframe1, int iters,
                                                         msc::PinholeCamera<Scalar> camera,
                                                         int numberOfPoints, float huberDelta,
                                                         bool stochastic)
        : OptimizeWork<Scalar>(iters, false),
          keyframe0_(keyframe0), keyframe1_(keyframe1),
          camera_(camera), numberOfPoints_(numberOfPoints),
          huberDelta_(huberDelta), stochastic_(stochastic)
          {}

        template<typename Scalar, int CS>
        gtsam::NonlinearFactorGraph OptimizeGeometric<Scalar, CS>::constructFactors()
        {
            gtsam::NonlinearFactorGraph graph;
            auto pose0Key = poseKey(keyframe0_->id_);
            auto pose1Key = poseKey(keyframe1_->id_);
            auto code0Key = codeKey(keyframe0_->id_);
            auto code1Key = codeKey(keyframe1_->id_);

            graph.emplace_shared<GeometricFactorT>(camera_, keyframe0_, keyframe1_, pose0Key, pose1Key,
                                                   code0Key, code1Key, numberOfPoints_, huberDelta_,
                                                   stochastic_);

            return graph;
        }

        template<typename Scalar, int CS>
        bool OptimizeGeometric<Scalar, CS>::involves(FramePtr ptr) const
        {
            return keyframe0_ == ptr || keyframe1_ == ptr;
        }

        template<typename Scalar, int CS>
        std::string OptimizeGeometric<Scalar, CS>::name()
        {
            std::stringstream  ss;
            ss << this->id() << "OptimizeGeo " << keyframe0_->name() << " -> " << keyframe1_->name()
               << " iterators = " << this->iterators_[this->activeLevel_]
               << " finished = " << this->finished();
            return ss.str();
        }

        template class OptimizeGeometric<float, 32>;
    } //namespace work

} //namespace msc