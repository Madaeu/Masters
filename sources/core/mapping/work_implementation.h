//
// Created by madaeu on 5/4/21.
//

#ifndef MASTERS_WORK_IMPLEMENTATION_H
#define MASTERS_WORK_IMPLEMENTATION_H

#include "work.h"
#include "keyframe.h"
#include "cuda_SFM_aligner.h"
#include "camera_pyramid.h"
#include "gtsam_utilities.h"
#include "photometric_factor.h"
#include "reprojection_factor.h"
#include "sparse_geometric_factor.h"

#include <functional>

namespace msc
{
    namespace work
    {
        class WorkCallback : public Work
        {
            using Callback = std::function<void()>;

        public:
            virtual void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                                     gtsam::FactorIndices& removeIndices,
                                     gtsam::Values& initVariables) override
            {}

            virtual void update() override
            {
                function_();
                finished_ = true;
            }

            virtual bool finished() const override
            {
                return finished_;
            }

            virtual std::string name() override
            {
                return "Callback: Work";
            }

            bool finished_;
            Callback function_;

        };

        template<typename Scalar, int CS>
        class InitVariables : public Work
        {
        public:
            using KeyframePtr = typename Keyframe<Scalar>::Ptr;
            using FramePtr = typename Frame<Scalar>::Ptr;

            InitVariables(KeyframePtr keyframe, Scalar codePrior);

            InitVariables(KeyframePtr keyframe, Scalar codePrior, Scalar posePrior);

            InitVariables(FramePtr frame);

            virtual ~InitVariables() = default;

            virtual void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                                     gtsam::FactorIndices& removeIndices,
                                     gtsam::Values& initValues) override;

            virtual void update() override;

            virtual bool finished() const override
            {
                return !first_;
            }

            virtual std::string name() override;

        private:
            bool first_;
            gtsam::Values initVariables_;
            gtsam::NonlinearFactorGraph priors_;
            std::string name_;
        };

        template<typename Scalar>
        class OptimizeWork: public Work
        {
        public:
            using KeyframePtr = typename Keyframe<Scalar>::Ptr;
            using FramePtr = typename Frame<Scalar>::Ptr;
            using IteratorList = std::vector<int>;

            OptimizeWork(int iterator, bool removeAfter = false);

            OptimizeWork(IteratorList iterators, bool removeAfter = false);

            virtual ~OptimizeWork() = default;

            virtual void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                                     gtsam::FactorIndices& removeIndices,
                                     gtsam::Values& initVariables) override;

            virtual gtsam::NonlinearFactorGraph constructFactors();

            virtual void update() override;

            virtual bool finished() const override;

            virtual void signalNoRelinearize() override;
            virtual void signalRemove() override;
            virtual void lastFactorIndices(gtsam::FactorIndices& indices) override;

            virtual bool involves(FramePtr frame) const = 0;

            bool isCoarsestLevel() const;
            bool isNewLevelStart() const;

        protected:
            bool remove_;
            bool first_;
            IteratorList iterators_;
            IteratorList originalIterators_;
            int activeLevel_;
            gtsam::FactorIndices lastIndices_;
            bool removeAfter_;

        };

        template<typename Scalar, int CS>
        class OptimizePhotometric: public OptimizeWork<Scalar>
        {
        public:
            using KeyframePtr = typename Keyframe<Scalar>::Ptr;
            using FramePtr = typename Frame<Scalar>::Ptr;
            using AlignerPtr = typename SFMAligner<Scalar,CS>::Ptr;
            using CameraPyramidT = CameraPyramid<Scalar>;
            using PhotometricFactorT = PhotometricFactor<Scalar, CS>;
            using IteratorList = typename OptimizeWork<Scalar>::IteratorList;

            OptimizePhotometric(KeyframePtr keyframe, FramePtr frame, IteratorList iterators, CameraPyramidT cameraPyramid,
                                AlignerPtr aligner, bool updateValid = false, bool removeAfter = false);

            virtual ~OptimizePhotometric() = default;

            gtsam::NonlinearFactorGraph constructFactors() override;
            virtual bool involves(FramePtr frame) const;
            virtual std::string name();

        private:
            KeyframePtr keyframe_;
            FramePtr frame_;
            CameraPyramidT cameraPyramid_;
            AlignerPtr aligner_;
            bool updateValid_;
        };

        template<typename Scalar, int CS>
        class OptimizeReprojection: public OptimizeWork<Scalar>
        {
        public:
            using KeyframePtr = typename Keyframe<Scalar>::Ptr;
            using FramePtr = typename Frame<Scalar>::Ptr;
            using ReprojectionFactorT = ReprojectionFactor<Scalar,CS>;
            using CameraT = PinholeCamera<Scalar>;

            OptimizeReprojection(KeyframePtr keyframe, FramePtr frame, int iterators,
                                 CameraT camera, float maxFeatureDistance, float huberDelta,
                                 float sigma, int maxIterations, float threshold);

            virtual ~OptimizeReprojection() = default;

            virtual bool finished() const override;

            virtual gtsam::NonlinearFactorGraph constructFactors() override;

            virtual bool involves(FramePtr frame ) const override;

            virtual std::string name() override;

        private:
            KeyframePtr keyframe_;
            FramePtr frame_;
            CameraT camera_;
            float maxDistance_;
            float huberDelta_;
            float sigma_;
            int maxIterations_;
            float threshold_;
            bool finished_ = false;
        };

        template<typename Scalar, int CS>
        class OptimizeGeometric : public OptimizeWork<Scalar>
        {
        public:
            using FramePtr = typename msc::Frame<Scalar>::Ptr;
            using KeyframePtr = typename msc::Keyframe<Scalar>::Ptr;
            using GeometricFactorT = SparseGeometricFactor<Scalar, CS>;

            OptimizeGeometric(KeyframePtr keyframe0, KeyframePtr keyframe1,
                              int iters, msc::PinholeCamera<Scalar> camera,
                              int numberOfPoints, float huberDelta, bool stochastic);

            virtual ~OptimizeGeometric() = default;

            virtual gtsam::NonlinearFactorGraph constructFactors() override;
            virtual bool involves(FramePtr ptr) const override;
            virtual std::string name() override;

        private:
            KeyframePtr keyframe0_;
            KeyframePtr keyframe1_;
            msc::PinholeCamera<Scalar> camera_;
            int numberOfPoints_;
            float huberDelta_;
            bool stochastic_;
        };

    } //namespace work

} // namespace msc

#endif //MASTERS_WORK_IMPLEMENTATION_H
