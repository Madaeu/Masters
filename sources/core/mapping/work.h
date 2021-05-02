//
// Created by madaeu on 5/1/21.
//

#ifndef MASTERS_WORK_H
#define MASTERS_WORK_H

#include "gtsam/nonlinear/NonlinearFactorGraph.h"
#include "gtsam/base/FastVector.h"

namespace gtsam
{
    using FactorIndices = gtsam::FastVector<size_t>;
} // namespace gtsam

namespace msc
{
    namespace work
    {
        class Work
        {
        public:
            using Ptr = std::shared_ptr<Work>;

            Work();

            virtual ~Work() = default;

            virtual void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                                     gtsam::FastVector<size_t>& removeIndices,
                                     gtsam::Values& initVariables) = 0;
            virtual void update() = 0;
            virtual bool finished() const = 0;
            virtual std::string name() = 0;

            virtual void signalNoRelinearize() { }
            virtual void signalRemove() { }
            virtual void lastFactorIndices(gtsam::FactorIndices& indices) {}

            template< typename T, typename... Args>
            Ptr addChild(Args&&... args) {
                auto child = std::make_shared<T>(std::forward<Args>(args)...);
                return addChild(child);
            }

            Ptr addChild(Ptr child);
            Ptr removeChild();

            std::string id() const {return id; }
        private:
            Ptr child_;
            std::string id_;

            static int nextId_
        };

        Work::Work()
        {
            id = "[" + std::to_string(nextId_++) + "]";
        }

        int Work::nextId_ = 0;

        Work::Ptr Work::addChild(Ptr child)
        {
            child_ = child;
            return child;
        }

        Work::Ptr Work::removeChild()
        {
            auto child = child_;
            child_.reset();
            return child;
        }

    } // namespace work
} //namespace msc

#endif //MASTERS_WORK_H
