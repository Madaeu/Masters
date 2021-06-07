//
// Created by madaeu on 5/4/21.
//

#ifndef MASTERS_WORK_MANAGER_H
#define MASTERS_WORK_MANAGER_H

#include "work.h"
#include "work_implementation.h"

#include <functional>
#include <list>

namespace msc
{
    namespace work
    {
        class WorkManager
        {
        public:
            using WorkPtr = Work::Ptr;

            template <typename T, typename... Args>
            WorkPtr addWork(Args&&... args)
            {
                auto work = std::make_shared<T>(std::forward<Args>(args)...);
                return addWork(work);
            }

            WorkPtr addWork(WorkPtr work);

            void bookkeeping(gtsam::NonlinearFactorGraph& newFactors,
                             gtsam::FactorIndices& removeIndices,
                             gtsam::Values& initValues);

            void distributeIndices(gtsam::FactorIndices indices);
            void remove(std::function<bool (WorkPtr)> func);
            void erase(std::function<bool (WorkPtr)> func);

            void update();

            void signalNoRelinearize();

            void printWork();
            bool empty() const { return work_.empty(); }
            void clear();

        private:
            std::list<WorkPtr> work_;
            std::map<std::string, WorkPtr> workMap_;
            std::map<std::string, int> lastNewFactor_;

        };

        WorkManager::WorkPtr  WorkManager::addWork(WorkPtr work)
        {
            work_.push_back(work);
            workMap_.insert({work->id(), work});
            return work;
        }

        void WorkManager::bookkeeping(gtsam::NonlinearFactorGraph &newFactors,
                                      gtsam::FactorIndices &removeIndices,
                                      gtsam::Values &initValues)
        {
            lastNewFactor_.clear();
            for (auto& work : work_)
            {
                gtsam::NonlinearFactorGraph addedFactors;
                gtsam::FactorIndices rememberedFactors;
                gtsam::Values initialValues;
                work->bookkeeping(addedFactors, rememberedFactors, initValues);

                if(!addedFactors.empty())
                {
                   lastNewFactor_.insert({work->id(), addedFactors.size()});
                }

                newFactors += addedFactors;
                removeIndices.insert(removeIndices.end(), rememberedFactors.begin(),
                                     rememberedFactors.end());
                initValues.insert(initialValues);
            }
        }

        void WorkManager::distributeIndices(gtsam::FactorIndices indices)
        {
            for (auto& keyValue: lastNewFactor_)
            {
                std::string id = keyValue.first;
                auto work = workMap_[id];
                int n = keyValue.second;

                gtsam::FactorIndices newIndices(indices.begin(), indices.begin()+n);
                if (work)
                {
                    work->lastFactorIndices(newIndices);
                }

                indices.erase(indices.begin(), indices.begin()+n);
            }
            lastNewFactor_.clear();
        }

        void WorkManager::remove(std::function<bool(WorkPtr)> func)
        {
            auto it = std::remove_if(work_.begin(), work_.end(), func);
            for (auto ii = it; ii != work_.end(); ++ii)
            {
                (*ii)->signalRemove();
            }
        }

        void WorkManager::erase(std::function<bool(WorkPtr)> func)
        {
            auto it = std::remove_if(work_.begin(), work_.end(), func);
            work_.erase(it, work_.end());
        }

        void WorkManager::update()
        {
            auto it = work_.begin();
            while (it != work_.end())
            {
                WorkPtr  work = *it;

                auto this_it = it;

                it++;

                work->update();

                if(work->finished())
                {
                    auto child = work->removeChild();
                    if(child)
                    {
                        addWork(child);
                    }
                    workMap_.erase(work->id());
                    work_.erase(this_it);
                }
            }
        }

        void WorkManager::signalNoRelinearize()
        {
            for (auto& work : work_)
            {
                work->signalNoRelinearize();
            }
        }

        void WorkManager::printWork()
        {
        }

        void WorkManager::clear()
        {
            work_.clear();
            workMap_.clear();
            lastNewFactor_.clear();
        }

    } // namespace work

} // namespace msc
#endif //MASTERS_WORK_MANAGER_H
