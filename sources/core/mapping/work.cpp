//
// Created by madaeu on 5/6/21.
//

#include "work.h"

namespace msc
{
    namespace work
    {

        int Work::nextId_ = 0;

        Work::~Work() { }

        Work::Work()
        {
            id_ = "[" + std::to_string(nextId_++) + "]";
        }

        typename Work::Ptr Work::addChild(Ptr child)
        {
            child_ = child;
            return child;
        }

        typename Work::Ptr Work::removeChild()
        {
            auto child = child_;
            child_.reset();
            return child;
        }

    }

} // namespace msc