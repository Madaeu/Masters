//
// Created by madaeu on 3/9/21.
//

#ifndef MASTERS_INDEXED_MAP_H
#define MASTERS_INDEXED_MAP_H

#include <map>
#include <vector>

namespace msc {
    template<typename Item, typename IdType>
    class IndexedMap {
    public:
        using ContainerT = std::map<IdType, Item>;
        using IdList = std::vector<IdType>;

        IndexedMap() : lastID_(0) {}

        virtual Item get(const IdType &id) {
            return map_[id];
        }

        virtual const Item get(const IdType &id) const {
            return map_.at(id);
        }

        virtual bool exists(const IdType &id) const {
            return map_.find(id) != map_.end();
        }

        virtual IdType add(const Item &item) {
            map_[++lastID_] = item;
            ids_.push_back(lastID_);
            return lastID_;
        }

        virtual void remove(IdType id) {
            map_.erase(id);
            std::remove(ids_.begin(), ids_.end(), lastID_);
        }

        virtual void clear() {
            map_.clear();
            ids_.clear();
            lastID_ = 0;
        }

        virtual std::size_t size() const { return map_.size(); }

        virtual const IdList &ids() const { return ids_; }

        virtual const IdType lastID() const { return lastID_; }

    protected:
        IdType lastID_;
        ContainerT map_;
        IdList ids_;
    };
} //namespace msc
#endif //MASTERS_INDEXED_MAP_H
