//
// Created by madaeu on 3/9/21.
//

#ifndef MASTERS_KEYFRAME_MAP_H
#define MASTERS_KEYFRAME_MAP_H

#include "indexed_map.h"
#include "frame.h"
#include "keyframe.h"

#include <map>
#include <vector>
#include <memory>

template <typename FrameT, typename IdType = typename FrameT::IdType>
class FrameGraph : public IndexedMap<FrameT, IdType>
{
public:
    using Base = IndexedMap<FrameT, IdType>;
    using LinkT = std::pair<IdType, IdType>;
    using LinkContainer = std::vector<LinkT>;

    void addLink(IdType firstItem, IdType secondItem){
        links_.template emplace_back(firstItem, secondItem);
    }

    void remove(IdType id) override
    {
        Base::remove(id);
        for (int i = 0; i < links_.size(); i++) {
            if (links_[i].first == id || links_[i].second == id){
                links_.erase(links_.begin() + i);
            }
        }
    }

    void clear() override
    {
        Base::clear();
        links_.clear();
    }

    LinkContainer& getLinks() { return links_; }

    FrameT& last() { return this->map_[this->lastID()]; }

    std::vector<IdType> getConnections(IdType id, bool directed = false)
    {
        std::vector<IdType> connections;
        for (auto& c: links_)
        {
            if (c.first == id) {
                connections.push_back(c.second);
            }
            if (c.second == id && !directed) {
                connections.push_back(c.first);
            }
        }
        return connections;
    }

    bool linkExists( IdType firstID, IdType secondID){
        for (auto& c: links_){
            if ((c.first == firstID && c.second == secondID) ||
                (c.first == secondID && c.second == firstID)){
                return true;
            }
        }
        return false;
    }

    typename Base::ContainerT::iterator begin() { return this->map_.begin(); }
    typename Base::ContainerT::iterator end() { return this->map_.end(); }

private:
    LinkContainer links_;
};

template <typename Scalar>
class Map
{
public:
    using This = Map<Scalar>;
    using Ptr = std::shared_ptr<This>;
    using FrameT = Frame<Scalar>;
    using KeyframeT = Keyframe<Scalar>;
    using FrameID = typename FrameT::IdType;
    using KeyframePtr = typename KeyframeT::Ptr;
    using FramePtr = typename FrameT::Ptr;
    using FrameGraphT = FrameGraph<FramePtr, FrameID>;
    using KeyframeGraphT = FrameGraph<KeyframePtr, FrameID>;

    void clear()
    {
        frames_.clear();
        keyframes_.clear();
    }

    void addFrame( FramePtr frame) {frame->id_ = frames_.add(frame); }
    void addKeyframe( KeyframePtr keyframe) { keyframe->id_ = keyframes_.add(keyframe); }

    std::size_t numberOfKeyframes() const { return keyframes_.size(); }
    std::size_t numberOfFrames() const { return frames_.size(); }

    FrameGraphT frames_;
    KeyframeGraphT keyframes_;
};
#endif //MASTERS_KEYFRAME_MAP_H
