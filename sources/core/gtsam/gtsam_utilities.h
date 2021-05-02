//
// Created by madaeu on 5/1/21.
//

#ifndef MASTERS_GTSAM_UTILITIES_H
#define MASTERS_GTSAM_UTILITIES_H

#include <gtsam/inference/Symbol.h>
#include <gtsam/inference/Key.h>

inline gtsam::Key auxPoseKey(std::uint64_t j) { return gtsam::Symbol('a', j); }
inline gtsam::Key poseKey(std::uint64_t j) {return gtsam::Symbol('p', j); }
inline gtsam::Key codeKey(std::uint64_t j) {return gtsam::Symbol('c', j); }

#endif //MASTERS_GTSAM_UTILITIES_H
