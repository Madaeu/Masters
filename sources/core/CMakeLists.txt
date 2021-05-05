set(core_sources
        network/decoder_network.cpp
        mapping/work_implementation.cpp
        slam_system.cpp)

set(core_headers
        vision/camera_tracker.h
        vision/loop_detector.h
        features/feature_detector.h
        gtsam/uniform_sampler.h
        gtsam/gtsam_utilities.h
        gtsam/gtsam_traits.h
        gtsam/photometric_factor.h
        gtsam/reprojection_factor.h
        gtsam/depth_prior_factor.h
        gtsam/factor_graph.h
        mapping/mapper.h
        mapping/frame.h
        mapping/keyframe.h
        mapping/keyframe_map.h
        mapping/work.h
        mapping/work_implementation.h
        mapping/work_manager.h
        network/tfwrap.h
        network/decoder_network.h
        slam_system_options.h
        slam_system.h)

add_library(msc_core SHARED ${core_sources} ${core_headers})

target_include_directories(msc_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/vision
        ${CMAKE_CURRENT_SOURCE_DIR}/features
        ${CMAKE_CURRENT_SOURCE_DIR}/mapping
        ${CMAKE_CURRENT_SOURCE_DIR}/network
        ${CMAKE_CURRENT_SOURCE_DIR}/gtsam
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(msc_core
        PUBLIC
        msc_common
        msc_cuda
        Eigen3::Eigen
        VisionCore
        gtsam
        DBOW2
        PRIVATE
        TensorFlow
        jsoncpp)
