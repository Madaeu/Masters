#find_package(OpenCV)

set(core_sources
        network/decoder_network.cpp)

set(core_headers
        vision/camera_tracker.h
        features/feature_detector.h
        gtsam/uniform_sampler.h
        gtsam/gtsam_utilities.h
        gtsam/gtsam_traits.h
        gtsam/photometric_factor.h
        mapping/mapper.h
        mapping/frame.h
        mapping/keyframe.h
        mapping/keyframe_map.h
        mapping/work.h
        network/tfwrap.h
        network/decoder_network.h
        slam_system.h)

#add_library(core SHARED ${core_sources} ${core_headers})
add_library(msc_core SHARED ${core_sources} ${core_headers})

target_include_directories(msc_core INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/vision
        ${CMAKE_CURRENT_SOURCE_DIR}/features
        ${CMAKE_CURRENT_SOURCE_DIR}/mapping
        ${CMAKE_CURRENT_SOURCE_DIR}/network
        ${CMAKE_CURRENT_SOURCE_DIR}/gtsam)

target_link_libraries(msc_core PUBLIC
        msc_common
        msc_cuda
        VisionCore
        TensorFlow
        gtsam
        jsoncpp)
