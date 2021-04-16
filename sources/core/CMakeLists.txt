#find_package(OpenCV)

#set(core_sources)

set(core_headers
        vision/camera_tracker.h
        features/feature_detector.h
        mapping/mapper.h
        mapping/frame.h
        mapping/keyframe.h
        mapping/keyframe_map.h
        slam_system.h)

#add_library(core SHARED ${core_sources} ${core_headers})
add_library(core INTERFACE)

target_include_directories(core INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/vision
        ${CMAKE_CURRENT_SOURCE_DIR}/features
        ${CMAKE_CURRENT_SOURCE_DIR}/mapping)

target_link_libraries(core INTERFACE
        common
        masters_cuda
        VisionCore)
