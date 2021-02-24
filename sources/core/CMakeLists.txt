find_package(OpenCV)

set(core_sources
        vision/camera_tracker.cpp)

set(core_headers
        vision/camera_tracker.h
        slam_system.h)

add_library(core SHARED ${core_sources} ${core_headers})

target_include_directories(core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/vision)

target_link_libraries(core PUBLIC
        common)