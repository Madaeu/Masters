set(common_headers
        pinhole_camera.h
        camera_pyramid.h)

add_library(common INTERFACE)
target_include_directories(common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(common INTERFACE
        opencv_core)
