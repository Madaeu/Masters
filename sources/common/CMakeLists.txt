set(common_headers
        pinhole_camera.h
        camera_pyramid.h
        index_map.h
        transformations.h
        correspondence.h
        M_estimators.h
        lucas_kanade_dense.h
        dense_SFM.h)

add_library(msc_common INTERFACE)
target_include_directories(msc_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(msc_common INTERFACE
        Eigen3::Eigen
        Sophus::Sophus
        VisionCore
        opencv_core)
