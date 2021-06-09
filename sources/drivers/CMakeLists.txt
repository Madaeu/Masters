set(driver_sources
        dataset/kitti_interface.cpp
        dataset/scannet_interface.cpp
        camera_interface_factory.cpp)

set(driver_headers
        dataset/dataset_interface.h
        dataset/kitti_interface.h
        dataset/scannet_interface.h
        camera_interface.h
        camera_interface_factory.h)

add_library(msc_drivers SHARED ${driver_sources} ${driver_headers})

target_include_directories(msc_drivers PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/dataset
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(msc_drivers PUBLIC
        msc_common
        Sophus::Sophus
        Eigen3::Eigen
        boost_filesystem)