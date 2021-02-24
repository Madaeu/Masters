set(driver_sources
        camera_interface_factory.cpp)

set(driver_headers
        dataset/dataset_interface.h
        camera_interface.h
        camera_interface_factory.h)

add_library(drivers SHARED ${driver_sources} ${driver_headers})

target_include_directories(drivers PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/dataset
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(drivers PUBLIC
        common)