add_subdirectory(common)
add_subdirectory(core)
add_subdirectory(drivers)
add_subdirectory(cuda)

#find_package(OpenCV REQUIRED)
find_package(DBoW2 REQUIRED)

add_executable(main main.cpp)
add_executable(brisk brisk.cpp)
add_executable(BoW BoW.cpp)
add_executable(lkopt opticalFlow.cpp)
add_executable(demo demo.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${DBoW2_INCLUDE_DIRS})


target_link_libraries(brisk ${OpenCV_LIBS})
target_link_libraries(BoW ${OpenCV_LIBS})
target_link_libraries(BoW ${DBoW2_LIBS})

target_link_libraries(demo core common drivers masters_cuda ${OpenCV_LIBS})

target_link_libraries(main ${OpenCV_LIBS})

target_link_libraries(lkopt ${OpenCV_LIBS})