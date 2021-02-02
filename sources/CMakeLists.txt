find_package(OpenCV REQUIRED)

add_executable(main main.cpp)
add_executable(brisk brisk.cpp)
add_executable(lkopt opticalFlow.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(brisk ${OpenCV_LIBS})
target_link_libraries(main ${OpenCV_LIBS})
target_link_libraries(lkopt ${OpenCV_LIBS})