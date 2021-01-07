find_package(OpenCV REQUIRED)


add_executable(main main.cpp)
add_executable(brisk brisk.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(brisk ${OpenCV_LIBS})