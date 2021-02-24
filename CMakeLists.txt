cmake_minimum_required(VERSION 3.13)

project(Masters)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party)

find_package(OpenCV )
add_subdirectory(sources)
