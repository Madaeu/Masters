cmake_minimum_required(VERSION 3.13)

project(Masters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(third_party)

find_package(CUDA QUIET REQUIRED)
find_package(Sophus QUIET REQUIRED)
find_package(OpenCV QUIET REQUIRED)
find_package(Eigen3 QUIET REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS filesystem)
find_package(VisionCore QUIET REQUIRED)

add_subdirectory(sources)
