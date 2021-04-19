cmake_minimum_required(VERSION 3.10.2)

project(Masters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>)
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-variable>)

add_definitions(-DEIGEN_DONT_VECTORIZE
        -DEIGEN_DISABLE_UNALIGNED_ARRAY_ASSERT
        -DEIGEN_DONT_ALIGN)

message("Project: ${PROJECT_NAME}")
message("Build type: ${CMAKE_BUILD_TYPE}")
message("Compiled on: ${CMAKE_SYSTEM}")

add_subdirectory(third_party)

find_package(CUDA QUIET REQUIRED)
find_package(Sophus QUIET REQUIRED)
find_package(OpenCV QUIET REQUIRED)
find_package(Eigen3 QUIET REQUIRED)
find_package(Boost QUIET REQUIRED COMPONENTS filesystem)
find_package(VisionCore QUIET REQUIRED)

add_subdirectory(sources)
