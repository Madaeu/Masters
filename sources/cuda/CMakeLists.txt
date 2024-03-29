set(CUDA_ARCH Auto)
CUDA_SELECT_NVCC_ARCH_FLAGS(CUDA_NVCC_ARCH_FLAGS ${CUDA_ARCH})
set(CMAKE_CUDA_FLAGS ${CUDA_NVCC_ARCH_FLAGS};--expt-relaxed-constexpr;--expt-extended-lambda;--use_fast_math)

# append debug flags
if(CMAKE_BUILD_TYPE MATCHES Debug)
    #list(APPEND CMAKE_CUDA_FLAGS --device-debug;--debug;-Xcompiler;-rdynamic;)#--ptxas-options=-v)
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
    list(APPEND CMAKE_CUDA_FLAGS -g;-lineinfo)
endif()

string(REPLACE ";" " " _TMP_STR "${CMAKE_CUDA_FLAGS}")
set(CMAKE_CUDA_FLAGS "${_TMP_STR}")

set(cuda_sources
        cuda_image_proc.cpp
        cuda_SE3_aligner.cpp
        cuda_SFM_aligner.cpp
        cuda_depth_aligner.cpp
        cuda_context.cpp)

set(cuda_headers
        device_info.h
        cuda_context.h
        kernel_utilities.h
        launch_utilities.h
        reduction_items.h
        synced_pyramid.h
        cuda_SE3_aligner.h
        cuda_SFM_aligner.h
        cuda_depth_aligner.h
        cuda_image_proc.h)

set_source_files_properties(${cuda_sources} PROPERTIES LANGUAGE CUDA)

add_library(msc_cuda SHARED ${cuda_sources} ${cuda_headers})
target_include_directories(msc_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(msc_cuda PUBLIC
        msc_common
        VisionCore
        Eigen3::Eigen
        cuda)

set_property(TARGET msc_cuda PROPERTY CUDA_STANDARD 17)
set_target_properties(msc_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(msc_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(msc_cuda PROPERTIES RELOCATABLE_DEVICE_CODE ON)