cmake_minimum_required(VERSION 3.21)

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(cudaComputeMatrix LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)


if(CUDA_ENABLED)
    enable_language(CUDA)
endif()

set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/samples/common/inc)


#add_executable(cudaComputeMatrix main.cpp)

#set(CUDA_SRC
#        cudaComputeMatrix.cuh
#        cudaComputeMatrix.cu)
#cuda_add_library(CUDA_LIB ${CUDA_SRC}) #使用 cuda_add_library 命令将CUDA源文件编译为一个CUDA库（.lib 或 .a 文件）。
#target_link_libraries(cudaComputeMatrix ${CUDA_CUDA_LIBRARY} CUDA_LIB)


add_executable(global_memory global_memory.cu)
target_link_libraries(global_memory ${CUDA_CUDA_LIBRARY})

add_executable(static_shared_memory static_shared_memory.cu)
target_link_libraries(static_shared_memory ${CUDA_CUDA_LIBRARY})

add_executable(constant_memory constant_memory.cu)
target_link_libraries(constant_memory ${CUDA_CUDA_LIBRARY})

add_executable(GPU_cache GPU_cache.cu)
target_link_libraries(GPU_cache ${CUDA_CUDA_LIBRARY})

add_executable(CUDA_info CUDA_info.cu)
target_link_libraries(CUDA_info ${CUDA_CUDA_LIBRARY})

add_executable(matrix_add_gpu matrix_add_gpu.cu)
target_link_libraries(matrix_add_gpu ${CUDA_CUDA_LIBRARY})