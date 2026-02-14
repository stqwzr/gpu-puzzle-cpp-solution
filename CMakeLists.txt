cmake_minimum_required(VERSION 3.18)
set(CMAKE_FIND_ROOT_PATH /opt/cuda)

project(cuda_dev CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_ARCHITECTURES native)

add_executable(puzzle9
        puzzle9_pooling.cu)