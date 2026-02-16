cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "/usr/bin/nvcc")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler")

project("RedeNeural" LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(raylib CONFIG REQUIRED)

add_subdirectory("RedeNeural")