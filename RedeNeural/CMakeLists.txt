cmake_minimum_required(VERSION 3.18)

project(RedeNeuralProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CORE_SOURCES
        "Src/Matrix/Matrix.cpp"
        "Include/Matrix/Matrix.h"
        "Include/Artificial_Neural_Network/Layer.h"
        "Src/Artificial_Neural_Network/Layer.cpp"
        "Src/Matrix/TensorComputeCore.cpp"
        "Include/Matrix/TensorComputeCore.h"
        "Include/Matrix/ProcessingType.h"
)

add_library(NeuralCore STATIC ${CORE_SOURCES})

target_include_directories(NeuralCore PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/Include"
)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralCore PUBLIC
            raylib
            OpenMP::OpenMP_CXX
            CUDA::cudart
    )
endif()

add_executable(RedeNeural RedeNeural.cpp)
target_link_libraries(RedeNeural PRIVATE NeuralCore)

add_executable(Matrix_Benchmark Matrix_Benchmark.cpp)
target_link_libraries(Matrix_Benchmark PRIVATE NeuralCore)

add_executable(teste_cuda teste.cu)
target_link_libraries(teste_cuda PRIVATE CUDA::cudart)