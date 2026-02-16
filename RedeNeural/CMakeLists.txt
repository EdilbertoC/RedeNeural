cmake_minimum_required(VERSION 3.18)

project(RedeNeuralProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(SOURCES
    "RedeNeural.cpp"
    "RedeNeural.h"
    "Src/Matrix/Matrix.cpp" 
    "Include/Matrix/Matrix.h"
    "Include/Artificial_Neural_Network/Layer.h"
    "Src/Artificial_Neural_Network/Layer.cpp"
)

add_executable(RedeNeural ${SOURCES})


target_include_directories(RedeNeural PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/Include"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(RedeNeural PUBLIC 
        raylib 
        OpenMP::OpenMP_CXX
        CUDA::cudart 
    )
endif()

add_executable(teste_cuda teste.cu)
# Linka com as bibliotecas do CUDA que carregamos
target_link_libraries(teste_cuda PRIVATE CUDA::cudart)