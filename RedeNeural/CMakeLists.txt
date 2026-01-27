cmake_minimum_required(VERSION 3.12)
project(RedeNeuralProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP REQUIRED)

set(SOURCES
    "RedeNeural.cpp"
    "RedeNeural.h"
    "Src/Matriz/Matriz.cpp"       # <--- Adicionado
    "Include/Matriz/Matriz.h"
)
add_executable(RedeNeural ${SOURCES})

target_include_directories(RedeNeural PRIVATE 
    "${CMAKE_CURRENT_SOURCE_DIR}/Include"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(RedeNeural PUBLIC 
        raylib 
        OpenMP::OpenMP_CXX
    )
endif()
