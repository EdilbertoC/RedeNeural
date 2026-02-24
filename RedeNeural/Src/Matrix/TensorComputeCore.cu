#include "Matrix/TensorComputeCore.h"
#include <stdexcept>
#include <cuda_runtime.h>
#include <cstdio>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void multiply(const float* d_mx, const float* d_my, float* d_result, const int rows, const int cols, const int cols_mx, const int cols_my)
{
    //x * cols_ + y
    int id = threadIdx.x + blockDim.x * threadIdx.y;

    float sum = 0;
    for (int i = 0; i < cols; i ++)
    {
        //row = [threadIdx.x, i]
        const int mx_id = threadIdx.x * cols_mx + i;
        //col = [i, threadIdx.y]
        const int my_id = i * cols_my + threadIdx.y;
        sum += d_mx[mx_id] * d_my[my_id];
    }
    d_result[id] = sum;
}

bool verify_cuda_device()
{
    int device_count = 0;

    if (cudaError_t const error = cudaGetDeviceCount(&device_count); error != cudaSuccess)
    {
        std::cout << "[CUDA ERROR] Failed to communicate with the GPU: "
                  << cudaGetErrorString(error) << "\n";
        return false;
    }
    else if (device_count == 0)
    {
        std::cout << "[CUDA ERROR] No CUDA GPUs found: "
                  << cudaGetErrorString(error) << "\n";
        return false;
    }else
    {
        return true;
    }
}

ann::Matrix ann::TensorComputeCore::multiply_matrix(const ann::Matrix& mx, const ann::Matrix& my)
{
    if (!verify_cuda_device())
    {
        throw std::runtime_error("No CUDA device connected.");
    }

    ann::Matrix result(mx.get_rows_count(), my.get_cols_count(), ProcessingType::Device);

    std::vector<float> elements_result(mx.get_rows_count(), my.get_cols_count());

    int const size_mx = mx.get_rows_count() * mx.get_cols_count() * sizeof(float);
    int const size_result = elements_result.size() * sizeof(float);
    int const size_my = my.get_rows_count() * my.get_cols_count() * sizeof(float);
    float* d_mx;
    float* d_my;
    float* d_result;

    cudaMalloc((void**)&d_mx, size_mx);
    cudaMalloc((void**)&d_my, size_my);
    cudaMalloc((void**)&d_result, size_result);
    cudaMemcpy(d_mx, mx.get_elements().data(), size_mx, cudaMemcpyHostToDevice);
    cudaMemcpy(d_my, my.get_elements().data(), size_my, cudaMemcpyHostToDevice);

    multiply<<< mx.get_rows_count(), my.get_cols_count() >>>(d_mx, d_my, d_result, result.get_rows_count(), result.get_cols_count(), mx.get_cols_count(), my.get_cols_count());

    cudaMemcpy(elements_result.data(), d_result, size_result, cudaMemcpyDeviceToHost);
    result.set_elements(elements_result);

    cudaFree(d_mx);
    cudaFree(d_my);
    cudaFree(d_result);
    return result;
}
