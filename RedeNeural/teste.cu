#include <stdio.h>
#include <cuda_runtime.h>
#include <chrono>

__global__ void test_cuda(float* d_out, int iterations) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    for (int i = 0; i < iterations; i++) {
        val += sinf(idx * i) * cosf(idx * i);
    }

    if (idx < 30720) {
        d_out[idx] = val;
    }
}

int main() {
    int blocks_count = 120;
    int threads_count = 256;
    int times = 1;
    int iterations = 10'000;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device detected: %s\n", prop.name);
    printf("------------------------------------------------\n");

    float *d_ptr;
    float size = blocks_count * threads_count * sizeof(float);
    cudaMalloc((void**)&d_ptr, size);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < times; i++) {
        test_cuda << <blocks_count, threads_count >> > (d_ptr, iterations);
    }
    cudaEventRecord(stop);
    cudaFree(d_ptr);

    cudaEventSynchronize(stop);
    float benchmark = 0;
    cudaEventElapsedTime(&benchmark, start, stop);

    
    printf("------------------------------------------------\n");
    printf("Test completed: %fms\n", benchmark);
    return 0;
}