#include <stdio.h>
#include <cuda_runtime.h>

__global__ void test_cuda() {
    printf("Thread: %d\n", threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device detected: %s\n", prop.name);
    printf("------------------------------------------------\n");

    int blocks_count = 120;
    int threads_count = 256;
    test_cuda<<<blocks_count, threads_count>>>();

    cudaDeviceSynchronize();

    printf("------------------------------------------------\n");
    printf("Test completed successfully.!\n");
    return 0;
}