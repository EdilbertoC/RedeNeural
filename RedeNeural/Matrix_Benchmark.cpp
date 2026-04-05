#include "raylib.h"
#include "RedeNeural.h"
#include "Matrix/Matrix.h"
#include "Artificial_Neural_Network/Layer.h"
#include <cmath>
#include <chrono>
#include <cuda_runtime_api.h>

using namespace std;

int main()
{
    constexpr size_t magnitude = 1024;
    std::cout.imbue(std::locale(""));

    std::cout << "Benchmark init:\n";
    ann::Matrix mX(magnitude, magnitude, ann::ProcessingType::Host);
    mX.fill_random(-1, 1);

    ann::Matrix mY(magnitude, magnitude, ann::ProcessingType::Host);
    mY.fill_random(-1, 1);

    std::chrono::time_point const start_host = std::chrono::steady_clock::now();
    ann::Matrix result = mX * mY;
    std::chrono::time_point const end_host = std::chrono::steady_clock::now();
    std::chrono::duration const time_host = end_host - start_host;
    float const ms_host = std::chrono::duration<float, std::milli>(time_host).count();

    std::cout << magnitude << "^2 x " << magnitude << "^2 Host: " << ms_host << " ms\n";

    ann::Matrix mX_device(magnitude, magnitude, ann::ProcessingType::Device);
    mX_device.fill_random(-1, 1);

    ann::Matrix mY_device(magnitude, magnitude, ann::ProcessingType::Device);
    mY_device.fill_random(-1, 1);

    std::chrono::time_point const start_device = std::chrono::steady_clock::now();
    ann::Matrix result_device = mX_device * mY_device;
    cudaDeviceSynchronize();
    std::chrono::time_point const end_device = std::chrono::steady_clock::now();
    std::chrono::duration const time_device = end_device - start_device;
    float const ms_device = std::chrono::duration<float, std::milli>(time_device).count();

    std::cout << magnitude << "^2 x " << magnitude << "^2 Device: " << ms_device << " ms\n";
    return 0;
}
