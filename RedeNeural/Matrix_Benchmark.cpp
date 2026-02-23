#include "raylib.h"
#include "RedeNeural.h"
#include "Matrix/Matrix.h"
#include "Artificial_Neural_Network/Layer.h"
#include <cmath>
#include <chrono>

using namespace std;

int main()
{
    setlocale(LC_NUMERIC, "");

    std::cout << "Benchmark init:\n";
    ann::Matrix mX(1024, 1024, ann::ProcessingType::Host);
    mX.fill_random(-1, 1);

    ann::Matrix mY(1024, 1024, ann::ProcessingType::Host);
    mY.fill_random(-1, 1);

    std::chrono::time_point const start_host = std::chrono::steady_clock::now();
    ann::Matrix result = mX * mY;
    std::chrono::time_point const end_host = std::chrono::steady_clock::now();
    std::chrono::duration const time_host = end_host - start_host;
    float const ms_host = std::chrono::duration<float, std::milli>(time_host).count();

    printf("1024² Host: %'fms \n", ms_host);

    ann::Matrix mA(1024, 1024, ann::ProcessingType::Device);
    mA.fill_random(-1, 1);

    ann::Matrix mB(1024, 1024, ann::ProcessingType::Device);
    mB.fill_random(-1, 1);

    std::chrono::time_point const start_device = std::chrono::steady_clock::now();
    ann::Matrix result_device = mA * mB;
    std::chrono::time_point const end_device = std::chrono::steady_clock::now();
    std::chrono::duration const time_device = end_device - start_device;
    float const ms_device = std::chrono::duration<float, std::milli>(time_device).count();

    printf("1024² Device: %'fms", ms_device);

    return 0;
}
