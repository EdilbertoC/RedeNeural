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

    std::cout << "Benchmark init.\n";
    ann::Matrix mX(2048, 2048, ann::ProcessingType::Host);
    mX.fill_random(-1, 1);

    ann::Matrix mY(2048, 2048, ann::ProcessingType::Host);
    mY.fill_random(-1, 1);

    std::chrono::time_point const start_host = std::chrono::steady_clock::now();
    ann::Matrix result = mX * mY;
    std::chrono::time_point const end_host = std::chrono::steady_clock::now();
    std::chrono::duration const time_host = end_host - start_host;
    float const ms_host = std::chrono::duration<float, std::milli>(time_host).count();

    printf("Benchmark 1024² Host: %'fms", ms_host);

    return 0;
}
