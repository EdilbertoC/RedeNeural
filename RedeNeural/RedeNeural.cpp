#include "raylib.h"
#include "RedeNeural.h"
#include "Include/Matrix/Matrix.h"
#include "Include/Artificial_Neural_Network/Layer.h"
#include <cmath>
#include <chrono>


using namespace std;

void DrawMatrix(const ann::Matrix& matrix, int x, int y, int cell_size);

int main()
{
    setlocale(LC_NUMERIC, "");
    float ns_layer_calc = 0;
    constexpr double cooldown = 0.2;
    double timer = cooldown;
    InitWindow(1200, 1000, "Rede Neural");
    SetTargetFPS(144);

    ann::Matrix mR(512, 512, ann::ProcessingType::Host);
    mR.fill_randon(0, 10);

    ann::Matrix mX(2, 2, ann::ProcessingType::Host, std::vector<float>{
                       1, 1,
                       1, 1
                   });

    ann::Matrix mZ(2, 1, ann::ProcessingType::Host, std::vector<float>{
                       1,
                       0
                   });

    ann::Matrix bias(2, 1, ann::ProcessingType::Host, std::vector<float>{
                         -1,
                         0
                     });

    ann::Layer l1(2, 2, mX, bias);

    ann::Matrix result = l1.activation(mZ);

    while (!WindowShouldClose())
    {
        if (IsKeyDown(KEY_ENTER) && timer >= cooldown)
        {
            timer = 0;
            mZ(0, 0) = GetRandomValue(0, 1);
            mZ(1, 0) = GetRandomValue(0, 1);

            std::chrono::time_point start_layer_calc = std::chrono::steady_clock::now();
            result = l1.activation(mZ);
            std::chrono::time_point end_layer_calc = std::chrono::steady_clock::now();
            std::chrono::duration duration_layer_calc = end_layer_calc - start_layer_calc;
            ns_layer_calc = std::chrono::duration_cast<std::chrono::nanoseconds>(duration_layer_calc).count();
        }
        else if (timer < cooldown)
        {
            timer += GetFrameTime();
        }

        BeginDrawing();
        ClearBackground(BLACK);

        DrawFPS(10, 10);
        DrawText("Rede Neural", 10, 40, 20, WHITE);
        DrawText(TextFormat("Timer: %.2f", cooldown - timer), 10, 80, 20, WHITE);
        DrawText(TextFormat("Duration Layer 1 Calc: %'.0fns", ns_layer_calc), 10, 120, 20, WHITE);
        DrawMatrix(mR, 600, 400, 10);
        DrawMatrix(mZ, 200, 400, 1);
        DrawMatrix(result, 1000, 400, 10);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}

void DrawMatrix(const ann::Matrix& matrix, const int x, const int y, const int cell_size)
{
    const int cell_spacing = cell_size / 2;
    const int font_size = cell_size / 2;
    const double width = (matrix.get_cols_count() * cell_size) + ((matrix.get_cols_count() - 1) * cell_spacing);
    const double height = (matrix.get_rows_count() * cell_size) + ((matrix.get_rows_count() - 1) * cell_spacing);
    const double initial_width = x - (width / 2);
    const double initial_height = y - (height / 2);

    for (int i = 0; i < matrix.get_rows_count(); i++)
    {
        for (int j = 0; j < matrix.get_cols_count(); j++)
        {
            const int cell_y = (i * cell_size) + (i * cell_spacing) + (cell_size / 2) + initial_height;
            const int cell_x = (j * cell_size) + (j * cell_spacing) + (cell_size / 2) + initial_width;

            DrawText(TextFormat("%.0f", matrix(i, j)), cell_x - font_size, cell_y - font_size / 2, font_size, WHITE);
        }
    }
}
