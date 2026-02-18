#include "raylib.h"
#include "RedeNeural.h"
#include "Include/Matrix/Matrix.h"
#include "Include/Artificial_Neural_Network/Layer.h"
#include <cmath>


using namespace std;

void DrawMatriz(ann::Matrix& matriz, int x, int y, int tamanho_celula);

int main()
{
	double cooldown = 3;
	double timer = 0;
	InitWindow(1200, 1000, "Rede Neural");
	SetTargetFPS(144);

	ann::Matrix mX(2, 2, ann::ProcessingType::Host,  std::vector<float>{
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

	while (!WindowShouldClose()) {
		/*if (timer < cooldown) {
			timer += GetFrameTime();
		}
		else {
			timer = 0;
			mZ.set_element_at(0, 0, GetRandomValue(0, 1));
			mZ.set_element_at(1, 0, GetRandomValue(0, 1));
			result = l1.activation(mZ);
		}*/

		if (IsKeyDown(KEY_SPACE)) {
			mZ(0, 0) = GetRandomValue(0, 1);
			mZ(1, 0) = GetRandomValue(0, 1);
			result = l1.activation(mZ);
		}

		BeginDrawing();
		ClearBackground(BLACK);

		DrawFPS(10, 10);
		DrawText("Rede Neural", 10, 40, 20, WHITE);
		DrawText(TextFormat("Timer: %.2f", cooldown - timer), 10, 80, 20, WHITE);
		DrawMatriz(mX, 600, 200, 40);
		DrawMatriz(mZ, 200, 200, 40);
		DrawMatriz(result, 1000, 200, 40);
		EndDrawing();
	}

	CloseWindow();
	return 0;
}

void DrawMatriz(ann::Matrix& matriz, int x, int y, int tamanho_celula) {
	int espacamento_celula = tamanho_celula / 2;
	int tamanho_font = tamanho_celula / 2;

	double largura = (matriz.get_cols_count() * tamanho_celula) + ((matriz.get_cols_count() - 1) * espacamento_celula);
	double altura = (matriz.get_rows_count() * tamanho_celula) + ((matriz.get_rows_count() - 1) * espacamento_celula);
	double largura_inicial = x - (largura / 2);
	double altura_inicial = y - (altura / 2);
	double largura_final = x + (largura / 2);
	double altura_final = y + (altura / 2);

	for (int i = 0; i < matriz.get_rows_count(); i++) {
		for (int j = 0; j < matriz.get_cols_count(); j++) {
			int cell_y = (i * tamanho_celula) + (i * espacamento_celula) + (tamanho_celula / 2) + altura_inicial;
			int cell_x = (j * tamanho_celula) + (j * espacamento_celula) + (tamanho_celula / 2) + largura_inicial;

			//DrawCircle(cell_x, cell_y, tamanho_celula / 2, RED);
			DrawText(TextFormat("%.0f", matriz(i, j)), cell_x - tamanho_font, cell_y - tamanho_font / 2, tamanho_font, WHITE);
		}
	}
}