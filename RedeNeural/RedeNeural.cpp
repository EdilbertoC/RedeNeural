#include "raylib.h"
#include "RedeNeural.h"
#include "Include/Matriz/Matriz.h"
#include <cmath>

using namespace std;

Matriz multplicarMatrizes(Matriz x, Matriz y);
void DrawMatriz(const Matriz& matriz, int x, int y, int tamanho_celula);

int main()
{
	InitWindow(1200, 1000, "Rede Neural");
	SetTargetFPS(144);

	Matriz mX(3, 3, std::vector<double>{
		1, 2, 3,
			4, 5, 6,
			7, 8, 9
	});

	Matriz mZ(3, 5, std::vector<double>{
		1, 2, 3, 4, 5,
			6, 7, 8, 9, 10,
			11, 12, 13, 14, 15
	});

	Matriz resultado = multplicarMatrizes(mX, mZ);

	while (!WindowShouldClose()) {

		BeginDrawing();
		ClearBackground(BLACK);

		DrawFPS(10, 10);
		DrawText("Rede Neural", 10, 40, 20, WHITE);
		DrawMatriz(mX, 200, 500, 40);
		DrawMatriz(mZ, 600, 500, 40);
		DrawMatriz(resultado, 1000, 500, 40);
		EndDrawing();
	}

	CloseWindow();
	return 0;
}

void DrawMatriz(const Matriz& matriz, int x, int y, int tamanho_celula) {
	int espacamento_celula = tamanho_celula / 2;
	int tamanho_font = tamanho_celula / 2;

	double largura = (matriz.getColunas() * tamanho_celula) + ((matriz.getColunas() - 1) * espacamento_celula);
	double altura = (matriz.getLinhas() * tamanho_celula) + ((matriz.getLinhas() - 1) * espacamento_celula);
	double largura_inicial = x - (largura / 2);
	double altura_inicial = y - (altura / 2);
	double largura_final = x + (largura / 2);
	double altura_final = y + (altura / 2);

	for (int i = 0; i < matriz.getLinhas(); i++) {
		for (int j = 0; j < matriz.getColunas(); j++) {
			int cell_y = (i * tamanho_celula) + (i * espacamento_celula) + (tamanho_celula / 2) + altura_inicial;
			int cell_x = (j * tamanho_celula) + (j * espacamento_celula) + (tamanho_celula / 2) + largura_inicial;

			//DrawCircle(cell_x, cell_y, tamanho_celula / 2, RED);
			DrawText(TextFormat("%.0f", matriz.getElemento(i, j)), cell_x - tamanho_font, cell_y - tamanho_font / 2, tamanho_font, WHITE);
		}
	}
}

Matriz multplicarMatrizes(Matriz x, Matriz y) {
	Matriz resultado(x.getLinhas(), y.getColunas());
	for (int linhaResultado = 0; linhaResultado < resultado.getLinhas(); linhaResultado++) {
		for (int colunaResultado = 0; colunaResultado < resultado.getColunas(); colunaResultado++) {
			int elemento = 0;
			for (int colunaX = 0; colunaX < x.getColunas(); colunaX++) {
				elemento += x.getElemento(linhaResultado, colunaX) * y.getElemento(colunaX, colunaResultado);
			}
			resultado.setElemento(linhaResultado, colunaResultado, elemento);
		}
	}
	return resultado;
}
