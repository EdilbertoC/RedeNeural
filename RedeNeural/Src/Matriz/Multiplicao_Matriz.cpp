#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include "Matriz.h"

void printMatriz(Matriz matriz);
Matriz multplicarMatrizes(Matriz x, Matriz y);
void printBenchmark(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end);

int main()
{
	Matriz matrizX(3, 3, {
		1, -1, 1,
		-1, 1, -1,
		1, -1, 1
	});

	Matriz matrizY(3, 3, {
		2, 2, 2,
		2, 2, 2,
		2, 2, 2
	});

	std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();

	printMatriz(multplicarMatrizes(matrizX, matrizY));

	std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();

	printBenchmark(start, end);
}

void printBenchmark(std::chrono::high_resolution_clock::time_point start, std::chrono::high_resolution_clock::time_point end) {
	std::chrono::high_resolution_clock::duration duracao = end - start;
	std::cout << "Tempo de execucao: " << duracao;

	std::ofstream arquivoLog("log.txt", std::ios::app);

	if (arquivoLog.is_open()) {
		arquivoLog << duracao << "º\n";
		arquivoLog.close();
	}
}

void printMatriz(Matriz matriz) {
	for (int i = 0; i < matriz.getLinhas(); i++)
	{
		for (int j = 0; j < matriz.getColunas(); j++)
		{
			std::cout << matriz.getElemento(i, j);
			if ((j + 1) % matriz.getLinhas() == 0) {
				std::cout << "\n";
			}
			else {
				std::cout << " | ";
			}
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