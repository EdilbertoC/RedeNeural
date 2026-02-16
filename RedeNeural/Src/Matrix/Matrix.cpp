#include "Matrix/Matrix.h"
#include <iostream>
#include <stdexcept>

ann::Matrix::Matrix(int linhas, int colunas)
	: linhas(linhas), colunas(colunas), elementos(std::vector<double>(linhas * colunas)) {
	std::cout << "Matrix created!\n";
}

ann::Matrix::Matrix(int linhas, int colunas, std::vector<double> elementos)
	: linhas(linhas), colunas(colunas), elementos(elementos) {
	if ((linhas * colunas) != elementos.size()) {
		throw std::invalid_argument("Size does not match the indicated size.");
	}
	std::cout << "Matrix created!\n";
}

int ann::Matrix::get_cols_count() const {
	return colunas;
}

int ann::Matrix::get_rows_count() const {
	return linhas;
}

double ann::Matrix::get_element_at(int linha, int coluna) const {
	return elementos[linha * colunas + coluna];
}

void ann::Matrix::set_element_at(int linha, int coluna, double valor) {
	elementos[linha * colunas + coluna] = valor;
}

ann::Matrix::~Matrix() {
	std::cout << "Matrix deleted!\n";
}
