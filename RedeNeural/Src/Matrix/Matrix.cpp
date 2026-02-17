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

ann::Matrix ann::Matrix::operator*(ann::Matrix mx)
{
	ann::Matrix x = *this;
	if (x.get_cols_count() != mx.get_rows_count()) {
		throw std::invalid_argument("Invalid multiplication.");
	}

	ann::Matrix result(x.get_rows_count(), mx.get_cols_count());
	for (int row_result = 0; row_result < result.get_rows_count(); row_result++) {
		for (int col_result = 0; col_result < result.get_cols_count(); col_result++) {
			int element = 0;
			for (int colunaX = 0; colunaX < x.get_cols_count(); colunaX++) {
				element += x.get_element_at(row_result, colunaX) * mx.get_element_at(colunaX, col_result);
			}
			result.set_element_at(row_result, col_result, element);
		}
	}
	return result;
}

ann::Matrix::~Matrix() {
	std::cout << "Matrix deleted!\n";
}


