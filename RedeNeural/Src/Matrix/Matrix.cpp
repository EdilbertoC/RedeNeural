#include "Matrix/Matrix.h"
#include <iostream>
#include <stdexcept>

ann::Matrix::Matrix(int rows, int cols)
	: rows(rows), cols(cols), elements(std::vector<float>(rows * cols)) {
	std::cout << "Matrix created!\n";
}

ann::Matrix::Matrix(int rows, int cols, std::vector<float> elements)
	: rows(rows), cols(cols), elements(elements) {
	if ((rows * cols) != elements.size()) {
		throw std::invalid_argument("Size does not match the indicated size.");
	}
	std::cout << "Matrix created!\n";
}

int ann::Matrix::get_cols_count() const {
	return cols;
}

int ann::Matrix::get_rows_count() const {
	return rows;
}

float ann::Matrix::get_element_at(int row, int col) const {
	return elements[row * cols + col];
}

void ann::Matrix::set_element_at(int row, int col, float value) {
	elements[row * cols + col] = value;
}

ann::Matrix ann::Matrix::operator*(ann::Matrix my)
{
	ann::Matrix x = *this;
	if (x.get_cols_count() != my.get_rows_count()) {
		throw std::invalid_argument("Invalid multiplication.");
	}

	ann::Matrix result(x.get_rows_count(), my.get_cols_count());
	for (int row_result = 0; row_result < result.get_rows_count(); row_result++) {
		for (int col_result = 0; col_result < result.get_cols_count(); col_result++) {
			int element = 0;
			for (int colunaX = 0; colunaX < x.get_cols_count(); colunaX++) {
				element += x.get_element_at(row_result, colunaX) * my.get_element_at(colunaX, col_result);
			}
			result.set_element_at(row_result, col_result, element);
		}
	}
	return result;
}

ann::Matrix ann::Matrix::operator+(ann::Matrix my)
{
	ann::Matrix mx = *this;
	if (mx.get_cols_count() != my.get_cols_count() || mx.get_rows_count() != my.get_rows_count())
	{
		throw std::invalid_argument("Invalid Addition");
	}
	ann::Matrix result(mx.get_rows_count(), mx.get_cols_count());
	for (int row_result = 0; row_result < result.get_rows_count(); row_result++)
	{
		for (int col_result = 0; col_result < result.get_cols_count(); col_result++)
		{
			float sum = mx.get_element_at(row_result, col_result) + my.get_element_at(row_result, col_result);
			result.set_element_at(row_result, col_result, sum);
		}
	}
	return result;
}

ann::Matrix::~Matrix() {
	std::cout << "Matrix deleted!\n";
}


