#include "Artificial_Neural_Network/Layer.h"
#include <stdexcept>
#include <functional>
#include <cmath>

ann::Matrix multiply_matrices(ann::Matrix& x, ann::Matrix& y);
ann::Matrix sum_matrices(ann::Matrix& x, ann::Matrix& y);
ann::Matrix map_matrix(ann::Matrix& matrix, std::function<double(double)> func);
double sigmoid(double x);


ann::Layer::Layer(int neuron_count, int weight_count)
	: neuron_count_(neuron_count),
	weight_count_(weight_count),
	weights_(ann::Matrix(neuron_count, weight_count)),
	bias_(ann::Matrix(neuron_count, 1))
{
}

ann::Layer::Layer(int neuron_count, int weight_count, ann::Matrix weights, ann::Matrix bias)
	: neuron_count_(neuron_count), weight_count_(weight_count), weights_(weights), bias_(bias)
{
}

int ann::Layer::get_neuron_count()
{
	return neuron_count_;
}

int ann::Layer::get_weight_count()
{
	return weight_count_;
}

ann::Matrix ann::Layer::get_weights()
{
	return weights_;
}

ann::Matrix ann::Layer::get_bias()
{
	return bias_;
}

void ann::Layer::set_weights(ann::Matrix weights)
{
	weights_ = weights;
}

void ann::Layer::set_bias(ann::Matrix bias)
{
	bias_ = bias;
}

ann::Matrix ann::Layer::activation(ann::Matrix& input)
{
	if (input.get_cols_count() != 1) {
		throw std::invalid_argument("WIP: It does not support input matrix with more than one column.");
	}
	if (input.get_rows_count() != weights_.get_cols_count()) {
		throw std::invalid_argument("The input matrix cannot have more rows than the weight matrix.");

	}

	ann::Matrix input_weight = multiply_matrices(weights_, input);
	ann::Matrix result = sum_matrices(input_weight, bias_);
	result = map_matrix(result, sigmoid);
	return result;
}

ann::Matrix multiply_matrices(ann::Matrix& x, ann::Matrix& y) {
	if (x.get_cols_count() != y.get_rows_count()) {
		throw std::invalid_argument("Invalid multiplication.");
	}

	ann::Matrix result(x.get_rows_count(), y.get_cols_count());
	for (int row_result = 0; row_result < result.get_rows_count(); row_result++) {
		for (int col_result = 0; col_result < result.get_cols_count(); col_result++) {
			int element = 0;
			for (int colunaX = 0; colunaX < x.get_cols_count(); colunaX++) {
				element += x.get_element_at(row_result, colunaX) * y.get_element_at(colunaX, col_result);
			}
			result.set_element_at(row_result, col_result, element);
		}
	}
	return result;
}

ann::Matrix sum_matrices(ann::Matrix& x, ann::Matrix& y) {
	if (x.get_cols_count() != y.get_cols_count() && x.get_rows_count() != y.get_rows_count()) {
		throw std::invalid_argument("Invalid sum.");
	}

	ann::Matrix result(x.get_cols_count(), x.get_rows_count());
	for (int row_result = 0; row_result < result.get_rows_count(); row_result++) {
		for (int col_result = 0; col_result < result.get_cols_count(); col_result++) {
			double element_result = x.get_element_at(row_result, col_result) + y.get_element_at(row_result, col_result);
			result.set_element_at(row_result, col_result, element_result);
		}
	}
	return result;
}

ann::Matrix map_matrix(ann::Matrix& matrix, std::function<double(double)> func) {
	ann::Matrix result(matrix.get_rows_count(), matrix.get_cols_count());
	for (int i = 0; i < matrix.get_rows_count(); i++) {
		for (int j = 0; j < matrix.get_cols_count(); j++) {
			double element = matrix.get_element_at(i, j);
			element = func(element);
			result.set_element_at(i, j, element);
		}
	}
	return result;
}

double sigmoid(double x) {
	return 1 / (1 + (std::exp(-x)));
}


