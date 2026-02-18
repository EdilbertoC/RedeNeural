#include "Artificial_Neural_Network/Layer.h"
#include <stdexcept>
#include <functional>
#include <cmath>

float sigmoid(float x);

ann::Layer::Layer(const int neuron_count, const int weight_count, const ProcessingType processing)
	: neuron_count_(neuron_count),
	weight_count_(weight_count),
	weights_(ann::Matrix(neuron_count, weight_count, processing)),
	bias_(ann::Matrix(neuron_count, 1, processing))
{
}

ann::Layer::Layer(const int neuron_count, const int weight_count, const ann::Matrix& weights, const ann::Matrix& bias)
	: neuron_count_(neuron_count), weight_count_(weight_count), weights_(weights), bias_(bias)
{
}

int ann::Layer::get_neuron_count() const
{
	return neuron_count_;
}

int ann::Layer::get_weight_count() const
{
	return weight_count_;
}

ann::Matrix ann::Layer::get_weights() const
{
	return weights_;
}

ann::Matrix ann::Layer::get_bias() const
{
	return bias_;
}

void ann::Layer::set_weights(const ann::Matrix& weights)
{
	weights_ = weights;
}

void ann::Layer::set_bias(const ann::Matrix& bias)
{
	bias_ = bias;
}

ann::Matrix ann::Layer::activation(const ann::Matrix& input) const
{
	if (input.get_cols_count() != 1) {
		throw std::invalid_argument("WIP: It does not support input matrix with more than one column.");
	}
	if (input.get_rows_count() != weights_.get_cols_count()) {
		throw std::invalid_argument("The input matrix cannot have more rows than the weight matrix.");

	}

	ann::Matrix result = weights_ * input + bias_;
	return result.map(sigmoid);
}

float sigmoid(float x) {
	return 1 / (1 + (std::exp(-x)));
}


