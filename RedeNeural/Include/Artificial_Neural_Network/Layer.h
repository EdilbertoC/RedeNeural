#pragma once
#include <vector>
#include "Matrix/Matrix.h"

namespace ann {
	class Layer {
	private:
		int neuron_count_;
		int weight_count_;
		ann::Matrix weights_;
		ann::Matrix bias_;
	public:
		Layer(int neuron_count, int weight_count);
		Layer(int neuron_count, int weight_count, ann::Matrix weights, ann::Matrix bias);
		int get_neuron_count();
		int get_weight_count();
		ann::Matrix get_weights();
		ann::Matrix get_bias();
		void set_weights(ann::Matrix weights);
		void set_bias(ann::Matrix bias);
		ann::Matrix activation(ann::Matrix& input);
	};
}