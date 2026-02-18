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
		Layer(int neuron_count, int weight_count, ProcessingType processing);
		Layer(int neuron_count, int weight_count, const ann::Matrix& weights, const ann::Matrix& bias);
		[[nodiscard]] int get_neuron_count() const;
		[[nodiscard]] int get_weight_count() const;
		[[nodiscard]] ann::Matrix get_weights() const;
		[[nodiscard]] ann::Matrix get_bias() const;
		void set_weights(const ann::Matrix& weights);
		void set_bias(const ann::Matrix& bias);
		[[nodiscard]] ann::Matrix activation(const ann::Matrix& input) const;
	};
}