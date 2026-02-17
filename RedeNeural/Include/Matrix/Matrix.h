#pragma once
#include <vector>
namespace ann {
	class Matrix
	{
	private:
		int rows;
		int cols;
		std::vector<float> elements;

	public:
		Matrix(int rows, int cols);
		Matrix(int rows, int cols, std::vector<float> elements);
		~Matrix();
		int get_rows_count() const;
		int get_cols_count() const;
		float get_element_at(int row, int col) const;
		void set_element_at(int row, int col, float value);
		ann::Matrix operator*(ann::Matrix my);
		ann::Matrix operator+(ann::Matrix my);

	};
}

