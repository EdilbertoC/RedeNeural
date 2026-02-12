#pragma once
#include <vector>
namespace ann {
	class Matrix
	{
	private:
		int linhas;
		int colunas;
		std::vector<double> elementos;

	public:
		Matrix(int linhas, int colunas);
		Matrix(int linhas, int colunas, std::vector<double> elementos);
		~Matrix();
		int get_rows_count() const;
		int get_cols_count() const;
		double get_element_at(int linha, int coluna) const;
		void set_element_at(int linha, int coluna, double valor);
	};
}

