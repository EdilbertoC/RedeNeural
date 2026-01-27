#pragma once
#include <vector>
class Matriz
{
private:
	int linhas;
	int colunas;
	std::vector<double> elementos;

public:
	Matriz(int linhas, int colunas);
	Matriz(int linhas, int colunas, std::vector<double> elementos);
	~Matriz();
	int getLinhas() const;
	int getColunas() const;
	double getElemento(int linha, int coluna) const;
	void setElemento(int linha, int coluna, double valor);
};

