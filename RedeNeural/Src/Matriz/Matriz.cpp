#include "/Projetos/Pessoais/RedeNeural/RedeNeural/Include/Matriz/Matriz.h"
#include <iostream>

Matriz::Matriz(int linhas, int colunas)
	: linhas(linhas), colunas(colunas), elementos(std::vector<double>(linhas * colunas)) {
	std::cout << "Matriz criada!\n";
}

Matriz::Matriz(int linhas, int colunas, std::vector<double> elementos)
	: linhas(linhas), colunas(colunas), elementos(elementos) {
	std::cout << "Matriz criada!\n";
}

int Matriz::getColunas() const {
	return colunas;
}

int Matriz::getLinhas() const {
	return linhas;
}

double Matriz::getElemento(int linha, int coluna) const {
	return elementos[linha * colunas + coluna];
}

void Matriz::setElemento(int linha, int coluna, double valor) {
	elementos[linha * colunas + coluna] = valor;
}

Matriz::~Matriz() {
	std::cout << "Matriz deletada!\n";
}
