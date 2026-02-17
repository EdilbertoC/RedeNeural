#include "Matrix/Matrix.h"
#include <iostream>
#include <stdexcept>

ann::Matrix::Matrix(int rows, int cols)
    : rows(rows), cols(cols), elements(std::vector<float>(rows * cols))
{
    std::cout << "Matrix created!\n";
}

ann::Matrix::Matrix(int rows, int cols, std::vector<float> elements)
    : rows(rows), cols(cols), elements(elements)
{
    if ((rows * cols) != elements.size())
    {
        throw std::invalid_argument("Size does not match the indicated size.");
    }
    std::cout << "Matrix created!\n";
}

int ann::Matrix::get_cols_count() const
{
    return cols;
}

int ann::Matrix::get_rows_count() const
{
    return rows;
}

float& ann::Matrix::operator()(int x, int y)
{
    return elements[x * cols + y];
}

ann::Matrix ann::Matrix::operator*(ann::Matrix my)
{
    ann::Matrix mx = *this;
    if (mx.get_cols_count() != my.get_rows_count())
    {
        throw std::invalid_argument("Invalid multiplication.");
    }

    ann::Matrix result(mx.get_rows_count(), my.get_cols_count());
    for (int row_result = 0; row_result < result.get_rows_count(); row_result++)
    {
        for (int col_result = 0; col_result < result.get_cols_count(); col_result++)
        {
            int element = 0;
            for (int colunaX = 0; colunaX < mx.get_cols_count(); colunaX++)
            {
                element += mx(row_result, colunaX) * my(colunaX, col_result);
            }
            result(row_result, col_result) = element;
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
            float sum = mx(row_result, col_result) + my(row_result, col_result);
            result(row_result, col_result) = sum;
        }
    }
    return result;
}

ann::Matrix& ann::Matrix::map(const std::function<float(float)>& func)
{
    for (int i = 0; i < rows * cols; i++)
    {
        elements[i] = func(elements[i]);
    }
    return *this;
}

ann::Matrix::~Matrix()
{
    std::cout << "Matrix deleted!\n";
}
