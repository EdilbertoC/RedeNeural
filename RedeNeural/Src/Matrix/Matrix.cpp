#include "Matrix/Matrix.h"
#include <iostream>
#include <stdexcept>

ann::Matrix::Matrix(const int rows, const int cols, const ProcessingType processing)
    : rows_(rows), cols_(cols), processing_(processing), elements(std::vector<float>(rows * cols))
{
    std::cout << "Matrix created!\n";
}

ann::Matrix::Matrix(const int rows, const int cols, const ProcessingType processing, const std::vector<float>& elements)
    : rows_(rows), cols_(cols), processing_(processing), elements(elements)
{
    if ((rows * cols) != elements.size())
    {
        throw std::invalid_argument("Size does not match the indicated size.");
    }
    std::cout << "Matrix created!\n";
}

int ann::Matrix::get_cols_count() const
{
    return cols_;
}

int ann::Matrix::get_rows_count() const
{
    return rows_;
}

float& ann::Matrix::operator()(const int x,const int y)
{
    return elements[x * cols_ + y];
}

float ann::Matrix::operator()(const int x,const int y) const
{
    return elements[x * cols_ + y];
}

ann::Matrix ann::Matrix::operator*(const ann::Matrix& my) const
{
    ann::Matrix mx = *this;
    if (mx.get_cols_count() != my.get_rows_count())
    {
        throw std::invalid_argument("Invalid multiplication.");
    }
    if (processing_ == ProcessingType::Host)
    {
        ann::Matrix result(mx.get_rows_count(), my.get_cols_count(), ProcessingType::Host);
        for (int row_result = 0; row_result < result.get_rows_count(); row_result++)
        {
            for (int col_result = 0; col_result < result.get_cols_count(); col_result++)
            {
                float element = 0;
                for (int col_x = 0; col_x < mx.get_cols_count(); col_x++)
                {
                    element += mx(row_result, col_x) * my(col_x, col_result);
                }
                result(row_result, col_result) = element;
            }
        }
        return result;
    }else
    {
        throw std::invalid_argument("WIP");
    }
}

ann::Matrix ann::Matrix::operator+(const ann::Matrix& my) const
{
    ann::Matrix mx = *this;
    if (mx.get_cols_count() != my.get_cols_count() || mx.get_rows_count() != my.get_rows_count())
    {
        throw std::invalid_argument("Invalid Addition");
    }
    if (processing_ == ProcessingType::Host)
    {
        ann::Matrix result(mx.get_rows_count(), mx.get_cols_count(), ProcessingType::Host);
        for (int row_result = 0; row_result < result.get_rows_count(); row_result++)
        {
            for (int col_result = 0; col_result < result.get_cols_count(); col_result++)
            {
                const float sum = mx(row_result, col_result) + my(row_result, col_result);
                result(row_result, col_result) = sum;
            }
        }
        return result;
    }else
    {
        throw std::invalid_argument("WIP");
    }
}

ann::Matrix& ann::Matrix::map(const std::function<float(float)>& func)
{
    if (processing_ == ProcessingType::Host)
    {
    for (int i = 0; i < rows_ * cols_; i++)
    {
        elements[i] = func(elements[i]);
    }
    return *this;
    }else
    {
        throw std::invalid_argument("WIP");
    }
}

ann::Matrix& ann::Matrix::fill_randon(int min, int max)
{
    for (int i = 0; i < rows_ * cols_; i++)
    {
        elements[i] = 10;
    }

    return *this;
}

ann::Matrix::~Matrix()
{
    std::cout << "Matrix deleted!\n";
}
