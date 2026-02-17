#pragma once
#include <vector>
#include <functional>

namespace ann
{
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
        ann::Matrix operator*(ann::Matrix my);
        ann::Matrix operator+(ann::Matrix my);
        float& operator()(int x, int y);
        ann::Matrix& map(const std::function<float(float)>& func);
    };
}
