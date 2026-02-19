#pragma once
#include <vector>
#include <functional>

#include "ProcessingType.h"

namespace ann
{
    class Matrix
    {
    private:
        int rows_;
        int cols_;
        ProcessingType processing_;
        std::vector<float> elements;

    public:
        Matrix(int rows, int cols, ProcessingType processing);
        Matrix(int rows, int cols, ProcessingType processing, const std::vector<float>& elements);
        ~Matrix();
        [[nodiscard]] int get_rows_count() const;
        [[nodiscard]] int get_cols_count() const;
        ann::Matrix operator*(const ann::Matrix& my) const;
        ann::Matrix operator+(const ann::Matrix& my) const;
        float& operator()(int x, int y);
        float operator()(int x, int y) const;
        ann::Matrix& map(const std::function<float(float)>& func);
        ann::Matrix& fill_randon(int min, int max);
    };
}
