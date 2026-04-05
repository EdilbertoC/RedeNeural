#pragma once
#include "Matrix.h"
namespace ann
{

    class TensorComputeCore
    {
    public:
        TensorComputeCore() = delete;
        static ann::Matrix multiply_matrix(const ann::Matrix& mx, const ann::Matrix& my);
    };
}
