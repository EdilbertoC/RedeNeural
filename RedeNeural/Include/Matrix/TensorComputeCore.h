#pragma once
#include "Matrix.h"
namespace ann
{
    class TensorComputeCore
    {
    public:
        ann::Matrix multiply_matrix(ann::Matrix mx, ann::Matrix my);

    };
}
