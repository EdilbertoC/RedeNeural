#include "Matrix/TensorComputeCore.h"
#include <stdexcept>

ann::Matrix ann::TensorComputeCore::multiply_matrix(const ann::Matrix& mx, const ann::Matrix& my)
{
    ann::Matrix result(mx.get_rows_count(), my.get_cols_count(), ProcessingType::Device);
    return result;
}
