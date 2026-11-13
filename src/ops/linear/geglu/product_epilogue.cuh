#pragma once
#include "ops/common/numeric/math.cuh"
#include "ops/linear/swiglu/product_epilogue.cuh"
#include "ops/linear/swiglu/row_major_mma_epilogue.cuh"

namespace ninfer::ops::detail {
// Exact erf GELU of the gate accumulator times the up accumulator, both in FP32.
struct GeGluProduct {
    __device__ __forceinline__ float operator()(float gate, float up) const {
        return common::gelu<false>(gate) * up;
    }
};

using GeGluPairEpilogue          = BasicSwiGluPairEpilogue<GeGluProduct>;
using GeGluRowMajorMmaEpilogue   = BasicSwiGluRowMajorMmaEpilogue<GeGluProduct>;
using GeGluTokenMajorMmaEpilogue = BasicSwiGluTokenMajorMmaEpilogue<GeGluProduct>;
} // namespace ninfer::ops::detail
