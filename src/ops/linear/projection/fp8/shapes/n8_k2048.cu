#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<8, 2048>;
using ScaleType = float;
// Eight output rows fit one warp: GEMV for T=1, token-tiled CUDA-core contraction otherwise.
using Gemv = Fp8A16GemvSchedule<1, 8, 8, 4, Fp8CodeCache::Default, 2, 16>;
using Simt =
    Fp8A16SimtSchedule<1, 8, 8, 4, 4, Fp8SimtActivationAccess::TokenPacked, Fp8CodeCache::Default,
                       1, Fp8SimtBlockOrder::TokenTilesContiguous, 8>;

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    if (x.ne[1] == 1) return fp8_linear_a16_gemv<Geometry, Gemv, ScaleType>(x, weight, out, stream);
    fp8_linear_a16_simt<Geometry, 0, Simt, false, ScaleType>(x, weight, out, stream);
}

// Eight rows cannot form an A8 MMA row tile; every policy executes the A16 route.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8RowFp32N8K2048{8, 2048, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
