#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry  = Fp8Geometry<131072, 2048>;
using ScaleType = float;
using Gemv      = Fp8A16GemvSchedule<8, 2, 8, 4, Fp8CodeCache::Default, 2, 2>;

template <class Schedule>
void sliced(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    fp8_linear_a16_sliced_k<Geometry, Schedule, ScaleType>(x, w, out, stream);
}

template <class Schedule>
void mma(const Tensor& x, const Weight& w, Tensor& out, cudaStream_t stream) {
    fp8_linear_a16_mma<Geometry, Schedule, ScaleType>(x, w, out, stream);
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& out, cudaStream_t stream) {
    const int tokens = x.ne[1];
    if (tokens == 1) return fp8_linear_a16_gemv<Geometry, Gemv, ScaleType>(x, weight, out, stream);
    if (tokens <= 16) return sliced<Fp8SlicedInstance<16, 4, 1>>(x, weight, out, stream);
    if (tokens <= 32) return sliced<Fp8SlicedInstance<32, 4, 1>>(x, weight, out, stream);
    if (tokens <= 64)
        return mma<Fp8A16MmaSchedule<32, 64, 128, 16, 16, 1, 3>>(x, weight, out, stream);
    mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

// The vocabulary head stays A16, like the other registered heads.
bool uses_a8(std::int32_t, std::int32_t) { return false; }
} // namespace

const Fp8LinearShape kFp8RowFp32N131072K2048{131072, 2048, launch_a16, nullptr, uses_a8};
} // namespace ninfer::ops::detail
