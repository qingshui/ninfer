#include <algorithm>

#include "ops/linear/projection/fp8/fp8_shapes.h"
#include "ops/linear/projection/fp8/fp8_launch.cuh"

namespace ninfer::ops::detail {
namespace {
using Geometry   = Fp8Geometry<2048, 6656>;
using ScaleType  = float;
using Gemv       = Fp8A16GemvSchedule<4, 2, 8, 4, Fp8CodeCache::Default, 2, 4>;
using Tma64x128  = Fp8A8TmaMmaSchedule<64, 128, 128, 2, 4, 2, 1>;
using Tma128x128 = Fp8A8TmaMmaSchedule<128, 128, 128, 4, 4, 2, 1, Fp8MmaRaster::RowFast>;
// 64 row tiles split in K across 128 CTAs, so every SM streams weights with deep TMA stages.
using DecodeSplit = Fp8A8TmaSplitKSchedule<Fp8A8TmaMmaSchedule<16, 32, 128, 1, 2, 6, 2>, 170, 2, 4>;
// Full waves write directly; an underfilled final wave (<= 85 tiles) is split in K.
using PrefillSplit = Fp8A8TmaSplitKSchedule<Tma128x128, 170, 16, 16>;

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
    if (tokens <= 128)
        return mma<Fp8A16MmaSchedule<64, 64, 128, 32, 16, 2, 2>>(x, weight, out, stream);
    mma<Fp8A16MmaSchedule<64, 128, 64, 64, 16, 2, 2>>(x, weight, out, stream);
}

void launch_a8(const Tensor& x, const Weight& weight, Tensor& out, Fp8A8Workspace scratch,
               cudaStream_t stream) {
    if (x.ne[1] <= 16)
        return launch_fp8_a8_tma<Geometry, DecodeSplit, ScaleType>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 32)
        return launch_fp8_a8<Geometry, Fp8A8T32R32K128, ScaleType>(x, weight, out, scratch, stream);
    if (x.ne[1] <= 96)
        return launch_fp8_a8<Geometry, Fp8A8T32R128K128, ScaleType>(x, weight, out, scratch,
                                                                    stream);
    if (x.ne[1] <= 128)
        return launch_fp8_a8_tma<Geometry, Tma64x128, ScaleType>(x, weight, out, scratch, stream);
    launch_fp8_a8_tma<Geometry, PrefillSplit, ScaleType>(x, weight, out, scratch, stream);
}

bool uses_a8(std::int32_t, std::int32_t max_tokens) { return max_tokens >= 5; }

std::size_t partial_capacity_bytes(std::int32_t max_tokens) {
    if (max_tokens > 128) return std::max(PrefillSplit::kPartialBytes, DecodeSplit::kPartialBytes);
    return max_tokens >= 5 ? DecodeSplit::kPartialBytes : 0;
}
} // namespace

const Fp8LinearShape kFp8RowFp32N2048K6656{2048,      6656,    launch_a16,
                                           launch_a8, uses_a8, partial_capacity_bytes};
} // namespace ninfer::ops::detail
