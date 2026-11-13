#include "ops/linear/geglu/dispatch.h"
#include "ops/linear/geglu/product_epilogue.cuh"
#include "ops/linear/swiglu/bf16/paired.cuh"

namespace ninfer::ops::detail {
namespace {
constexpr int kK            = 2048;
constexpr int kIntermediate = 6656;

template <class S>
using Static = Bf16ScheduleInstance<S, kK>;

template <class Schedule>
void tma(const Bf16A16Operands& p, LinearBf16Output out, cudaStream_t stream) {
    launch_swiglu_bf16_tma<Static<Schedule>, kIntermediate>(p, out, GeGluProduct{}, stream);
}

template <class Sliced>
void sliced(const Bf16A16Operands& p, LinearBf16Output out, cudaStream_t stream) {
    using S = SwiGluBf16SlicedSchedule<Static<Sliced>, kIntermediate>;
    launch_bf16_a16_sliced_k_mma<S>(p, out, SwiGluBf16FragmentEpilogue<GeGluProduct>{}, stream);
}
} // namespace

void launch_bf16_geglu_n13312_k2048(const Tensor& x, const Weight& weight, Tensor& y,
                                    cudaStream_t stream) {
    const auto p = bf16_a16_operands(x, weight);
    const LinearBf16Output out{static_cast<__nv_bfloat16*>(y.data), kIntermediate};
    if (p.tokens == 1) {
        swiglu_bf16_paired_gemv<kK, kIntermediate>
            <<<kIntermediate / 4, 128, 0, stream>>>(p.x, p.weight, out, GeGluProduct{});
        CUDA_CHECK(cudaGetLastError());
    } else if (p.tokens <= 8) {
        sliced<Bf16A16SlicedKMmaSchedule<16, 8, 8, 64, 2>>(p, out, stream);
    } else if (p.tokens <= 16) {
        sliced<Bf16A16SlicedKMmaSchedule<16, 16, 4, 64, 2>>(p, out, stream);
    } else if (p.tokens <= 32) {
        tma<Bf16A16TmaMmaSchedule<32, 16, 128, 16, 8, 2, 2>>(p, out, stream);
    } else if (p.tokens <= 64) {
        tma<Bf16A16TmaMmaSchedule<32, 32, 128, 16, 16, 2, 2>>(p, out, stream);
    } else if (p.tokens <= 128) {
        tma<Bf16A16TmaMmaSchedule<64, 32, 64, 32, 16, 2, 2>>(p, out, stream);
    } else if (p.tokens <= 512) {
        tma<Bf16A16TmaMmaSchedule<128, 64, 64, 64, 16, 2, 1>>(p, out, stream);
    } else if (p.tokens <= 2048) {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1>>(p, out, stream);
    } else {
        tma<Bf16A16TmaMmaSchedule<128, 128, 64, 64, 32, 2, 1, Bf16MmaRaster::RowFast>>(p, out,
                                                                                       stream);
    }
}
} // namespace ninfer::ops::detail
