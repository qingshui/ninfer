#pragma once

// Epilogues and host-side launch plumbing shared by the NVFP4 and BF16 SM100 GEMMs.

#include "core/device.h"
#include "ops/common/cuda/sm100/gemm_mainloop.cuh"
#include "ops/linear/kernels/sm100/gemm.h"

#include <cuda_bf16.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::sm100_gemm {

namespace sm = ::ninfer::ops::common::sm100;

inline constexpr int kMaxSegments = 4;

// Identity, Add and Split share one epilogue: the accumulator of weight row r, token t is
// scaled, optionally added to the residual and written to the segment that owns row r.
struct PlainEpilogue {
    float alpha;
    bool add;
    const __nv_bfloat16* residual;
    __nv_bfloat16* outputs[kMaxSegments];
    int segment_begin[kMaxSegments + 1];
    int segment_count;
    int rows;
    int tokens;

    template <class Cfg>
    __device__ __forceinline__ void drain(const sm::EpilogueTile& t) const {
        constexpr int C = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
        const int row0  = t.m_tile * Cfg::kTileM + t.warp_q * 32;
        int segment     = 0;
        while (segment + 1 < segment_count && row0 >= segment_begin[segment + 1]) ++segment;
        __nv_bfloat16* out = outputs[segment];
        const int width    = segment_begin[segment + 1] - segment_begin[segment];
        const int column0  = row0 - segment_begin[segment];
        const int token0   = t.n_tile * Cfg::kTileN;
        auto* stage        = reinterpret_cast<__nv_bfloat16*>(t.scratch);
        for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
            std::uint32_t r[C];
            sm::tmem_load_32x32b<C>(t.tmem + c0, r);
            sm::tmem_load_wait();
#pragma unroll
            for (int i = 0; i < C; ++i) {
                float v = __uint_as_float(r[i]) * alpha;
                if (add) {
                    const int token = token0 + c0 + i;
                    if (token < tokens)
                        v += __bfloat162float(
                            residual[static_cast<std::int64_t>(token) * rows + row0 + t.lane]);
                }
                stage[i * 32 + t.lane] = __float2bfloat16_rn(v);
            }
            __syncwarp();
#pragma unroll
            for (int it = 0; it < C / 8; ++it) {
                const int local = it * 8 + (t.lane >> 2);
                const int token = token0 + c0 + local;
                if (token < tokens) {
                    const uint4 value =
                        *reinterpret_cast<const uint4*>(stage + local * 32 + (t.lane & 3) * 8);
                    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(token) * width +
                                              column0 + (t.lane & 3) * 8) = value;
                }
            }
            __syncwarp();
        }
    }
};

// Gate and up weight tiles accumulate side by side (accumulator part 0 / part 1), so one thread
// owns both operands of an output element: y = silu(gate * alpha) * (up * alpha).
struct SwiGluEpilogue {
    float alpha;
    __nv_bfloat16* out;
    int out_rows; // rows / 2
    int tokens;

    template <class Cfg>
    __device__ __forceinline__ void drain(const sm::EpilogueTile& t) const {
        static_assert(Cfg::kParts == 2);
        constexpr int C = Cfg::kTileN >= 32 ? 32 : Cfg::kTileN;
        const int row0  = t.m_tile * Cfg::kTileM + t.warp_q * 32;
        const int token0 = t.n_tile * Cfg::kTileN;
        auto* stage     = reinterpret_cast<__nv_bfloat16*>(t.scratch);
        for (int c0 = 0; c0 < t.valid_tokens; c0 += C) {
            std::uint32_t gate[C];
            std::uint32_t up[C];
            sm::tmem_load_32x32b<C>(t.tmem + c0, gate);
            sm::tmem_load_32x32b<C>(t.tmem + t.part_stride + c0, up);
            sm::tmem_load_wait();
#pragma unroll
            for (int i = 0; i < C; ++i) {
                const float g = __uint_as_float(gate[i]) * alpha;
                const float u = __uint_as_float(up[i]) * alpha;
                stage[i * 32 + t.lane] = __float2bfloat16_rn(g / (1.0f + __expf(-g)) * u);
            }
            __syncwarp();
#pragma unroll
            for (int it = 0; it < C / 8; ++it) {
                const int local = it * 8 + (t.lane >> 2);
                const int token = token0 + c0 + local;
                if (token < tokens) {
                    const uint4 value =
                        *reinterpret_cast<const uint4*>(stage + local * 32 + (t.lane & 3) * 8);
                    *reinterpret_cast<uint4*>(out + static_cast<std::int64_t>(token) * out_rows +
                                              row0 + (t.lane & 3) * 8) = value;
                }
            }
            __syncwarp();
        }
    }
};

inline int device_sm_count() {
    static std::array<int, 64> counts{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(counts.size()))
        throw std::runtime_error("sm100 gemm: device ordinal out of range");
    if (counts[device] == 0)
        CUDA_CHECK(cudaDeviceGetAttribute(&counts[device], cudaDevAttrMultiProcessorCount, device));
    return counts[device];
}

template <class Cfg, class Epilogue>
void launch_kernel(const CUtensorMap& a_map, const CUtensorMap& b_map, const sm::GemmArgs& args,
                   const Epilogue& epilogue, cudaStream_t stream) {
    static std::array<bool, 64> configured{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (!configured[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(sm::sm100_gemm_kernel<Cfg, Epilogue>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg::kSharedBytes));
        configured[device] = true;
    }
    const int tiles = args.tiles_m * args.tiles_n;
    const int grid  = sm::persistent_grid(tiles, device_sm_count());
    sm::sm100_gemm_kernel<Cfg, Epilogue>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(a_map, b_map, args, epilogue);
    CUDA_CHECK(cudaGetLastError());
}

[[noreturn]] inline void reject(const char* what) {
    throw std::invalid_argument(std::string("sm100 gemm: ") + what);
}

inline bool aligned(const void* p, std::uintptr_t a) {
    return p != nullptr && reinterpret_cast<std::uintptr_t>(p) % a == 0;
}

// Checks the epilogue against the problem and flattens it into kernel parameters.
inline PlainEpilogue make_plain_epilogue(const Sm100Epilogue& e, __nv_bfloat16* y, float alpha,
                                         int rows, int tokens) {
    PlainEpilogue p{};
    p.alpha   = alpha;
    p.add     = e.kind == Sm100Epilogue::Add;
    p.residual = e.residual;
    p.rows    = rows;
    p.tokens  = tokens;
    if (p.add && !aligned(e.residual, 2)) reject("Add epilogue requires a residual");
    if (e.kind == Sm100Epilogue::Split) {
        const int n = static_cast<int>(e.segments.size());
        if (n < 1 || n > kMaxSegments || e.outputs.size() != e.segments.size())
            reject("Split needs 1..4 segments with one output each");
        int begin = 0;
        for (int i = 0; i < n; ++i) {
            if (e.segments[i] <= 0 || e.segments[i] % 32 != 0)
                reject("segment widths must be positive multiples of 32");
            if (!aligned(e.outputs[i], 16)) reject("segment outputs must be 16-byte aligned");
            p.outputs[i]       = e.outputs[i];
            p.segment_begin[i] = begin;
            begin += e.segments[i];
        }
        if (begin != rows) reject("segments must sum to the weight rows");
        p.segment_begin[n] = begin;
        p.segment_count    = n;
    } else {
        if (!aligned(y, 16)) reject("output must be 16-byte aligned");
        p.outputs[0]       = y;
        p.segment_begin[0] = 0;
        p.segment_begin[1] = rows;
        p.segment_count    = 1;
    }
    return p;
}

} // namespace ninfer::ops::detail::sm100_gemm
