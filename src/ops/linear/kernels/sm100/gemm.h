#pragma once

// Dense tcgen05 GEMM for B200 (sm_100a): y[token, row] = epilogue(sum_k W[row, k] * x[token, k]).
//
// The weight is the MMA's M operand and the tokens are its N operand (swap-AB). The weight
// operands are consumed exactly as stored in the artifact: NVFP4 codes with
// block_scale_k16_m128x4 scales (one 512-byte atom per 128 rows and 64 K values), plain BF16, or
// E4M3 codes with FP32 row scales. Nothing is repacked on the execution path.

#include "ops/linear/kernels/nvfp4/nvfp4_operands.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace ninfer::ops::detail {

// Output transform applied to the FP32 accumulator before the single BF16 rounding.
//   Identity  y = acc * alpha                               y: [tokens, rows]
//   Add       y = acc * alpha + residual                    residual: [tokens, rows]; may alias y
//   Split     rows are cut into `segments`; segment i is written to outputs[i] as
//             [tokens, segments[i]]. y is unused. Segment widths are multiples of 32.
//   SwiGLU    weight rows are [gate; up] (rows / 2 each); y = silu(gate) * up   y: [tokens, rows / 2]
//   Scores    FP32 scores[token * score_stride + row] = acc; y is unused. BF16 only: the weight
//             rows may end inside a 128-row tile (the missing rows read as zero and are not
//             stored)
struct Sm100Epilogue {
    enum Kind { Identity, Add, Split, SwiGLU, Scores } kind = Identity;
    const __nv_bfloat16* residual = nullptr;
    std::span<const int> segments{};
    std::span<__nv_bfloat16* const> outputs{};
    float* scores   = nullptr;
    int score_stride = 0;
};

struct Sm100Bf16Operands {
    const __nv_bfloat16* x;      // [tokens, k] row-major
    const __nv_bfloat16* weight; // [rows, k] row-major
    int rows, k, tokens;
};

// E4M3 codes with FP32 scales: value = code * scale. Weight scales are per row, activation scales
// per token; the accumulator takes both before the epilogue transform.
struct Sm100Fp8Operands {
    const std::uint8_t* x;       // [tokens, k] row-major codes
    const float* x_scales;       // [tokens]
    const std::uint8_t* weight;  // [rows, k] row-major codes
    const float* weight_scales;  // [rows]
    int rows, k, tokens;
};

// Tile choice for one call. `tile_tokens` is the MMA's N. `cluster` is the CTA-pair width used by
// the weight multicast (1 = independent CTAs) and `ctas_per_sm` the persistent grid multiplier.
struct Nvfp4Sm100Tuning {
    int tile_tokens;
    int cluster;
    int ctas_per_sm;
};

// `weight_parts` is 2 for the SwiGLU epilogue, whose tile multiplies a gate and an up weight tile.
Nvfp4Sm100Tuning bf16_sm100_tuning(int rows, int k, int tokens, int weight_parts = 1);

// FP32 scratch that lets an Identity, Add or Split call of at most kSm100MaxSplitTokens tokens cut
// the K loop of the tiles a last partial round of the persistent grid would run on few SMs. Each
// cut tile runs as several K ranges side by side; a second kernel sums their FP32 partials in
// range order and applies the epilogue, so the result does not depend on scheduling. Without
// scratch every tile runs its whole K loop. kSm100SplitPartialBytes holds any split.
struct Sm100Partials {
    float* data       = nullptr; // 16-byte aligned
    std::size_t bytes = 0;
};
inline constexpr int kSm100MaxSplitTokens = 128;
inline constexpr int kSm100MaxSplitUnits  = 160;
inline constexpr int kSm100MaxKSplits     = 16;
inline constexpr std::size_t kSm100SplitPartialBytes =
    std::size_t{kSm100MaxSplitUnits} * kSm100MaxSplitTokens * 128 * sizeof(float);

// How one call runs: its token tile (the MMA's N) and, when it splits, its last `split_tiles`
// tiles cut into `k_splits` K ranges each. The plan functions take the scratch bytes the call
// passes; without scratch the plan never splits.
struct Sm100GemmPlan {
    int tile_tokens;
    int split_tiles = 0;
    int k_splits    = 1;
};

Sm100GemmPlan nvfp4_sm100_plan(int rows, int k, int tokens, std::size_t partial_bytes = 0);
Sm100GemmPlan fp8_sm100_plan(int rows, int k, int tokens, std::size_t partial_bytes = 0);

// Operands follow Nvfp4A4Operands: x is the quantized activation (codes + K16M128x4 scales padded
// to whole 128-token tiles), alpha = 1 / (input_scale * weight_scale).
void launch_sm100_nvfp4_gemm(const Nvfp4A4Operands& operands, const Sm100Epilogue& epilogue,
                             __nv_bfloat16* y, cudaStream_t stream, Sm100Partials partials = {});

void launch_sm100_bf16_gemm(const Sm100Bf16Operands& operands, const Sm100Epilogue& epilogue,
                            __nv_bfloat16* y, cudaStream_t stream);

// SwiGLU is not offered for FP8: the weight rows would need per-part scales.
void launch_sm100_fp8_gemm(const Sm100Fp8Operands& operands, const Sm100Epilogue& epilogue,
                           __nv_bfloat16* y, cudaStream_t stream, Sm100Partials partials = {});

} // namespace ninfer::ops::detail
