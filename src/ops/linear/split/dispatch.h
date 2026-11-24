#pragma once

#include "ops/linear/kernels/fp8/fp8_a8_plan.h"
#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/split/operands.h"

#include <array>

namespace ninfer::ops::detail::linear_split {
using SplitLaunch   = void (*)(const Tensor&, Weights, std::span<const Tensor>, MemoryArena*,
                             cudaStream_t);
using SplitCapacity = std::size_t (*)(LinearPolicy, std::int32_t, std::int32_t);

struct SplitShape {
    int first_rows, input_rows;
    std::span<const int> segments;
    SplitLaunch launch;
    SplitCapacity capacity = nullptr;
    int second_rows        = 0;
};

using SplitBiasLaunch = void (*)(const Tensor&, const Weight&, const Tensor&,
                                 std::span<const Tensor>, LinearPolicy, MemoryArena*, cudaStream_t);

struct SplitBiasShape {
    int rows, input_rows;
    std::span<const int> segments;
    SplitBiasLaunch launch;
    SplitCapacity capacity                                = nullptr;
    bool (*uses_tensor_cores)(LinearPolicy, std::int32_t) = nullptr;
};

template <int... Rows>
inline constexpr std::array kSplitRows{Rows...};

extern const SplitShape kBf16SplitN20480K1536;
extern const SplitShape kBf16SplitN1152K2560, kBf16SplitN7296K2560;
extern const SplitShape kBf16SplitN1152K2560ThreeOutput, kBf16SplitN640K2560;
extern const SplitShape kBf16SplitN288K4096, kBf16SplitN1664K2560, kBf16SplitN13952K2560,
    kBf16SplitN14336K5120, kBf16SplitN20480K3072;
extern const SplitShape kQ8SplitN12288K2560, kQ8SplitN16384K2560, kQ8SplitN9216K2048;
extern const SplitShape kQ8SplitN6144K2048, kQ8SplitN6144K5120, kQ8SplitN12288K2048;
extern const SplitShape kQ8SplitN8192K2560, kQ8SplitN12288K4096;
extern const SplitShape kQ8SplitN6144K2560;
extern const SplitShape kQ4Q5SplitN7168N7168K5120, kQ4Q5SplitN4096N12288K5120;
extern const SplitShape kFp8SplitN14336K5120, kFp8SplitN16384K5120;
extern const SplitShape kBf16SplitN3072K2048, kFp8RowFp32SplitN3072K2048;
extern const SplitShape kFp8RowFp32SplitN20480K3072; // B200 only
extern const SplitShape kNvfp4SplitN14336K5120, kNvfp4SplitN16384K5120, kNvfp4SplitN17408K3072;

extern const SplitShape kMxfp8SplitN20480K1536;
extern const SplitShape kFp8BlockBf16SplitN16384N4096K1536;
extern const SplitBiasShape kBf16SplitBiasN3072K1024, kMxfp8SplitBiasN3072K1024;

const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const std::int32_t> output_rows);
const SplitBiasShape& select_split_bias_shape(QType type, int n, int k, LinearPolicy policy,
                                              std::span<const Tensor> outputs);

const SplitShape& select_split_shape(ProjectionProfile weights,
                                     std::span<const std::int32_t> output_rows);
const SplitShape& select_split_shape(ProjectionProfile weights, std::span<const Tensor> outputs);

// Single-parent A16 dispatch for the Qwen3.5-122B GDN input projection ({qkv, z} segmentation),
// reused by the GDN input-conv Op for its decode and snapshot paths. The caller owns operand
// validation; the route needs no workspace.
void bf16_split_n20480_k3072_dispatch(const Tensor& x, const Weight& weight,
                                      std::span<const Tensor> outputs, cudaStream_t stream);

// B200 only. The same geometry with E4M3 row-scaled weights: quantizes x into `workspace` and runs
// the tcgen05 GEMM. The caller owns operand validation and the scratch; workspace partials of
// kFp8SplitN20480K3072PartialBytes let a decode call split its K loop across idle SMs.
inline constexpr std::size_t kFp8SplitN20480K3072PartialBytes = kSm100SplitPartialBytes;
void fp8_split_n20480_k3072_dispatch(const Tensor& x, const Weight& weight,
                                     std::span<const Tensor> outputs, Fp8A8Workspace workspace,
                                     cudaStream_t stream);
} // namespace ninfer::ops::detail::linear_split
