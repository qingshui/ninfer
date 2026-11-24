#include "ops/softmax_attention/kernels/causal/sm100/launch.h"

#include "core/device.h"
#include "ops/softmax_attention/kernels/causal/merge.cuh"
#include "ops/softmax_attention/kernels/causal/sm100/fmha.cuh"

#include <array>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail {
namespace {

namespace fm = sm100_fmha;

// The inverse rotation transforms whole D256 rows, so those merges take every feature in one CTA.
template <int DChunk>
struct MergeSchedule {
    static constexpr int kDChunk  = DChunk;
    static constexpr int kThreads = 256;
};

constexpr int kKeyShift = 6; // log2(kSm100TileKv): every active split owns whole key tiles

[[noreturn]] void reject(const char* what) {
    throw std::invalid_argument(std::string("sm100 causal attention: ") + what);
}

template <KvDType Kv>
void launch_policy(const Sm100CausalOperands& p, const Sm100AttentionPlan& plan,
                   cudaStream_t stream) {
    using C = fm::Cfg<Kv>;
    static std::array<bool, 64> configured{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(configured.size()))
        reject("device ordinal out of range");
    if (!configured[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(fm::fmha_kernel<Kv>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        C::kSharedBytes));
        configured[device] = true;
    }

    const int row_tiles = (p.width * fm::kGroup + C::kRows - 1) / C::kRows;
    fm::FmhaParams params{};
    params.q                = p.q;
    params.positions        = p.positions;
    params.out              = p.out;
    params.partial          = p.partial;
    params.cache            = p.cache;
    params.scale            = p.scale;
    params.width            = p.width;
    params.logical_capacity = p.visible_capacity;
    params.splits           = plan.splits;
    params.row_tiles        = row_tiles;
    params.key_shift        = kKeyShift;
    const dim3 grid(static_cast<unsigned>(fm::Geometry::KVHeads * row_tiles),
                    static_cast<unsigned>(plan.splits), static_cast<unsigned>(p.batch));
    fm::fmha_kernel<Kv><<<grid, C::kThreads, C::kSharedBytes, stream>>>(params);
    CUDA_CHECK(cudaGetLastError());

    if (plan.splits > 1) {
        const CausalKvPartition partition{plan.splits, plan.splits, kKeyShift};
        constexpr bool kInverse = C::Policy::kInverseRotate;
        using Schedule          = MergeSchedule<kInverse ? 256 : 64>;
        if (p.cache.valid_columns != nullptr)
            launch_causal_natural_merge<fm::Geometry, Schedule, true, true, kInverse>(
                p, p.cache.valid_columns, partition, p.partial, stream);
        else
            launch_causal_natural_merge<fm::Geometry, Schedule, true, false, kInverse>(
                p, nullptr, partition, p.partial, stream);
    }
}

template <KvDType Kv>
void launch_group_policy(const Sm100GroupOperands& p, std::span<const Sm100AttentionGroup> groups,
                         cudaStream_t stream) {
    using C = fm::Cfg<Kv>;
    static std::array<bool, 64> configured{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    if (device < 0 || device >= static_cast<int>(configured.size()))
        reject("device ordinal out of range");
    if (!configured[device]) {
        CUDA_CHECK(cudaFuncSetAttribute(fm::fmha_group_kernel<Kv>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        C::kSharedBytes));
        configured[device] = true;
    }

    fm::GroupParams params{};
    params.base.q                = p.q;
    params.base.positions        = p.positions;
    params.base.out              = p.out;
    params.base.cache            = p.cache;
    params.base.scale            = p.scale;
    params.base.logical_capacity = p.visible_capacity;
    params.base.splits           = 1;
    params.base.key_shift        = kKeyShift;
    for (std::size_t next = 0; next < groups.size();) {
        std::int64_t ctas = 0;
        int count         = 0;
        for (; next < groups.size() && count < fm::kMaxGroups; ++next, ++count) {
            const auto& g       = groups[next];
            params.group[count] = {g.token_base, g.width, g.row, g.batch, static_cast<int>(ctas)};
            ctas += static_cast<std::int64_t>(g.batch) * fm::Geometry::KVHeads *
                    fm::row_tiles_of(g.width);
            if (ctas > std::numeric_limits<int>::max()) reject("group launch is too large");
        }
        params.groups = count;
        fm::fmha_group_kernel<Kv><<<dim3(static_cast<unsigned>(ctas)), C::kThreads, C::kSharedBytes,
                                    stream>>>(params);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

void launch_sm100_causal_attention(const Sm100CausalOperands& p, KvDType kv,
                                   const Sm100AttentionPlan& plan, cudaStream_t stream) {
    if (plan.q_heads != fm::Geometry::QHeads || plan.kv_heads != fm::Geometry::KVHeads ||
        plan.head_dim != fm::Geometry::kHeadDim || p.query_heads != plan.q_heads ||
        p.head_dim != plan.head_dim || p.cache.kv_heads != plan.kv_heads)
        reject("unsupported head geometry");
    if (plan.tile_q != kSm100TileQ || plan.tile_kv != kSm100TileKv)
        reject("unsupported tile shape");
    if (plan.splits < 1 || plan.splits > CausalKvPartition::kMaxSplits) reject("invalid split count");
    if (p.width < 1 || p.batch < 1 || p.visible_capacity < 1) reject("invalid width/batch");
    if (!p.q || !p.positions || !p.out || !p.cache.keys || !p.cache.values || !p.cache.tables)
        reject("missing operand");
    if (kv != KvDType::Bf16 && (!p.cache.key_scales || !p.cache.value_scales))
        reject("quantized KV needs scale planes");
    if (plan.splits > 1 && (!p.partial.acc || !p.partial.maximum || !p.partial.sum))
        reject("split plan needs partial storage");

    switch (kv) {
    case KvDType::Bf16:
        return launch_policy<KvDType::Bf16>(p, plan, stream);
    case KvDType::Int8:
        return launch_policy<KvDType::Int8>(p, plan, stream);
    case KvDType::Fp8:
        return launch_policy<KvDType::Fp8>(p, plan, stream);
    case KvDType::K8V4:
        return launch_policy<KvDType::K8V4>(p, plan, stream);
    case KvDType::Nvfp4:
        return launch_policy<KvDType::Nvfp4>(p, plan, stream);
    }
    reject("unknown KV dtype");
}

void launch_sm100_causal_attention_groups(const Sm100GroupOperands& p, KvDType kv,
                                          std::span<const Sm100AttentionGroup> groups,
                                          cudaStream_t stream) {
    if (p.query_heads != fm::Geometry::QHeads || p.head_dim != fm::Geometry::kHeadDim ||
        p.cache.kv_heads != fm::Geometry::KVHeads)
        reject("unsupported head geometry");
    if (groups.empty() || p.visible_capacity < 1) reject("invalid groups");
    if (!p.q || !p.positions || !p.out || !p.cache.keys || !p.cache.values || !p.cache.tables ||
        !p.cache.valid_columns || !p.cache.table_rows)
        reject("missing operand");
    if (kv != KvDType::Bf16 && (!p.cache.key_scales || !p.cache.value_scales))
        reject("quantized KV needs scale planes");
    for (const auto& g : groups) {
        if (g.token_base < 0 || g.width < 1 || g.batch < 1 || g.row < 0) reject("invalid group");
    }

    switch (kv) {
    case KvDType::Bf16:
        return launch_group_policy<KvDType::Bf16>(p, groups, stream);
    case KvDType::Int8:
        return launch_group_policy<KvDType::Int8>(p, groups, stream);
    case KvDType::Fp8:
        return launch_group_policy<KvDType::Fp8>(p, groups, stream);
    case KvDType::K8V4:
        return launch_group_policy<KvDType::K8V4>(p, groups, stream);
    case KvDType::Nvfp4:
        return launch_group_policy<KvDType::Nvfp4>(p, groups, stream);
    }
    reject("unknown KV dtype");
}

} // namespace ninfer::ops::detail
