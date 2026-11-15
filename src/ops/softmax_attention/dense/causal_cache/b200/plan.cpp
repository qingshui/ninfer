#include "ops/softmax_attention/dense/causal_cache/b200/plan.h"

#include "ninfer/types.h"
#include "ops/kv_cache/kernels/write.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/kernels/causal/operands.h"
#include "ops/softmax_attention/kernels/causal/partition.h"

#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kQueryHeads = 32;
constexpr int kKvHeads    = 2;
constexpr int kHeadDim    = 256;
constexpr int kBatchLimit = static_cast<int>(ninfer::kMaximumConcurrency);

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

kv::PagedWriteInstance select_write(KvDType kv, int heads, int width, int batch) {
    switch (kv) {
    case KvDType::Bf16:
        return batch > 1 ? kv::PagedWriteInstance::BatchRows256 : select_bf16_kv_write(heads);
    case KvDType::Int8:
        return select_int8_kv_write(heads, width, batch);
    case KvDType::Fp8:
        return select_fp8_kv_write(heads, width, batch);
    case KvDType::K8V4:
        return select_k8v4_kv_write(heads, width, batch);
    case KvDType::Nvfp4:
        return select_nvfp4_kv_write(heads, width, batch);
    }
    throw std::invalid_argument("sm100 attention: unknown KV dtype");
}

Sm100KvCache make_cache(const ConstPagedKVBatchLayerView& cache, const Tensor* valid,
                        const Tensor* rows) {
    const auto& offsets = cache.k_pages.records.page_offsets;
    return {cache.k_pages.data(),
            cache.v_pages.data(),
            cache.k_scale_pages.data(),
            cache.v_scale_pages.data(),
            static_cast<const std::int64_t*>(offsets.data),
            valid ? static_cast<const std::int32_t*>(valid->data) : nullptr,
            rows ? static_cast<const std::int32_t*>(rows->data) : nullptr,
            offsets.ne[0],
            cache.num_kv_heads};
}

void attend(KvDType kv, const Tensor& q, const Tensor& positions, float scale,
            const Sm100KvCache& cache, CausalAttentionExecutionEnvelope envelope,
            MemoryArena& workspace, Tensor& out, CudaExecutionView execution) {
    const auto plan = select_sm100_attention_plan(q.ne[1], q.ne[2], q.ne[3], envelope,
                                                  execution.multiprocessor_count());
    auto scope = workspace.scope();
    CausalPartialView partial{};
    if (plan.splits > 1)
        partial = allocate_causal_partials(workspace, q.ne[1], q.ne[2], plan.splits, q.ne[3]).view();
    const Sm100CausalOperands operands{
        make_causal_operands(q, positions, out, scale, envelope.max_visible_keys), cache, partial};
    launch_sm100_causal_attention(operands, kv, plan, execution.stream());
}

} // namespace

Sm100AttentionPlan select_sm100_attention_plan(int heads, int width, int batch,
                                               CausalAttentionExecutionEnvelope envelope,
                                               int multiprocessor_count) {
    if (multiprocessor_count <= 0 || heads != kQueryHeads || width < 1 || batch < 1 ||
        batch > kBatchLimit || (batch > 1 && width > 16) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys)
        throw std::invalid_argument("sm100 attention: invalid plan inputs");
    const int row_tiles   = ceil_div(width * (kQueryHeads / kKvHeads), kSm100TileQ);
    const int independent = batch * kKvHeads * row_tiles;
    const int key_tiles   = ceil_div(static_cast<int>(envelope.max_visible_keys), kSm100TileKv);
    // Single split versus split-plus-merge fixes the kernel topology of a captured graph, so it
    // depends only on the call shape. The split count may follow the envelope.
    if (independent >= multiprocessor_count)
        return {kQueryHeads, kKvHeads, kHeadDim, kSm100TileQ, kSm100TileKv, 1};
    const int target = causal_partition_target(multiprocessor_count, independent);
    const int splits = std::max(2, std::min(target, key_tiles / 2));
    return {kQueryHeads, kKvHeads, kHeadDim, kSm100TileQ, kSm100TileKv, splits};
}

std::size_t sm100_attention_workspace_bytes(int heads, int batch, int min_width, int max_width,
                                            CausalAttentionExecutionEnvelope envelope,
                                            int multiprocessor_count) {
    std::size_t maximum = 0;
    for (int width = min_width; width <= max_width; ++width) {
        const auto plan =
            select_sm100_attention_plan(heads, width, batch, envelope, multiprocessor_count);
        if (plan.splits == 1) continue;
        WorkspaceLayoutBuilder layout;
        (void)allocate_causal_partials(layout, heads, width, plan.splits, batch);
        maximum = std::max(maximum, layout.peak_bytes(1));
    }
    return maximum;
}

void sm100_kv_append_attention(KvDType kv, const Tensor& q, const Tensor& k, const Tensor& v,
                               const Tensor& positions, const Tensor& valid_columns,
                               const Tensor& table_rows, float scale, PagedKVBatchLayerView cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution) {
    kv::launch_paged_write(select_write(kv, k.ne[1], k.ne[2], k.ne[3]), k, v, positions,
                           valid_columns, table_rows, cache, execution.stream());
    attend(kv, q, positions, scale, make_cache(cache, &valid_columns, &table_rows), envelope,
           workspace, out, execution);
}

void sm100_kv_cached_attention(KvDType kv, const Tensor& q, const Tensor& positions, float scale,
                               const ConstPagedKVLayerView& cache,
                               CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                               Tensor& out, CudaExecutionView execution) {
    attend(kv, q, positions, scale, make_cache(single_row_paged_kv_batch_view(cache), nullptr, nullptr),
           envelope, workspace, out, execution);
}

} // namespace ninfer::ops::detail
