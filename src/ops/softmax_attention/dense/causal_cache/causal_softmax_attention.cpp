#include "ninfer/ops/softmax_attention.h"
#include "ninfer/types.h"

#include "core/layout.h"
#include "ops/common/host/operands.h"
#include "core/paged_kv_storage.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/plan.h"
#include "ops/softmax_attention/dense/causal_cache/bf16/launch.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/fp8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/int8/plan.h"
#include "ops/softmax_attention/dense/causal_cache/int8/launch.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/nvfp4/launch.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/plan.h"
#include "ops/softmax_attention/dense/causal_cache/k8v4/launch.h"
#include "ops/softmax_attention/backend.h"
#if NINFER_TARGET_B200
#include "ops/softmax_attention/dense/causal_cache/b200/plan.h"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim             = 256;
constexpr float kExpectedScale              = 0.0625f;
constexpr std::int32_t kMaximumVerifyTokens = 16;
constexpr std::int32_t kMaximumBatchSize    = 8;
// The Hq=8, Hkv=2 profile serves wider verification batches.
constexpr std::int32_t kMaximumWideBatchSize = 16;
// The Hq=32, Hkv=2 service profile spans the full engine concurrency ceiling: automatic lane
// fitting plans the workspace at kMaximumConcurrency lanes and the model planning loop probes
// every decode batch up to it, so the registered execution domain must cover the same range.
constexpr std::int32_t kMaximumServiceBatchSize =
    static_cast<std::int32_t>(ninfer::kMaximumConcurrency);

std::int32_t maximum_batch_size(AttentionHeadGeometry geometry) {
    if (geometry.query_heads == 32) return kMaximumServiceBatchSize;
    return geometry.query_heads == 8 ? kMaximumWideBatchSize : kMaximumBatchSize;
}

void require_causal_geometry(AttentionHeadGeometry geometry, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        !((geometry.query_heads == 32 && geometry.kv_heads == 2) ||
          (geometry.query_heads == 24 && geometry.kv_heads == 4) ||
          (geometry.query_heads == 16 && geometry.kv_heads == 2) ||
          (geometry.query_heads == 8 && geometry.kv_heads == 2))) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
}

std::uint32_t validate_batch_cache(const ConstPagedKVBatchLayerView& cache, std::int32_t kv_heads,
                                   const char* op, Device device) {
    common::operand::device(cache.device(), device, op, "cache");
    PagedKVStorageLayout layout{};
    try {
        layout = paged_kv_storage_layout(cache.storage, kHeadDim);
    } catch (const std::invalid_argument&) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }
    if (cache.num_kv_heads != kv_heads || cache.head_dim != kHeadDim) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache geometry or storage");
    }

    const auto& offsets              = cache.k_pages.records.page_offsets;
    const std::int32_t logical_pages = offsets.ne[0];
    const std::int32_t table_rows    = offsets.ne[1];
    const std::int64_t capacity      = static_cast<std::int64_t>(logical_pages) * kPagedKVPageSize;
    if (logical_pages <= 0 || table_rows <= 0 ||
        capacity > std::numeric_limits<std::int32_t>::max() ||
        cache.k_pages.records.page_tokens != kPagedKVPageSize) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache capacity");
    }
    common::operand::paged_plane(cache.k_pages, layout.key.data_dtype,
                                 {layout.key.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache K", device);
    common::operand::paged_plane(cache.v_pages, layout.value.data_dtype,
                                 {layout.value.data_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                 op, "cache V", device);
    common::operand::same_paged_records(cache.k_pages, cache.v_pages, op, "cache K/V");
    const auto validate_scale = [&](const ConstPagedPlaneView& plane,
                                    const PagedKVVectorLayout& vector, const char* name) {
        if (!vector.has_scale()) {
            if (plane.records.storage.data != nullptr) {
                throw std::invalid_argument(std::string(op) +
                                            ": unscaled KV cache must not have scales");
            }
            return;
        }
        common::operand::paged_plane(plane, vector.scale_dtype,
                                     {vector.scale_leading_extent, kPagedKVPageSize, kv_heads}, 16,
                                     op, name, device);
        common::operand::same_paged_records(cache.k_pages, plane, op, name);
    };
    validate_scale(cache.k_scale_pages, layout.key, "cache K scales");
    validate_scale(cache.v_scale_pages, layout.value, "cache V scales");
    return static_cast<std::uint32_t>(capacity);
}

void validate_envelope(CausalAttentionExecutionEnvelope envelope,
                       const ConstPagedKVLayerView& cache, std::int32_t tokens, const char* op,
                       Device device) {
    common::operand::shape(cache.k_pages.records.page_offsets,
                           {cache.k_pages.records.page_offsets.ne[0], 1, 1, 1}, op, "block table");
    common::operand::contiguous_nonnull(cache.k_pages.records.page_offsets, op, "block table",
                                        device);
    const std::uint32_t capacity =
        validate_batch_cache(single_row_paged_kv_batch_view(cache), cache.num_kv_heads, op, device);
    if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
    }
    if (envelope.max_visible_keys < static_cast<std::uint32_t>(tokens)) {
        throw std::invalid_argument(std::string(op) + ": execution envelope is shorter than T");
    }
}

void validate_attention_tensors(const Tensor& q, const Tensor& positions, const Tensor& out,
                                AttentionHeadGeometry geometry, const ConstPagedKVLayerView& cache,
                                CausalAttentionExecutionEnvelope envelope, float scale,
                                const char* op, Device device) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    if (positions.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    if (tokens <= 0) { throw std::invalid_argument(std::string(op) + ": T must be positive"); }
    common::operand::shape(q, {kHeadDim, q_heads, tokens, 1}, op, "q");
    common::operand::shape(positions, {tokens, 1, 1, 1}, op, "positions");
    common::operand::shape(out, {kHeadDim, q_heads, tokens, 1}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    validate_envelope(envelope, cache, tokens, op, device);
}

void validate_batched_attention_tensors(const Tensor& q, const Tensor& positions,
                                        const Tensor& valid_columns, const Tensor& kv_table_rows,
                                        const Tensor& out, const ConstPagedKVBatchLayerView& cache,
                                        AttentionHeadGeometry geometry,
                                        CausalAttentionExecutionEnvelope envelope, float scale,
                                        const char* op, Device device) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/out must be BF16");
    }
    const bool masked = valid_columns.data != nullptr;
    if (positions.dtype != DType::I32 || kv_table_rows.dtype != DType::I32 ||
        (masked && valid_columns.dtype != DType::I32)) {
        throw std::invalid_argument(std::string(op) + ": batch metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    if (width <= 0 || batch <= 0 || batch > maximum_batch_size(geometry) ||
        (batch > 1 && width > kMaximumVerifyTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
    }
    common::operand::shape(q, {kHeadDim, q_heads, width, batch}, op, "q");
    common::operand::shape(positions, {width, batch, 1, 1}, op, "positions");
    if (masked) { common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns"); }
    common::operand::shape(kv_table_rows, {batch, 1, 1, 1}, op, "KV table rows");
    common::operand::shape(out, {kHeadDim, q_heads, width, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    if (masked) { common::operand::contiguous_nonnull(valid_columns, op, "valid columns", device); }
    common::operand::contiguous_nonnull(kv_table_rows, op, "KV table rows", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    if (cache.num_kv_heads != kv_heads) {
        throw std::invalid_argument(std::string(op) + ": invalid KV cache head geometry");
    }
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op, device);
    if (cache.k_pages.records.page_offsets.ne[1] < batch || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
        envelope.max_visible_keys > capacity ||
        (!masked && envelope.max_visible_keys < static_cast<std::uint32_t>(width))) {
        throw std::invalid_argument(std::string(op) + ": invalid execution envelope or table");
    }
}

void validate_group_tensors(const Tensor& q, const Tensor& k, const Tensor& v,
                           const Tensor& positions, const Tensor& valid_columns,
                           const Tensor& kv_table_rows,
                           std::span<const CausalAttentionGroup> groups,
                           AttentionHeadGeometry geometry, float scale,
                           const ConstPagedKVBatchLayerView& cache, const Tensor& out,
                           const char* op, Device device) {
    require_causal_geometry(geometry, op);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32 || valid_columns.dtype != DType::I32 ||
        kv_table_rows.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": metadata must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    const std::int32_t tokens   = q.ne[2];
    const std::int32_t rows     = valid_columns.ne[0];
    if (tokens <= 0 || rows <= 0 || groups.empty()) {
        throw std::invalid_argument(std::string(op) + ": empty token axis, rows or groups");
    }
    common::operand::shape(q, {kHeadDim, q_heads, tokens, 1}, op, "q");
    common::operand::shape(k, {kHeadDim, kv_heads, tokens, 1}, op, "k");
    common::operand::shape(v, {kHeadDim, kv_heads, tokens, 1}, op, "v");
    common::operand::shape(out, {kHeadDim, q_heads, tokens, 1}, op, "out");
    common::operand::shape(positions, {tokens, 1, 1, 1}, op, "positions");
    common::operand::shape(valid_columns, {rows, 1, 1, 1}, op, "valid columns");
    common::operand::shape(kv_table_rows, {rows, 1, 1, 1}, op, "KV table rows");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(k, op, "k", device);
    common::operand::contiguous_nonnull(v, op, "v", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    common::operand::contiguous_nonnull(valid_columns, op, "valid columns", device);
    common::operand::contiguous_nonnull(kv_table_rows, op, "KV table rows", device);
    const std::uint32_t capacity = validate_batch_cache(cache, kv_heads, op, device);

    std::uint64_t next_column = 0;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> row_ranges;
    row_ranges.reserve(groups.size());
    for (const auto& g : groups) {
        const auto& envelope = g.envelope;
        if (g.width == 0 || g.batch == 0 || g.batch > static_cast<std::uint32_t>(maximum_batch_size(geometry)) ||
            (g.batch > 1 && g.width > static_cast<std::uint32_t>(kMaximumVerifyTokens))) {
            throw std::invalid_argument(std::string(op) + ": unsupported B/W domain");
        }
        const std::uint64_t columns = static_cast<std::uint64_t>(g.width) * g.batch;
        if (g.token_offset < next_column || g.token_offset + columns > static_cast<std::uint64_t>(tokens) ||
            static_cast<std::uint64_t>(g.row) + g.batch > static_cast<std::uint64_t>(rows) ||
            cache.k_pages.records.page_offsets.ne[1] < static_cast<std::int32_t>(g.batch)) {
            throw std::invalid_argument(std::string(op) + ": group is outside the token axis or rows");
        }
        if (envelope.min_visible_keys == 0 || envelope.min_visible_keys > envelope.max_visible_keys ||
            envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys ||
            envelope.max_visible_keys > capacity) {
            throw std::invalid_argument(std::string(op) + ": invalid execution envelope");
        }
        next_column = g.token_offset + columns;
        row_ranges.emplace_back(g.row, g.batch);
    }
    std::sort(row_ranges.begin(), row_ranges.end());
    for (std::size_t i = 1; i < row_ranges.size(); ++i) {
        if (row_ranges[i].first < row_ranges[i - 1].first + row_ranges[i - 1].second) {
            throw std::invalid_argument(std::string(op) + ": groups share metadata entries");
        }
    }
}

} // namespace

std::size_t causal_softmax_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, KvCacheStorage cache_storage,
    CausalAttentionExecutionEnvelope envelope, std::int32_t batch_size, std::int32_t min_width,
    std::int32_t max_width, std::int32_t multiprocessor_count) {
    require_causal_geometry(geometry, "causal_softmax_attention workspace");
    const std::int32_t q_heads = geometry.query_heads;
    bool supported_dtype       = true;
    try {
        (void)paged_kv_storage_layout(cache_storage, kHeadDim);
    } catch (const std::invalid_argument&) { supported_dtype = false; }
    if (multiprocessor_count <= 0 || !supported_dtype || batch_size <= 0 ||
        batch_size > maximum_batch_size(geometry) || min_width <= 0 || max_width < min_width ||
        (batch_size > 1 && max_width > kMaximumVerifyTokens) || envelope.min_visible_keys == 0 ||
        envelope.min_visible_keys > envelope.max_visible_keys ||
        envelope.max_visible_keys > kCausalAttentionMaximumVisibleKeys) {
        throw std::invalid_argument(
            "causal_softmax_attention workspace: invalid profile or interval");
    }

#if NINFER_TARGET_B200
    if (detail::attention_backend(detail::kv_dtype(cache_storage), q_heads, geometry.kv_heads,
                                  kHeadDim) == detail::AttentionBackend::Tcgen05)
        return detail::sm100_attention_workspace_bytes(q_heads, batch_size, min_width, max_width,
                                                       envelope, multiprocessor_count);
#endif
    if (cache_storage == KvCacheStorage::BFloat16)
        return detail::bf16_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               multiprocessor_count);

    if (cache_storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
#if NINFER_TARGET_B200
        // The fp8 FMHA kernels are an SM120 family; on B200 the fp8 cache is only served by
        // the tcgen05 path above.
        throw std::invalid_argument("causal_softmax_attention workspace: fp8 is not available on b200");
#else
        return detail::fp8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                              multiprocessor_count);
#endif
    }

    if (cache_storage == KvCacheStorage::Int8Group64)
        return detail::int8_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                               multiprocessor_count);

    if (cache_storage == KvCacheStorage::Nvfp4Group16)
        return detail::nvfp4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                                multiprocessor_count);

#if NINFER_TARGET_B200
    // The k8v4 FMHA kernels are an SM120 family; on B200 the k8v4 cache is only served by
    // the tcgen05 path above.
    throw std::invalid_argument("causal_softmax_attention workspace: k8v4 is not available on b200");
#else
    return detail::k8v4_kv_workspace_bytes(q_heads, batch_size, min_width, max_width, envelope,
                                           multiprocessor_count);
#endif
}

void causal_softmax_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                              const Tensor& positions, const Tensor& valid_columns,
                              const Tensor& kv_table_rows, AttentionHeadGeometry geometry,
                              float scale, PagedKVBatchLayerView cache,
                              CausalAttentionExecutionEnvelope envelope, MemoryArena& workspace,
                              Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "causal_softmax_attention: workspace belongs to another device");
    }
    constexpr const char* op = "causal_softmax_attention";
    validate_batched_attention_tensors(q, positions, valid_columns, kv_table_rows, out, cache,
                                       geometry, envelope, scale, op, execution.device());
    if (k.dtype != DType::BF16 || v.dtype != DType::BF16) {
        throw std::invalid_argument("causal_softmax_attention: k/v must be BF16");
    }
    const std::int32_t width    = q.ne[2];
    const std::int32_t batch    = q.ne[3];
    const std::int32_t kv_heads = geometry.kv_heads;
    common::operand::shape(k, {kHeadDim, kv_heads, width, batch}, op, "k");
    common::operand::shape(v, {kHeadDim, kv_heads, width, batch}, op, "v");
    common::operand::contiguous_nonnull(k, op, "k", execution.device());
    common::operand::contiguous_nonnull(v, op, "v", execution.device());

#if NINFER_TARGET_B200
    if (detail::attention_backend(detail::kv_dtype(cache.storage), geometry.query_heads,
                                  geometry.kv_heads, kHeadDim) ==
        detail::AttentionBackend::Tcgen05) {
        detail::sm100_kv_append_attention(detail::kv_dtype(cache.storage), q, k, v, positions,
                                          valid_columns, kv_table_rows, scale, cache, envelope,
                                          workspace, out, execution);
        return;
    }
#endif
    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, envelope, workspace, out, execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
#if NINFER_TARGET_B200
        // The fp8 FMHA kernels are an SM120 family; on B200 the fp8 cache is only served by
        // the tcgen05 path above.
        throw std::invalid_argument("causal_softmax_attention: fp8 is not available on b200");
#else
        detail::fp8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                        cache, envelope, workspace, out, execution);
        return;
#endif
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                         cache, envelope, workspace, out, execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale,
                                          cache, envelope, workspace, out, execution);
        return;
    }

#if NINFER_TARGET_B200
    // The k8v4 FMHA kernels are an SM120 family; on B200 the k8v4 cache is only served by
    // the tcgen05 path above.
    throw std::invalid_argument("causal_softmax_attention: k8v4 is not available on b200");
#else
    detail::k8v4_kv_append_attention(q, k, v, positions, valid_columns, kv_table_rows, scale, cache,
                                     envelope, workspace, out, execution);
#endif
}

void causal_softmax_attention_groups(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     const Tensor& kv_table_rows,
                                     std::span<const CausalAttentionGroup> groups,
                                     AttentionHeadGeometry geometry, float scale,
                                     PagedKVBatchLayerView cache, MemoryArena& workspace,
                                     Tensor& out, CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "causal_softmax_attention_groups: workspace belongs to another device");
    }
    constexpr const char* op = "causal_softmax_attention_groups";
    validate_group_tensors(q, k, v, positions, valid_columns, kv_table_rows, groups, geometry, scale,
                           cache, out, op, execution.device());

#if NINFER_TARGET_B200
    if (detail::attention_backend(detail::kv_dtype(cache.storage), geometry.query_heads,
                                  geometry.kv_heads, kHeadDim) ==
        detail::AttentionBackend::Tcgen05) {
        detail::sm100_kv_append_attention_groups(detail::kv_dtype(cache.storage), q, k, v,
                                                 positions, valid_columns, kv_table_rows, scale,
                                                 cache, groups, out, execution);
        return;
    }
#endif
    const std::int32_t q_heads  = geometry.query_heads;
    const std::int32_t kv_heads = geometry.kv_heads;
    for (const auto& g : groups) {
        const auto begin   = static_cast<std::int32_t>(g.token_offset);
        const auto width   = static_cast<std::int32_t>(g.width);
        const auto batch   = static_cast<std::int32_t>(g.batch);
        const auto row     = static_cast<std::int32_t>(g.row);
        const auto columns = width * batch;
        const Tensor qg    = q.slice(2, begin, columns).view({kHeadDim, q_heads, width, batch});
        const Tensor kg    = k.slice(2, begin, columns).view({kHeadDim, kv_heads, width, batch});
        const Tensor vg    = v.slice(2, begin, columns).view({kHeadDim, kv_heads, width, batch});
        Tensor og          = out.slice(2, begin, columns).view({kHeadDim, q_heads, width, batch});
        causal_softmax_attention(qg, kg, vg, positions.slice(0, begin, columns).view({width, batch}),
                                 valid_columns.slice(0, row, batch),
                                 kv_table_rows.slice(0, row, batch), geometry, scale, cache,
                                 g.envelope, workspace, og, execution);
    }
}

void causal_softmax_attention_cached(const Tensor& q, const Tensor& positions,
                                     AttentionHeadGeometry geometry, float scale,
                                     const ConstPagedKVLayerView& cache,
                                     CausalAttentionExecutionEnvelope envelope,
                                     MemoryArena& workspace, Tensor& out,
                                     CudaExecutionView execution) {
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(
            "causal_softmax_attention_cached: workspace belongs to another device");
    }
    constexpr const char* op = "causal_softmax_attention_cached";
    validate_attention_tensors(q, positions, out, geometry, cache, envelope, scale, op,
                               execution.device());

#if NINFER_TARGET_B200
    if (detail::attention_backend(detail::kv_dtype(cache.storage), geometry.query_heads,
                                  geometry.kv_heads, kHeadDim) ==
        detail::AttentionBackend::Tcgen05) {
        detail::sm100_kv_cached_attention(detail::kv_dtype(cache.storage), q, positions, scale,
                                          cache, envelope, workspace, out, execution);
        return;
    }
#endif
    if (cache.storage == KvCacheStorage::BFloat16) {
        detail::bf16_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Fp8E4m3fnGroup128Ue8m0) {
#if NINFER_TARGET_B200
        // The fp8 FMHA kernels are an SM120 family; on B200 the fp8 cache is only served by
        // the tcgen05 path above.
        throw std::invalid_argument("causal_softmax_attention_cached: fp8 is not available on b200");
#else
        detail::fp8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                        execution);
        return;
#endif
    }

    if (cache.storage == KvCacheStorage::Int8Group64) {
        detail::int8_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                         execution);
        return;
    }

    if (cache.storage == KvCacheStorage::Nvfp4Group16) {
        detail::nvfp4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                          execution);
        return;
    }

#if NINFER_TARGET_B200
    // The k8v4 FMHA kernels are an SM120 family; on B200 the k8v4 cache is only served by
    // the tcgen05 path above.
    throw std::invalid_argument("causal_softmax_attention_cached: k8v4 is not available on b200");
#else
    detail::k8v4_kv_cached_attention(q, positions, scale, cache, envelope, workspace, out,
                                     execution);
#endif
}

} // namespace ninfer::ops
