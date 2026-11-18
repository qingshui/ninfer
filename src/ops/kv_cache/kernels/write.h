#pragma once

#include <cuda_runtime_api.h>

#include <span>

#include "core/paged_kv_cache.h"

namespace ninfer::ops::detail::kv {

enum class PagedWriteInstance {
    Copy96,
    Copy128,
    WarpRows256,
    Page8x256,
    BatchRows256,
};

// D256 packed-record writes from validated BF16 operands. The owning Op selects the instance;
// these bindings perform no tuning selection. Copy instances require batch size one;
// BatchRows256 accepts BF16 and quantized storage with device-selected valid prefixes.
void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, PagedKVLayerView cache, cudaStream_t stream);
void launch_paged_write(PagedWriteInstance instance, const Tensor& k, const Tensor& v,
                        const Tensor& positions, const Tensor& valid_columns,
                        const Tensor& table_rows, PagedKVBatchLayerView cache, cudaStream_t stream);

// `batch` consecutive entries of `width` columns of one token axis. Entry b uses entry `row + b` of
// the valid-column and table-row metadata, and its columns' positions run from the first column's.
struct PagedWriteGroup {
    std::int32_t token_base, width, batch, row;
};

inline constexpr int kPagedWriteMaxGroups = 72;

// FP8-E4M3FN-G128 storage with two KV heads: the live columns of every group on one token axis in
// a single launch. k/v are BF16 [256,2,T], positions I32 [T], and valid_columns/table_rows device
// I32 metadata addressed through the groups, which are ascending and disjoint in token_base.
void launch_paged_write_groups(const Tensor& k, const Tensor& v, const Tensor& positions,
                               const Tensor& valid_columns, const Tensor& table_rows,
                               PagedKVBatchLayerView cache, std::span<const PagedWriteGroup> groups,
                               cudaStream_t stream);

} // namespace ninfer::ops::detail::kv
