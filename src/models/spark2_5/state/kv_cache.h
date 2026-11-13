#pragma once

#include "core/layout.h"
#include "core/paged_kv_cache.h"
#include "models/common/context/storage.h"

#include <cstdint>

namespace ninfer::models::spark2_5 {

// Attention geometry of the model: grouped-query D256, 8 query heads over 2 KV heads.
inline constexpr std::int32_t kAttentionHeadDim = 256;
inline constexpr std::int32_t kQueryHeads       = 8;
inline constexpr std::int32_t kKvHeads          = 2;

// Paged history of the full-attention layers. Sliding layers keep their ring in the StateImage.
struct FullKVLayout {
    KVPageLayout pages;
    KVExecutionTableLayout tables;
    PagedKVStorageLayout storage;
    std::uint32_t layers   = 0;
    std::uint32_t capacity = 0;
};

[[nodiscard]] FullKVLayout plan_full_kv(LayoutBuilder& builder, std::uint32_t full_layers,
                                        std::uint32_t capacity, KvCacheStorage storage,
                                        std::int32_t table_rows);

// Borrows the Context arena that holds the page records (`context`) and the fixed backing that
// holds the execution tables (`fixed`). Row ownership and page allocation belong to the Context.
class FullKVCache {
public:
    FullKVCache(MemorySpan context, MemorySpan fixed, const FullKVLayout& layout);

    [[nodiscard]] std::uint32_t layers() const noexcept { return layout_.layers; }

    [[nodiscard]] std::uint32_t capacity() const noexcept { return layout_.capacity; }

    [[nodiscard]] const KVPageLayout& pages() const noexcept { return layout_.pages; }

    [[nodiscard]] KvCacheStorage storage() const noexcept { return layout_.storage.storage; }

    // Every row of the execution table is addressable; callers select rows with kv_table_rows.
    [[nodiscard]] PagedKVBatchLayerView batch_layer_view(std::uint32_t full_layer) const;

    [[nodiscard]] context::ExecutionTableStorage& tables() noexcept { return tables_; }

    [[nodiscard]] const context::ExecutionTableStorage& tables() const noexcept { return tables_; }

private:
    FullKVLayout layout_;
    MemorySpan context_;
    context::ExecutionTableStorage tables_;
};

} // namespace ninfer::models::spark2_5
