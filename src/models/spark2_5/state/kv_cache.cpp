#include "models/spark2_5/state/kv_cache.h"

#include <limits>
#include <stdexcept>

namespace ninfer::models::spark2_5 {

FullKVLayout plan_full_kv(LayoutBuilder& builder, std::uint32_t full_layers, std::uint32_t capacity,
                          KvCacheStorage storage, std::int32_t table_rows) {
    if (full_layers == 0 || capacity == 0 || table_rows <= 0)
        throw std::invalid_argument("Spark2_5 full KV geometry is invalid");
    const auto schema = paged_kv_storage_layout(storage, kAttentionHeadDim);
    KVPageGeometry geometry;
    geometry.planes.reserve(std::size_t(full_layers) * schema.planes_per_layer());
    for (std::uint32_t layer = 0; layer < full_layers; ++layer) {
        geometry.planes.push_back(
            {schema.key.data_dtype, schema.key.data_leading_extent, kKvHeads, 256});
        geometry.planes.push_back(
            {schema.value.data_dtype, schema.value.data_leading_extent, kKvHeads, 256});
        if (schema.key.has_scale())
            geometry.planes.push_back(
                {schema.key.scale_dtype, schema.key.scale_leading_extent, kKvHeads, 256});
        if (schema.value.has_scale())
            geometry.planes.push_back(
                {schema.value.scale_dtype, schema.value.scale_leading_extent, kKvHeads, 256});
    }
    const std::uint32_t pages = 1U + (capacity - 1U) / std::uint32_t(kPagedKVPageSize);
    return {plan_kv_page_layout(geometry), plan_kv_execution_tables(builder, {pages, table_rows}),
            schema, full_layers, capacity};
}

FullKVCache::FullKVCache(MemorySpan context, MemorySpan fixed, const FullKVLayout& layout)
    : layout_(layout), context_(context),
      tables_(fixed, layout.tables, context, layout.pages.page_bytes) {
    if (layout_.pages.planes.size() !=
        std::size_t(layout_.layers) * layout_.storage.planes_per_layer())
        throw std::invalid_argument("Spark2_5 full KV plane inventory is inconsistent");
}

PagedKVBatchLayerView FullKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layout_.layers) throw std::out_of_range("Spark2_5 full KV layer");
    const auto& s       = layout_.storage;
    const std::size_t b = std::size_t(layer) * s.planes_per_layer();
    const auto plane    = [&](std::size_t index) {
        return PagedPlaneView{{context_, tables_.matrix(), layout_.pages.geometry.page_tokens},
                              layout_.pages.planes.at(index)};
    };
    return {.k_pages       = plane(b),
            .v_pages       = plane(b + 1),
            .k_scale_pages = s.key.has_scale() ? plane(b + 2) : PagedPlaneView{},
            .v_scale_pages =
                s.value.has_scale() ? plane(b + 2 + s.key.has_scale()) : PagedPlaneView{},
            .head_dim     = s.head_dim,
            .num_kv_heads = kKvHeads,
            .storage      = s.storage};
}

} // namespace ninfer::models::spark2_5
