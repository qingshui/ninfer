#pragma once

#include "ops/common/cuda/memory.cuh"
#include "ops/common/cuda/reduce.cuh"
#include "ops/common/numeric/convert.cuh"
#include "ops/kv_cache/kernels/fp8.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/kv_cache/kernels/write.h"
#include "ops/kv_cache/kernels/hadamard.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace ninfer::ops::detail::kv {
template <typename Geometry>
__device__ __forceinline__ void
write_fp8_row(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
              std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
              std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v, int token,
              int kv_head, std::int64_t record, int page_off, int lane) {
    float values[8];
#pragma unroll
    for (int r = 0; r < 8; ++r)
        values[r] = __bfloat162float(k[source_offset<Geometry>(kv_head, lane + 32 * r, token)]);
    normalized_hadamard_d256_inplace(values, lane);
    const auto code  = Fp8G128Layout::code_offset(record, kv_head, 0, page_off);
    const auto scale = Fp8G128Layout::scale_offset(record, kv_head, page_off);
    store_fp8_d256<true>(values, cache_k + code, scale_k + scale, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r)
        values[r] = __bfloat162float(v[source_offset<Geometry>(kv_head, lane + 32 * r, token)]);
    store_fp8_d256<true>(values, cache_v + code, scale_v + scale, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_fp8_kernel(const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
                          const std::int32_t* __restrict__ positions, Metadata metadata,
                          std::uint8_t* __restrict__ cache_k, std::uint8_t* __restrict__ cache_v,
                          std::uint8_t* __restrict__ scale_k, std::uint8_t* __restrict__ scale_v,
                          std::int32_t width) {
    constexpr int Warps         = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int unit              = static_cast<int>(blockIdx.x) * Warps + warp;
    const int units             = tokens * Geometry::KVHeads;
    if (unit >= units) return;

    const int kv_head               = unit % Geometry::KVHeads;
    const int token                 = unit / Geometry::KVHeads;
    const int position              = positions[0] + token;
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? lookup_page(block_table, position) : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    write_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                            position & kPageMask, lane);
}

// Independent request rows share physical planes, but select separate input and table rows.
template <typename Geometry, bool Masked>
__launch_bounds__(256) __global__
    void write_fp8_batch_kernel(const __nv_bfloat16* k, const __nv_bfloat16* v,
                                const std::int32_t* positions, BatchMetadata<Masked> metadata,
                                std::uint8_t* cache_k, std::uint8_t* cache_v, std::uint8_t* scale_k,
                                std::uint8_t* scale_v, std::int32_t width) {
    const int batch = blockIdx.z;
    metadata.table_rows += batch;
    if constexpr (Masked) metadata.valid_columns += batch;
    const int tokens = metadata.valid_tokens(width);
    const int unit   = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (unit >= tokens * Geometry::KVHeads) return;
    const int token         = unit / Geometry::KVHeads;
    const int kv_head       = unit % Geometry::KVHeads;
    const int lane          = threadIdx.x & 31;
    const auto input_offset = static_cast<std::int64_t>(batch) * width * 256 * Geometry::KVHeads;
    const int position      = positions[batch * width] + token;
    const auto* table       = metadata.block_table();
    std::int64_t page       = lane == 0 ? lookup_page(table, position) : 0;
    page                    = __shfl_sync(0xffffffffU, page, 0);
    write_fp8_row<Geometry>(k + input_offset, v + input_offset, cache_k, cache_v, scale_k, scale_v,
                            token, kv_head, page, position & kPageMask, lane);
}

struct Fp8GroupWriteParams {
    const __nv_bfloat16* k;
    const __nv_bfloat16* v;
    const std::int32_t* positions;
    const std::int64_t* tables;
    const std::int32_t* valid_columns;
    const std::int32_t* table_rows;
    std::uint8_t* cache_k;
    std::uint8_t* cache_v;
    std::uint8_t* scale_k;
    std::uint8_t* scale_v;
    std::int32_t table_stride;
    std::int32_t columns; // token axis columns covered by the groups
    std::int32_t groups;  // ascending token_base
    PagedWriteGroup group[kPagedWriteMaxGroups];
};

// One warp per (column, KV head) of the token axis; columns outside every group or beyond an
// entry's valid columns write nothing.
template <typename Geometry>
__launch_bounds__(256) __global__
    void write_fp8_groups_kernel(const __grid_constant__ Fp8GroupWriteParams p) {
    const int unit   = blockIdx.x * 8 + (threadIdx.x >> 5);
    const int column = unit / Geometry::KVHeads;
    if (column >= p.columns) return;
    int low = 0, high = p.groups - 1; // the last group that starts at or before the column
    while (low < high) {
        const int mid = (low + high + 1) >> 1;
        if (p.group[mid].token_base <= column)
            low = mid;
        else
            high = mid - 1;
    }
    const PagedWriteGroup g = p.group[low];
    const int local         = column - g.token_base;
    if (local < 0 || local >= g.width * g.batch) return;
    const int entry = local / g.width;
    const int token = local - entry * g.width;
    const int row   = g.row + entry;
    const int valid = p.valid_columns[row];
    if (token >= (valid < g.width ? valid : g.width)) return;
    const int lane     = threadIdx.x & 31;
    const int position = p.positions[g.token_base + entry * g.width] + token;
    const auto* table  = p.tables + static_cast<std::int64_t>(p.table_rows[row]) * p.table_stride;
    std::int64_t page  = lane == 0 ? lookup_page(table, position) : 0;
    page               = __shfl_sync(0xffffffffU, page, 0);
    write_fp8_row<Geometry>(p.k, p.v, p.cache_k, p.cache_v, p.scale_k, p.scale_v, column,
                            unit % Geometry::KVHeads, page, position & kPageMask, lane);
}

template <typename Geometry, typename Metadata>
__launch_bounds__(256) __global__
    void write_fp8_page_kernel(const __nv_bfloat16* __restrict__ k,
                               const __nv_bfloat16* __restrict__ v,
                               const std::int32_t* __restrict__ positions, Metadata metadata,
                               std::uint8_t* __restrict__ cache_k,
                               std::uint8_t* __restrict__ cache_v,
                               std::uint8_t* __restrict__ scale_k,
                               std::uint8_t* __restrict__ scale_v, std::int32_t width) {
    constexpr int TokensPerTile = 8;
    constexpr unsigned FullMask = 0xffffffffU;
    const int tokens            = metadata.valid_tokens(width);
    const int warp              = static_cast<int>(threadIdx.x) >> 5;
    const int lane              = static_cast<int>(threadIdx.x) & 31;
    const int kv_head           = static_cast<int>(blockIdx.y);
    const int tile_delta        = static_cast<int>(blockIdx.x);
    const int base_position     = positions[0];
    const int tile_position     = (base_position / TokensPerTile + tile_delta) * TokensPerTile;
    const int logical_page      = tile_position >> kPageShift;
    const int token_begin       = max(0, tile_position - base_position);
    const int token_end         = min(tokens, tile_position + TokensPerTile - base_position);
    if (token_begin >= token_end) return;

    const int token = token_begin + warp;
    if (token >= token_end) return;
    const std::int64_t* block_table = metadata.block_table();
    std::int64_t record             = lane == 0 ? block_table[logical_page] : 0;
    record                          = __shfl_sync(FullMask, record, 0);
    const int position              = base_position + token;
    write_fp8_row<Geometry>(k, v, cache_k, cache_v, scale_k, scale_v, token, kv_head, record,
                            position & kPageMask, lane);
}

} // namespace ninfer::ops::detail::kv
