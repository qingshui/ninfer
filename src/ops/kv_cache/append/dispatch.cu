#include "ops/kv_cache/append/dispatch.h"
#include <algorithm>
#include "core/device.h"
#include "ops/common/host/operands.h"
#include "ops/kv_cache/append/prefix.cuh"
#include "ops/kv_cache/kernels/write.h"

namespace ninfer::ops::detail {
namespace {

kv::PagedWriteInstance select_append(KvCacheStorage storage, int heads, int tokens) {
    using Instance = kv::PagedWriteInstance;
    if (storage == KvCacheStorage::BFloat16)
        return heads == 4 ? Instance::Copy128 : Instance::Copy96;
    return tokens >= 128 && heads == 2 ? Instance::Page8x256 : Instance::WarpRows256;
}

int select_cyclic_threads(int rows, int batch) { return rows * batch <= 128 ? 64 : 128; }

template <int Dim, int Heads, int Capacity, int Threads>
void launch_cyclic(const Tensor& k, const Tensor& v, const Tensor& positions, const Tensor& counts,
                   CyclicKVCacheLayerView cache, KVCacheAppendPrefixExecutionEnvelope envelope,
                   cudaStream_t stream) {
    constexpr int UnitsPerToken = Heads * Dim / 8;
    // A row writes at most its last Capacity tokens.
    const int rows = static_cast<int>(std::min<std::uint32_t>(envelope.max_count, Capacity));
    const dim3 grid((rows * UnitsPerToken + Threads - 1) / Threads, k.ne[3]);
    prefix_cyclic_kernel<Dim, Heads, Capacity, Threads><<<grid, Threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int64_t*>(cache.k.records.byte_offsets.data),
        common::operand::state_field_base<__nv_bfloat16>(cache.k),
        common::operand::state_field_base<__half>(cache.v), envelope.min_count, envelope.max_count,
        k.ne[2], cache.padded_capacity);
}

} // namespace

void kv_cache_append_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                              PagedKVLayerView cache, cudaStream_t stream) {
    kv::launch_paged_write(select_append(cache.storage, k.ne[1], k.ne[2]), k, v, positions, cache,
                           stream);
}

void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, const Tensor& table_rows,
                                     PagedKVBatchLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream) {
    if (envelope.max_count == 0) return;
    if (cache.head_dim == 256) {
        kv::launch_paged_write(kv::PagedWriteInstance::BatchRows256, k, v, positions, counts,
                               table_rows, cache, stream);
        return;
    }
    const dim3 grid(1 + (envelope.max_count - 1) / 4, k.ne[3]);
    prefix_paged_kernel<<<grid, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const std::int32_t*>(positions.data),
        static_cast<const std::int32_t*>(counts.data),
        static_cast<const std::int32_t*>(table_rows.data),
        static_cast<__nv_bfloat16*>(cache.k_pages.data()),
        static_cast<__half*>(cache.v_pages.data()),
        static_cast<const std::int64_t*>(cache.k_pages.records.page_offsets.data),
        cache.k_pages.records.page_offsets.ne[0], envelope.min_count, envelope.max_count, k.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}

void kv_cache_append_prefix_dispatch(const Tensor& k, const Tensor& v, const Tensor& positions,
                                     const Tensor& counts, CyclicKVCacheLayerView cache,
                                     KVCacheAppendPrefixExecutionEnvelope envelope,
                                     cudaStream_t stream) {
    if (envelope.max_count == 0) return;
    const bool wide = cache.head_dim == 256;
    const int rows =
        static_cast<int>(std::min<std::uint32_t>(envelope.max_count, wide ? 512u : 4096u));
    const int threads = select_cyclic_threads(rows, k.ne[3]);
    const auto launch = [&]<int Dim, int Heads, int Capacity>() {
        if (threads == 64)
            launch_cyclic<Dim, Heads, Capacity, 64>(k, v, positions, counts, cache, envelope,
                                                    stream);
        else
            launch_cyclic<Dim, Heads, Capacity, 128>(k, v, positions, counts, cache, envelope,
                                                     stream);
    };
    if (wide)
        launch.template operator()<256, 2, 512>();
    else if (cache.capacity == 2048)
        launch.template operator()<128, 8, 2048>();
    else
        launch.template operator()<128, 8, 4096>();
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
