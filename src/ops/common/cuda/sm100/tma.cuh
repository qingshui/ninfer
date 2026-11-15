#pragma once

// TMA helpers for SM100 kernels: host tensor-map construction and the device-side copy forms the
// tcgen05 mainloop needs (2D tile loads, flat bulk loads, CTA-pair loads).

#include "ops/common/cuda/memory.cuh"

#include <cuda.h>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::common::sm100 {

// Mask that clears the CTA-rank bit of a shared::cluster address so that a CTA-pair copy signals
// the leader CTA's barrier regardless of which CTA of the pair issues it.
inline constexpr std::uint32_t kPeerBitMask = 0xFEFFFFFFu;

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------

inline void check_driver(CUresult status, const char* operation) {
    if (status == CUDA_SUCCESS) return;
    const char* name = nullptr;
    (void)cuGetErrorName(status, &name);
    throw std::runtime_error(std::string(operation) + ": " + (name ? name : "CUDA driver error"));
}

// Row-major 2D tensor: `inner` contiguous elements per row, `outer` rows, rows `row_bytes` apart.
// The box is [box_inner x box_outer] elements; elements outside the tensor are zero-filled.
inline CUtensorMap make_tensor_map_2d(const void* base, CUtensorMapDataType type,
                                      std::uint64_t inner, std::uint64_t outer,
                                      std::uint64_t row_bytes, std::uint32_t box_inner,
                                      std::uint32_t box_outer, CUtensorMapSwizzle swizzle,
                                      CUtensorMapL2promotion promotion,
                                      const char* operation) {
    CUtensorMap map{};
    const std::uint64_t dims[]     = {inner, outer};
    const std::uint64_t strides[]  = {row_bytes};
    const std::uint32_t box[]      = {box_inner, box_outer};
    const std::uint32_t elements[] = {1, 1};
    check_driver(cuTensorMapEncodeTiled(&map, type, 2, const_cast<void*>(base), dims, strides, box,
                                        elements, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
                                        promotion, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
                 operation);
    return map;
}

// ---------------------------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------------------------

// 2D tile load into this CTA's shared memory; completion counts bytes on a barrier of this CTA.
__device__ __forceinline__ void tma_load_tile_2d(void* destination, const CUtensorMap* map,
                                                 int c0, int c1, std::uint64_t* barrier,
                                                 std::uint64_t cache_hint) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                 ".L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
                 :
                 : "r"(smem_addr(destination)), "l"(map), "r"(smem_addr(barrier)), "r"(c0),
                   "r"(c1), "l"(cache_hint)
                 : "memory");
}

// CTA-pair load: executed by both CTAs of the pair, each filling its own shared memory; the
// transaction bytes are accounted on the leader CTA's barrier.
__device__ __forceinline__ void tma_load_tile_2d_cta_pair(void* destination,
                                                          const CUtensorMap* map, int c0, int c1,
                                                          std::uint64_t* leader_barrier,
                                                          std::uint64_t cache_hint) {
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global"
                 ".mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%3, %4}], [%2], %5;"
                 :
                 : "r"(smem_addr(destination)), "l"(map),
                   "r"(smem_addr(leader_barrier) & kPeerBitMask), "r"(c0), "r"(c1),
                   "l"(cache_hint)
                 : "memory");
}

// Flat global -> shared copy (no tensor map); size and addresses are 16-byte multiples.
__device__ __forceinline__ void bulk_load(void* destination, const void* source,
                                          std::uint32_t bytes, std::uint64_t* barrier) {
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
                 "[%0], [%1], %2, [%3];"
                 :
                 : "r"(smem_addr(destination)), "l"(source), "r"(bytes), "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void tma_prefetch_descriptor(const CUtensorMap* map) {
    asm volatile("prefetch.tensormap [%0];" : : "l"(map) : "memory");
}

// Barrier arrive with transaction count, and a spin wait with parity; thin forms of the
// mbarrier operations used by the pipeline.
__device__ __forceinline__ void mbarrier_init(std::uint64_t* barrier, std::uint32_t arrivals) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(arrivals)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_arrive_expect_tx(std::uint64_t* barrier,
                                                          std::uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
                 :
                 : "r"(smem_addr(barrier)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void mbarrier_arrive(std::uint64_t* barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
                 :
                 : "r"(smem_addr(barrier))
                 : "memory");
}

__device__ __forceinline__ void mbarrier_wait(std::uint64_t* barrier, std::uint32_t parity) {
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "wait_%=:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1;\n"
                 "@done bra done_%=;\n"
                 "bra wait_%=;\n"
                 "done_%=:\n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(parity)
                 : "memory");
}

// Arrive on a barrier in another CTA of the cluster (by shared::cluster address of the barrier).
__device__ __forceinline__ void mbarrier_arrive_remote(std::uint64_t* barrier,
                                                       std::uint32_t cta_rank) {
    asm volatile("{\n"
                 ".reg .b32 remote;\n"
                 "mapa.shared::cluster.u32 remote, %0, %1;\n"
                 "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remote];\n"
                 "}\n"
                 :
                 : "r"(smem_addr(barrier)), "r"(cta_rank)
                 : "memory");
}

__device__ __forceinline__ void cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n"
                 "barrier.cluster.wait.acquire.aligned;"
                 :
                 :
                 : "memory");
}

__device__ __forceinline__ std::uint32_t cluster_cta_rank() {
    std::uint32_t rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

} // namespace ninfer::ops::common::sm100
