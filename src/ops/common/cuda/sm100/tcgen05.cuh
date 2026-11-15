#pragma once

// SM100 (B200, sm_100a) tensor-memory primitives: TMEM allocation, tcgen05.mma issue forms
// (kind::mxf4nvf4.block_scale, kind::f16, kind::f8f6f4), tcgen05.ld / tcgen05.cp, commit and the
// fences that order them. Callers own barrier protocols and shared-memory layouts.
//
// Every form takes the CTA group (1 or 2) as a template argument. A group-2 MMA is issued by one
// thread of the leader CTA of a CTA pair and consumes operands from both CTAs' shared memory.

#include "ops/common/cuda/memory.cuh"

#include <cstdint>

namespace ninfer::ops::common::sm100 {

inline constexpr int kTmemColumns = 512;
inline constexpr int kTmemLanes   = 128;

// ---------------------------------------------------------------------------------------------
// Thread election and fences
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ bool elect_one() {
    std::uint32_t selected = 0;
    asm volatile("{\n"
                 ".reg .pred p;\n"
                 "elect.sync _|p, 0xffffffff;\n"
                 "selp.u32 %0, 1, 0, p;\n"
                 "}\n"
                 : "=r"(selected));
    return selected != 0;
}

__device__ __forceinline__ void fence_before_thread_sync() {
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

__device__ __forceinline__ void fence_after_thread_sync() {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

// Orders generic-proxy shared-memory writes before the async proxy (TMA stores, UMMA reads).
__device__ __forceinline__ void fence_proxy_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

// ---------------------------------------------------------------------------------------------
// TMEM allocation. The whole warp executes these (".sync.aligned").
// ---------------------------------------------------------------------------------------------

template <int CtaGroup>
__device__ __forceinline__ void tmem_alloc(std::uint32_t* smem_slot, int columns) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const unsigned address = smem_addr(smem_slot);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :
                     : "r"(address), "r"(columns)
                     : "memory");
    else
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                     :
                     : "r"(address), "r"(columns)
                     : "memory");
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_relinquish_alloc_permit() {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
    else
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
}

template <int CtaGroup>
__device__ __forceinline__ void tmem_dealloc(std::uint32_t taddr, int columns) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(taddr), "r"(columns)
                     : "memory");
    else
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                     :
                     : "r"(taddr), "r"(columns)
                     : "memory");
}

// TMEM address: lane in bits [31:16], column in bits [15:0].
__device__ __forceinline__ constexpr std::uint32_t tmem_address(std::uint32_t base, int lane,
                                                                int column) {
    return base + (static_cast<std::uint32_t>(lane) << 16) + static_cast<std::uint32_t>(column);
}

// ---------------------------------------------------------------------------------------------
// Commit: the arrive on the mbarrier happens when every prior tcgen05 async operation (MMA and
// tcgen05.cp) issued by this thread has completed. The barrier address is a shared::cluster
// address; for a CTA-local barrier the shared::cta window address is also a valid one.
// ---------------------------------------------------------------------------------------------

template <int CtaGroup>
__device__ __forceinline__ void umma_commit(std::uint64_t* barrier) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const unsigned address = smem_addr(barrier);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :
                     : "r"(address)
                     : "memory");
    else
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                     :
                     : "r"(address)
                     : "memory");
}

// Arrives on the barrier at the same shared-memory offset in every CTA selected by `cta_mask`.
template <int CtaGroup>
__device__ __forceinline__ void umma_commit_multicast(std::uint64_t* barrier,
                                                      std::uint16_t cta_mask) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const unsigned address = smem_addr(barrier);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster."
                     "multicast::cluster.b64 [%0], %1;"
                     :
                     : "r"(address), "h"(cta_mask)
                     : "memory");
    else
        asm volatile("tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
                     "multicast::cluster.b64 [%0], %1;"
                     :
                     : "r"(address), "h"(cta_mask)
                     : "memory");
}

// ---------------------------------------------------------------------------------------------
// tcgen05.cp: shared memory -> tensor memory. 32x128b.warpx4 copies a 512-byte block-scale atom
// (32 rows of 16 bytes) and replicates it into the four 32-lane quadrants.
// ---------------------------------------------------------------------------------------------

template <int CtaGroup>
__device__ __forceinline__ void tmem_copy_32x128b_warpx4(std::uint32_t taddr, std::uint64_t sdesc) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    if constexpr (CtaGroup == 1)
        asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                     :
                     : "r"(taddr), "l"(sdesc)
                     : "memory");
    else
        asm volatile("tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                     :
                     : "r"(taddr), "l"(sdesc)
                     : "memory");
}

// ---------------------------------------------------------------------------------------------
// tcgen05.ld: tensor memory -> registers. A warp may only touch its own 32-lane quadrant
// (warp_id % 4). Each thread receives one lane's `Columns` consecutive 32-bit columns.
// ---------------------------------------------------------------------------------------------

__device__ __forceinline__ void tmem_load_wait() {
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
}

__device__ __forceinline__ void tmem_store_wait() {
    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

template <int Columns>
__device__ __forceinline__ void tmem_load_32x32b(std::uint32_t taddr, std::uint32_t (&r)[Columns]) {
    static_assert(Columns == 8 || Columns == 16 || Columns == 32 || Columns == 64 ||
                  Columns == 128);
    if constexpr (Columns == 8) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
                       "=r"(r[6]), "=r"(r[7])
                     : "r"(taddr));
    } else if constexpr (Columns == 16) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
                     "[%16];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
                       "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                       "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                     : "r"(taddr));
    } else if constexpr (Columns == 32) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                     "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "
                     "%30, %31}, [%32];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
                       "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                       "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
                       "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]),
                       "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]),
                       "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                     : "r"(taddr));
    } else if constexpr (Columns == 64) {
        asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
                     "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, "
                     "%16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, "
                     "%30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
                     "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, "
                     "%58, %59, %60, %61, %62, %63}, [%64];"
                     : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]),
                       "=r"(r[6]), "=r"(r[7]), "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                       "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]), "=r"(r[16]),
                       "=r"(r[17]), "=r"(r[18]), "=r"(r[19]), "=r"(r[20]), "=r"(r[21]),
                       "=r"(r[22]), "=r"(r[23]), "=r"(r[24]), "=r"(r[25]), "=r"(r[26]),
                       "=r"(r[27]), "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31]),
                       "=r"(r[32]), "=r"(r[33]), "=r"(r[34]), "=r"(r[35]), "=r"(r[36]),
                       "=r"(r[37]), "=r"(r[38]), "=r"(r[39]), "=r"(r[40]), "=r"(r[41]),
                       "=r"(r[42]), "=r"(r[43]), "=r"(r[44]), "=r"(r[45]), "=r"(r[46]),
                       "=r"(r[47]), "=r"(r[48]), "=r"(r[49]), "=r"(r[50]), "=r"(r[51]),
                       "=r"(r[52]), "=r"(r[53]), "=r"(r[54]), "=r"(r[55]), "=r"(r[56]),
                       "=r"(r[57]), "=r"(r[58]), "=r"(r[59]), "=r"(r[60]), "=r"(r[61]),
                       "=r"(r[62]), "=r"(r[63])
                     : "r"(taddr));
    } else {
        // 128 columns are exposed as two x64 loads to keep the register constraint list sane.
        std::uint32_t lo[64];
        std::uint32_t hi[64];
        tmem_load_32x32b<64>(taddr, lo);
        tmem_load_32x32b<64>(taddr + 64, hi);
#pragma unroll
        for (int i = 0; i < 64; ++i) {
            r[i]      = lo[i];
            r[i + 64] = hi[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------
// tcgen05.mma issue forms. One elected thread issues; D lives in tensor memory.
//   adesc / bdesc : 64-bit shared-memory matrix descriptors
//   idesc         : 32-bit instruction descriptor
//   accumulate    : false overwrites D (first k-block of a tile), true accumulates
// ---------------------------------------------------------------------------------------------

template <int CtaGroup>
__device__ __forceinline__ void umma_f16_ss(std::uint32_t tmem_d, std::uint64_t adesc,
                                            std::uint64_t bdesc, std::uint32_t idesc,
                                            bool accumulate) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const std::uint32_t scale_d = accumulate ? 1u : 0u;
    if constexpr (CtaGroup == 1)
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d)
                     : "memory");
    else
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d)
                     : "memory");
}

template <int CtaGroup>
__device__ __forceinline__ void umma_f8f6f4_ss(std::uint32_t tmem_d, std::uint64_t adesc,
                                               std::uint64_t bdesc, std::uint32_t idesc,
                                               bool accumulate) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const std::uint32_t scale_d = accumulate ? 1u : 0u;
    if constexpr (CtaGroup == 1)
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d)
                     : "memory");
    else
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d)
                     : "memory");
}

// E2M1 x E2M1 with UE4M3 scales, one scale per 16 elements (block16): K = 64 per instruction.
// Scale factors are read from tensor memory at tmem_sfa / tmem_sfb (four columns each).
template <int CtaGroup>
__device__ __forceinline__ void umma_mxf4nvf4_ss(std::uint32_t tmem_d, std::uint64_t adesc,
                                                 std::uint64_t bdesc, std::uint32_t idesc,
                                                 std::uint32_t tmem_sfa, std::uint32_t tmem_sfb,
                                                 bool accumulate) {
    static_assert(CtaGroup == 1 || CtaGroup == 2);
    const std::uint32_t scale_d = accumulate ? 1u : 0u;
    if constexpr (CtaGroup == 1)
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 "
                     "[%0], %1, %2, %3, [%5], [%6], p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d),
                       "r"(tmem_sfa), "r"(tmem_sfb)
                     : "memory");
    else
        asm volatile("{\n"
                     ".reg .pred p;\n"
                     "setp.ne.b32 p, %4, 0;\n"
                     "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 "
                     "[%0], %1, %2, %3, [%5], [%6], p;\n"
                     "}\n"
                     :
                     : "r"(tmem_d), "l"(adesc), "l"(bdesc), "r"(idesc), "r"(scale_d),
                       "r"(tmem_sfa), "r"(tmem_sfb)
                     : "memory");
}

} // namespace ninfer::ops::common::sm100
