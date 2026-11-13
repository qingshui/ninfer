#pragma once

#include "core/platform.h"
#include "ops/common/cuda/memory.cuh"
#include "ops/common/quant/nvfp4.cuh"

namespace ninfer::ops::common {

__device__ __forceinline__ void ldmatrix_x2(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4(unsigned& r0, unsigned& r1, unsigned& r2, unsigned& r3,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x2_t(unsigned& r0, unsigned& r1, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r0), "=r"(r1)
                 : "r"(addr));
}

__device__ __forceinline__ void ldmatrix_x4_t(unsigned& r0, unsigned& r1, unsigned& r2,
                                              unsigned& r3, unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                         unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3, unsigned a0,
                                        unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// FP16 products accumulated in two packed half2 registers (rows g and g+8 of the C fragment).
// The caller bounds the partial-sum range; GeForce Blackwell issues this form at twice the
// rate of the FP32-accumulate form.
__device__ __forceinline__ void mma_f16_acc16(unsigned& c0, unsigned& c1, unsigned a0, unsigned a1,
                                              unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c0), "+r"(c1)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int& c0, int& c1, int& c2, int& c3, unsigned a0, unsigned a1,
                                       unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// On RTX 5090, block-scaled FP8 avoids the plain form's half-rate FP32 accumulation.
// UE8M0 0x7f is exactly 1: preserve the raw E4M3 dot product and keep the callers'
// existing row scales in FP32 after the MMA. No operand repacking is needed.
__device__ __forceinline__ void mma_fp8_e4m3(float& c0, float& c1, float& c2, float& c3,
                                             unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                             unsigned b0, unsigned b1) {
#if NINFER_TARGET_B200
    // SM100 has no mma.sync block-scale path (tcgen05 owns it there). The plain
    // E4M3 form has the same fragment layout and identical unit-scale products;
    // only the issue rate differs, which the tuning phase owns.
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    constexpr unsigned kUnitScale8          = 0x7Fu;
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(kUnitScale8),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(kUnitScale8), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#endif
}

// scale_vec::1X selects one byte from each packed scale word. For B,
// lane%4==0 supplies column lane/4.
// A uses lanes 0 and 1 in each quad for rows lane/4 and lane/4+8.
__device__ __forceinline__ void mma_fp8_e4m3_scaled(float& c0, float& c1, float& c2, float& c3,
                                                    unsigned a0, unsigned a1, unsigned a2,
                                                    unsigned a3, unsigned b0, unsigned b1,
                                                    unsigned scale_a, unsigned scale_b,
                                                    unsigned short scale_byte = 0) {
#if NINFER_TARGET_B200
    // SM100 emulation of the SM120 block-scale form: decode the E4M3 fragments and
    // apply the per-row/per-column UE8M0 scale in FP32. Fragment layout (g = lane/4,
    // t = lane%4): a0/a2 = row g, a1/a3 = row g+8 with k columns 4t..4t+3 and
    // 16+4t..16+4t+3; b0/b1 = column g with the same k columns; c0/c1 = row g,
    // c2/c3 = row g+8 at columns 2t and 2t+1. Lane 4r supplies the row-r scale byte,
    // lane 4r+1 the row-(r+8) one and lane 4c the column-c scale byte.
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned shift = static_cast<unsigned>(scale_byte) * 8;
    const float sa_g = common::ue8m0_to_float((scale_a >> shift) & 0xFF);
    const float sa_g8 = __shfl_sync(0xFFFFFFFFu, sa_g, (g << 2) | 1);
    const float sb_0 = common::ue8m0_to_float((scale_b >> shift) & 0xFF);
    const float sb_1 = __shfl_sync(0xFFFFFFFFu, sb_0, (((t << 1) | 1) << 2));
    float sum_g0 = 0.0F, sum_g1 = 0.0F, sum_g8_0 = 0.0F, sum_g8_1 = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned av = __shfl_sync(0xFFFFFFFFu, j < 4 ? a0 : a2, (g << 2) | (j & 3));
        const unsigned av8 = __shfl_sync(0xFFFFFFFFu, j < 4 ? a1 : a3, (g << 2) | (j & 3));
        const unsigned bv = __shfl_sync(0xFFFFFFFFu, j < 4 ? b0 : b1, ((t << 1) << 2) | (j & 3));
        const unsigned bv1 = __shfl_sync(0xFFFFFFFFu, j < 4 ? b0 : b1,
                                         (((t << 1) | 1) << 2) | (j & 3));
        float pd0 = 0.0F, pd1 = 0.0F, pd8_0 = 0.0F, pd8_1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a0v = common::decode_e4m3((av >> (i * 8)) & 0xFF);
            const float a1v = common::decode_e4m3((av8 >> (i * 8)) & 0xFF);
            const float b0v = common::decode_e4m3((bv >> (i * 8)) & 0xFF);
            const float b1v = common::decode_e4m3((bv1 >> (i * 8)) & 0xFF);
            pd0 = fmaf(a0v, b0v, pd0);
            pd1 = fmaf(a0v, b1v, pd1);
            pd8_0 = fmaf(a1v, b0v, pd8_0);
            pd8_1 = fmaf(a1v, b1v, pd8_1);
        }
        sum_g0 = fmaf(pd0, sa_g * sb_0, sum_g0);
        sum_g1 = fmaf(pd1, sa_g * sb_1, sum_g1);
        sum_g8_0 = fmaf(pd8_0, sa_g8 * sb_0, sum_g8_0);
        sum_g8_1 = fmaf(pd8_1, sa_g8 * sb_1, sum_g8_1);
    }
    c0 += sum_g0;
    c1 += sum_g1;
    c2 += sum_g8_0;
    c3 += sum_g8_1;
#else
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X."
                 "m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
                 "{%10}, {%11,%12}, {%13}, {%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(scale_a),
                   "h"(scale_byte), "h"(kScaleThreadId), "r"(scale_b), "h"(scale_byte),
                   "h"(kScaleThreadId));
#endif
}

__device__ __forceinline__ void mma_tf32_bits(float& c0, float& c1, float& c2, float& c3,
                                              unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                              unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_tf32(float& c0, float& c1, float& c2, float& c3, float a0,
                                         float a1, float a2, float a3, float b0, float b1) {
    mma_tf32_bits(c0, c1, c2, c3, __float_as_uint(a0), __float_as_uint(a1), __float_as_uint(a2),
                  __float_as_uint(a3), __float_as_uint(b0), __float_as_uint(b1));
}

__device__ __forceinline__ void mma_nvfp4_e4m3(float& c0, float& c1, float& c2, float& c3,
                                               unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                               unsigned b0, unsigned b1, unsigned sfa,
                                               unsigned sfb) {
#if NINFER_TARGET_B200
    // SM100 emulation of the SM120 NVFP4 block-scale form: decode the E2M1
    // fragments and apply the per-16-column UE4M3 scales in FP32. Fragment layout
    // (g = lane/4, t = lane%4): a0/a2 = row g, a1/a3 = row g+8 with k columns
    // 8t..8t+7 and 32+8t..32+8t+7; b0/b1 = column g with the same k columns;
    // c0/c1 = row g, c2/c3 = row g+8 at columns 2t and 2t+1. Lane 4r+0/4r+2 supply
    // the row-r scale word and lane 4r+1/4r+3 the row-(r+8) one; byte b of a scale
    // word covers k block b (16 columns per block). Lane 4c supplies the column-c
    // scale word. E2M1 products are exact in FP32; scales are applied per k block.
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int g = lane >> 2, t = lane & 3;
    const unsigned sa_g_word = __shfl_sync(0xFFFFFFFFu, sfa, g << 2);
    const unsigned sa_g8_word = __shfl_sync(0xFFFFFFFFu, sfa, (g << 2) | 1);
    const unsigned sb_0_word = __shfl_sync(0xFFFFFFFFu, sfb, (t << 1) << 2);
    const unsigned sb_1_word = __shfl_sync(0xFFFFFFFFu, sfb, ((t << 1) | 1) << 2);
    float sa_g[4], sa_g8[4], sb_0[4], sb_1[4];
#pragma unroll
    for (int block = 0; block < 4; ++block) {
        sa_g[block] = common::decode_ue4m3((sa_g_word >> (block * 8)) & 0xFF);
        sa_g8[block] = common::decode_ue4m3((sa_g8_word >> (block * 8)) & 0xFF);
        sb_0[block] = common::decode_ue4m3((sb_0_word >> (block * 8)) & 0xFF);
        sb_1[block] = common::decode_ue4m3((sb_1_word >> (block * 8)) & 0xFF);
    }
    float sum_g0 = 0.0F, sum_g1 = 0.0F, sum_g8_0 = 0.0F, sum_g8_1 = 0.0F;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const unsigned av = __shfl_sync(0xFFFFFFFFu, j < 4 ? a0 : a2, (g << 2) | (j & 3));
        const unsigned av8 = __shfl_sync(0xFFFFFFFFu, j < 4 ? a1 : a3, (g << 2) | (j & 3));
        const unsigned bv = __shfl_sync(0xFFFFFFFFu, j < 4 ? b0 : b1, ((t << 1) << 2) | (j & 3));
        const unsigned bv1 = __shfl_sync(0xFFFFFFFFu, j < 4 ? b0 : b1,
                                         (((t << 1) | 1) << 2) | (j & 3));
        float pd0 = 0.0F, pd1 = 0.0F, pd8_0 = 0.0F, pd8_1 = 0.0F;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float a0v = common::decode_e2m1((av >> (i * 4)) & 0xF);
            const float a1v = common::decode_e2m1((av8 >> (i * 4)) & 0xF);
            const float b0v = common::decode_e2m1((bv >> (i * 4)) & 0xF);
            const float b1v = common::decode_e2m1((bv1 >> (i * 4)) & 0xF);
            pd0 = fmaf(a0v, b0v, pd0);
            pd1 = fmaf(a0v, b1v, pd1);
            pd8_0 = fmaf(a1v, b0v, pd8_0);
            pd8_1 = fmaf(a1v, b1v, pd8_1);
        }
        const int block = j >> 1;
        sum_g0 = fmaf(pd0, sa_g[block] * sb_0[block], sum_g0);
        sum_g1 = fmaf(pd1, sa_g[block] * sb_1[block], sum_g1);
        sum_g8_0 = fmaf(pd8_0, sa_g8[block] * sb_0[block], sum_g8_0);
        sum_g8_1 = fmaf(pd8_1, sa_g8[block] * sb_1[block], sum_g8_1);
    }
    c0 += sum_g0;
    c1 += sum_g1;
    c2 += sum_g8_0;
    c3 += sum_g8_1;
#else
    constexpr unsigned short kScaleBlockId  = 0;
    constexpr unsigned short kScaleThreadId = 0;
    asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                 "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                 "{%0,%1,%2,%3}, "
                 "{%4,%5,%6,%7}, "
                 "{%8,%9}, "
                 "{%0,%1,%2,%3}, "
                 "{%10}, "
                 "{%11,%12}, "
                 "{%13}, "
                 "{%14,%15};\n"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(sfa),
                   "h"(kScaleBlockId), "h"(kScaleThreadId), "r"(sfb), "h"(kScaleBlockId),
                   "h"(kScaleThreadId));
#endif
}

} // namespace ninfer::ops::common
