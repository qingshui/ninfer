#pragma once

// KV-cache access policies for the SM100 FMHA.
//
// The tensor-core kernel consumes two shared-memory tiles per KV block: K as the B operand of
// Q.K^T and V as the B operand of P.V. A policy states how a cache representation becomes those
// tiles:
//   BF16     K bf16 and V fp16 are copied as stored.
//   FP8      K stays E4M3 (native kind::f8f6f4 MMA with per-(key, 128-group) UE8M0 scales applied to
//            the accumulators); V is expanded to fp16.
//   INT8, K8V4, NVFP4
//            K is expanded to bf16 and V to fp16 in registers before the shared-memory store.
// Rotation flags describe the basis the cache lives in (the write kernels apply a normalized H256
// to K, and for NVFP4 / K8V4 also to V); Q is rotated to match and the output rotated back.

#include "ops/common/numeric/convert.cuh"
#include "ops/common/quant/fp8.cuh"
#include "ops/common/quant/nvfp4.cuh"
#include "ops/kv_cache/kernels/layout.cuh"
#include "ops/softmax_attention/backend.h"
#include "ops/softmax_attention/kernels/causal/sm100/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail::sm100_fmha {

__device__ __forceinline__ uint4 pack_u4(unsigned a, unsigned b, unsigned c, unsigned d) {
    return make_uint4(a, b, c, d);
}

__device__ __forceinline__ unsigned pack_half2_bits(float a, float b) {
    const __half2 value = __floats2half2_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&value);
}

__device__ __forceinline__ unsigned pack_bf16x2_bits(float a, float b) {
    const __nv_bfloat162 value = __floats2bfloat162_rn(a, b);
    return *reinterpret_cast<const unsigned*>(&value);
}

// Eight signed codes times one FP16 group multiplier, as two-element pairs.
template <class Pack>
__device__ __forceinline__ uint4 int8_dequant8(const std::int8_t* codes, float scale, Pack pack) {
    const int2 raw = common::load_vec<int2>(codes);
    const auto* c  = reinterpret_cast<const std::int8_t*>(&raw);
    return pack_u4(pack(c[0] * scale, c[1] * scale), pack(c[2] * scale, c[3] * scale),
                   pack(c[4] * scale, c[5] * scale), pack(c[6] * scale, c[7] * scale));
}

template <KvDType Kv>
struct KvPolicy;

// ---------------------------------------------------------------------------------------------
// BF16: K bf16, V fp16.
// ---------------------------------------------------------------------------------------------
template <>
struct KvPolicy<KvDType::Bf16> {
    static constexpr bool kRotate       = false;
    static constexpr bool kNativeFp8    = false;
    static constexpr bool kInverseRotate = false;

    __device__ static __forceinline__ uint4 load_k8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto index = kv::record_offset<256, 2>(record, head, row, d);
        return common::load_vec<uint4>(static_cast<const __nv_bfloat16*>(c.keys) + index);
    }
    __device__ static __forceinline__ uint4 load_v8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto index = kv::record_offset<256, 2>(record, head, row, d);
        return common::load_vec<uint4>(static_cast<const __half*>(c.values) + index);
    }
};

// ---------------------------------------------------------------------------------------------
// INT8 G64 with FP16 scales: decoded = code * scale. K rotated, V in original coordinates.
// ---------------------------------------------------------------------------------------------
template <>
struct KvPolicy<KvDType::Int8> {
    static constexpr bool kRotate        = true;
    static constexpr bool kNativeFp8     = false;
    static constexpr bool kInverseRotate = false;

    template <class Pack>
    __device__ static __forceinline__ uint4 load8(const void* codes, const void* scales,
                                                  std::int64_t record, int head, int row, int d,
                                                  Pack pack) {
        const auto code  = kv::Int8G64Layout::code_offset(record, head, d, row);
        const auto scale = kv::Int8G64Layout::scale_offset(record, head, d / kv::kInt8Group, row);
        const float s    = __half2float(static_cast<const __half*>(scales)[scale]);
        return int8_dequant8(static_cast<const std::int8_t*>(codes) + code, s, pack);
    }
    __device__ static __forceinline__ uint4 load_k8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        return load8(c.keys, c.key_scales, record, head, row, d, pack_bf16x2_bits);
    }
    __device__ static __forceinline__ uint4 load_v8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        return load8(c.values, c.value_scales, record, head, row, d, pack_half2_bits);
    }
};

// ---------------------------------------------------------------------------------------------
// FP8 E4M3 G128 with UE8M0 scales. K rotated, V in original coordinates.
// ---------------------------------------------------------------------------------------------
template <>
struct KvPolicy<KvDType::Fp8> {
    static constexpr bool kRotate        = true;
    static constexpr bool kNativeFp8     = true;
    static constexpr bool kInverseRotate = false;

    // Sixteen raw E4M3 codes (one K row slice).
    __device__ static __forceinline__ uint4 load_k16(const Sm100KvCache& c, std::int64_t record,
                                                     int head, int row, int d) {
        const auto code = kv::Fp8G128Layout::code_offset(record, head, d, row);
        return common::load_vec<uint4>(static_cast<const std::uint8_t*>(c.keys) + code);
    }
    __device__ static __forceinline__ void load_k_scales(const Sm100KvCache& c, std::int64_t record,
                                                         int head, int row, float& s0, float& s1) {
        const auto scale = kv::Fp8G128Layout::scale_offset(record, head, 0, row);
        const unsigned pair =
            common::load_vec<std::uint16_t>(static_cast<const std::uint8_t*>(c.key_scales) + scale);
        s0 = common::ue8m0_to_float(static_cast<std::uint8_t>(pair & 255u));
        s1 = common::ue8m0_to_float(static_cast<std::uint8_t>(pair >> 8));
    }
    __device__ static __forceinline__ uint4 load_v8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto code  = kv::Fp8G128Layout::code_offset(record, head, d, row);
        const auto scale = kv::Fp8G128Layout::scale_offset(record, head, d / 128, row);
        const int4 value = common::fp8_dequant_f16x8(
            static_cast<const std::uint8_t*>(c.values) + code,
            static_cast<const std::uint8_t*>(c.value_scales)[scale]);
        return make_uint4(value.x, value.y, value.z, value.w);
    }
};

// ---------------------------------------------------------------------------------------------
// NVFP4 G16 with UE4M3 scales. K and V rotated.
// ---------------------------------------------------------------------------------------------
template <>
struct KvPolicy<KvDType::Nvfp4> {
    static constexpr bool kRotate        = true;
    static constexpr bool kNativeFp8     = false;
    static constexpr bool kInverseRotate = true;

    __device__ static __forceinline__ uint4 load_k8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto code  = kv::Nvfp4G16Layout::code_offset(record, head, d, row);
        const auto scale = kv::Nvfp4G16Layout::scale_offset(record, head, d / kv::kNvfp4Group, row);
        const auto* bytes = static_cast<const std::uint8_t*>(c.keys) + code;
        const std::uint8_t sc = static_cast<const std::uint8_t*>(c.key_scales)[scale];
        const unsigned word   = common::load_vec<unsigned>(bytes);
        return pack_u4(common::nvfp4_scaled_pair_bf16(word & 255u, sc),
                       common::nvfp4_scaled_pair_bf16((word >> 8) & 255u, sc),
                       common::nvfp4_scaled_pair_bf16((word >> 16) & 255u, sc),
                       common::nvfp4_scaled_pair_bf16(word >> 24, sc));
    }
    __device__ static __forceinline__ uint4 load_v8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto code  = kv::Nvfp4G16Layout::code_offset(record, head, d, row);
        const auto scale = kv::Nvfp4G16Layout::scale_offset(record, head, d / kv::kNvfp4Group, row);
        const int4 value = common::nvfp4_dequant_f16x8(
            static_cast<const std::uint8_t*>(c.values) + code,
            static_cast<const std::uint8_t*>(c.value_scales)[scale]);
        return make_uint4(value.x, value.y, value.z, value.w);
    }
};

// ---------------------------------------------------------------------------------------------
// K8V4: K is FP8 G128 (expanded to bf16), V is NVFP4 G16. K and V rotated.
// ---------------------------------------------------------------------------------------------
template <>
struct KvPolicy<KvDType::K8V4> {
    static constexpr bool kRotate        = true;
    static constexpr bool kNativeFp8     = false;
    static constexpr bool kInverseRotate = true;

    __device__ static __forceinline__ uint4 load_k8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        const auto code  = kv::Fp8G128Layout::code_offset(record, head, d, row);
        const auto scale = kv::Fp8G128Layout::scale_offset(record, head, d / 128, row);
        const std::uint8_t sc = static_cast<const std::uint8_t*>(c.key_scales)[scale];
        const int2 raw        = common::load_vec<int2>(static_cast<const std::uint8_t*>(c.keys) + code);
        const unsigned lo     = static_cast<unsigned>(raw.x), hi = static_cast<unsigned>(raw.y);
        return pack_u4(common::fp8_dequant_bf16x2(static_cast<std::uint16_t>(lo & 0xffffu), sc),
                       common::fp8_dequant_bf16x2(static_cast<std::uint16_t>(lo >> 16), sc),
                       common::fp8_dequant_bf16x2(static_cast<std::uint16_t>(hi & 0xffffu), sc),
                       common::fp8_dequant_bf16x2(static_cast<std::uint16_t>(hi >> 16), sc));
    }
    __device__ static __forceinline__ uint4 load_v8(const Sm100KvCache& c, std::int64_t record,
                                                    int head, int row, int d) {
        return KvPolicy<KvDType::Nvfp4>::load_v8(c, record, head, row, d);
    }
};

} // namespace ninfer::ops::detail::sm100_fmha
