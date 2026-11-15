#pragma once

// UMMA operand descriptors for tcgen05.mma: the 64-bit shared-memory matrix descriptor and the
// 32-bit instruction descriptor, built from named fields instead of literals.

#include <cstdint>

namespace ninfer::ops::common::sm100 {

// Shared-memory swizzle encoding of descriptor bits [61,64).
enum class UmmaSwizzle : std::uint32_t {
    None       = 0,
    B128Base32 = 1,
    B128       = 2,
    B64        = 4,
    B32        = 6,
};

enum class UmmaMajor : std::uint32_t { K = 0, MN = 1 };

// Bytes of one swizzle atom row. A K-major operand tile is a stack of 8-row atoms, each
// 8 x swizzle-span bytes, and the descriptor strides between atoms by 8 x span.
__host__ __device__ constexpr int swizzle_span_bytes(UmmaSwizzle swizzle) {
    switch (swizzle) {
    case UmmaSwizzle::B128:
    case UmmaSwizzle::B128Base32:
        return 128;
    case UmmaSwizzle::B64:
        return 64;
    case UmmaSwizzle::B32:
        return 32;
    default:
        return 16;
    }
}

// Shared-memory matrix descriptor.
//   [0,14)   start address >> 4
//   [16,30)  leading-dimension byte offset >> 4 (unused by swizzled K-major layouts; set to 1)
//   [32,46)  stride-dimension byte offset >> 4  (distance between 8-row atoms)
//   [46,48)  descriptor version, 1 on SM100
//   [49,52)  base offset (0 when the tile base is aligned to the swizzle repeat)
//   [52]     leading-dimension mode (0 = relative offset)
//   [61,64)  swizzle mode
__host__ __device__ constexpr std::uint64_t make_umma_smem_desc(std::uint32_t smem_address,
                                                                std::uint32_t leading_byte_offset,
                                                                std::uint32_t stride_byte_offset,
                                                                UmmaSwizzle swizzle) {
    return (static_cast<std::uint64_t>((smem_address & 0x3FFFFu) >> 4)) |
           (static_cast<std::uint64_t>((leading_byte_offset & 0x3FFFFu) >> 4) << 16) |
           (static_cast<std::uint64_t>((stride_byte_offset & 0x3FFFFu) >> 4) << 32) |
           (static_cast<std::uint64_t>(1) << 46) |
           (static_cast<std::uint64_t>(swizzle) << 61);
}

// K-major operand tile whose rows are one swizzle span long (e.g. 128 B rows with 128B swizzle).
__host__ __device__ constexpr std::uint64_t make_umma_k_major_desc(std::uint32_t smem_address,
                                                                   UmmaSwizzle swizzle) {
    return make_umma_smem_desc(smem_address, 16, 8u * swizzle_span_bytes(swizzle), swizzle);
}

// Advances the start-address field. `bytes` must be a multiple of 16.
__host__ __device__ constexpr std::uint64_t umma_desc_advance(std::uint64_t desc,
                                                              std::uint32_t bytes) {
    return desc + (bytes >> 4);
}

// Plain descriptor for a block-scale atom block in shared memory: contiguous 512-byte atoms read
// by tcgen05.cp.32x128b.warpx4, no swizzle.
__host__ __device__ constexpr std::uint64_t make_umma_scale_desc(std::uint32_t smem_address) {
    return make_umma_smem_desc(smem_address, 16, 128, UmmaSwizzle::None);
}

// Source element formats of the instruction descriptor.
enum class UmmaF16Format : std::uint32_t { F16 = 0, BF16 = 1, TF32 = 2 };
enum class UmmaF8F6F4Format : std::uint32_t { E4M3 = 0, E5M2 = 1, E2M3 = 3, E3M2 = 4, E2M1 = 5 };
enum class UmmaF4Format : std::uint32_t { E2M1 = 1 };
enum class UmmaScaleFormat : std::uint32_t { UE4M3 = 0, UE8M0 = 1 };

// Instruction descriptor, kind::f16. Accumulator is FP32.
//   [4,6) C format (1 = F32)   [7,10) A format   [10,13) B format
//   [15] A major   [16] B major   [17,23) N >> 3   [24,29) M >> 4
__host__ __device__ constexpr std::uint32_t make_umma_idesc_f16(int m, int n, UmmaF16Format a,
                                                                UmmaF16Format b,
                                                                UmmaMajor a_major = UmmaMajor::K,
                                                                UmmaMajor b_major = UmmaMajor::K) {
    return (1u << 4) | (static_cast<std::uint32_t>(a) << 7) |
           (static_cast<std::uint32_t>(b) << 10) | (static_cast<std::uint32_t>(a_major) << 15) |
           (static_cast<std::uint32_t>(b_major) << 16) | (static_cast<std::uint32_t>(n >> 3) << 17) |
           (static_cast<std::uint32_t>(m >> 4) << 24);
}

// Instruction descriptor, kind::f8f6f4 (unscaled).
__host__ __device__ constexpr std::uint32_t
make_umma_idesc_f8f6f4(int m, int n, UmmaF8F6F4Format a, UmmaF8F6F4Format b,
                       UmmaMajor a_major = UmmaMajor::K, UmmaMajor b_major = UmmaMajor::K) {
    return (1u << 4) | (static_cast<std::uint32_t>(a) << 7) |
           (static_cast<std::uint32_t>(b) << 10) | (static_cast<std::uint32_t>(a_major) << 15) |
           (static_cast<std::uint32_t>(b_major) << 16) | (static_cast<std::uint32_t>(n >> 3) << 17) |
           (static_cast<std::uint32_t>(m >> 4) << 24);
}

// Instruction descriptor, kind::mxf4nvf4.block_scale (K-major operands, FP32 accumulator).
//   [4,6) B scale-factor id   [7,10) A format   [10,13) B format   [17,23) N >> 3
//   [23] scale format (0 = UE4M3)   [24,29) M >> 4   [29,31) A scale-factor id   [31] K size
__host__ __device__ constexpr std::uint32_t
make_umma_idesc_mxf4nvf4(int m, int n, UmmaScaleFormat scale = UmmaScaleFormat::UE4M3) {
    return (static_cast<std::uint32_t>(UmmaF4Format::E2M1) << 7) |
           (static_cast<std::uint32_t>(UmmaF4Format::E2M1) << 10) |
           (static_cast<std::uint32_t>(n >> 3) << 17) | (static_cast<std::uint32_t>(scale) << 23) |
           (static_cast<std::uint32_t>(m >> 4) << 24);
}

} // namespace ninfer::ops::common::sm100
