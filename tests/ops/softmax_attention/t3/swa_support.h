// T3 / F3 (SWA): causal sliding-window attention (D256/Hq8/Hkv2, window 512, q-k <= 511) and the
// cyclic ring append (kv_cache_append_prefix), checked against a naive FP64 mask reference.
//
// Each scenario drives the public "attend, then append" flow chunk by chunk. The reference never
// looks at the ring: it keeps the complete logical K/V history of each batch row and admits key j
// for query position p exactly when 0 <= p-j <= 511. Values are large and keys nearly equal so
// the softmax is almost uniform and one mis-admitted or dropped key moves the output by far more
// than the BF16/FP16 arithmetic error.
#pragma once

#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/sliding_window_attention.h"

#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "ops/op_tester.h"
#include "ops/softmax_attention/oracle.h"
#include "ops/state_fixture.h"
#include "ops/suite.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {
constexpr int D = 256, Hq = 8, Hkv = 2, Window = 512, Padded = 520, Lanes = 16;
constexpr float Scale = 0.0625f;
constexpr ops::AttentionHeadGeometry Geometry{D, Hq, Hkv};
constexpr ReductionCriterion Criterion{2.3e-3, 3e-4, 4.5e-3};

std::vector<std::uint16_t> bf16_bits(const std::vector<float>& v) {
    std::vector<std::uint16_t> bits(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) bits[i] = f32_to_bf16(v[i]);
    return bits;
}

std::uint16_t fp16_bits_of(float x) {
    const __half h = __float2half_rn(x);
    std::uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

float fp16_round(float x) { return __half2float(__float2half_rn(x)); }

// Ring storage of 16 records, K and V packed into one state image per record, each field
// [D, Padded, Hkv]. Every slot starts as finite junk so that reading a slot that does not hold a
// committed position changes the result. `sync()` copies the packed records back into the dense
// device buffers `device_k` / `device_v`.
struct Ring {
    std::vector<std::uint16_t> k_init, v_init;
    GuardedDeviceBuffer device_k, device_v;
    std::optional<tool::StateRecordFixture> packed;

    static std::size_t index(int lane, int head, int position, int d) {
        return (((static_cast<std::size_t>(lane) * Hkv + head) * Padded + (position % Window)) *
                D) +
               d;
    }

    Ring()
        : device_k(static_cast<std::size_t>(D) * Padded * Hkv * Lanes * 2),
          device_v(device_k.bytes()) {
        std::vector<float> k(device_k.bytes() / 2), v(k.size());
        fill_uniform(k, 9001U, -0.1f, 0.1f);
        fill_uniform(v, 9002U, -12.0f, 12.0f);
        round_to_bf16(k);
        round_to_bf16(v);
        k_init = bf16_bits(k);
        v_init.resize(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) v_init[i] = fp16_bits_of(v[i]);
        device_k.copy_from_host(k_init.data(), k_init.size() * 2);
        device_v.copy_from_host(v_init.data(), v_init.size() * 2);
        const std::array<Tensor, 2> dense{
            Tensor(device_k.span(), DType::BF16, {D, Padded, Hkv, Lanes}),
            Tensor(device_v.span(), DType::FP16, {D, Padded, Hkv, Lanes})};
        packed.emplace(dense, 3);
    }

    CyclicKVCacheLayerView view(const Tensor& offsets) {
        return {.k               = packed->field(0, offsets),
                .v               = packed->field(1, offsets),
                .capacity        = Window,
                .padded_capacity = Padded,
                .num_kv_heads    = Hkv,
                .head_dim        = D};
    }

    void sync() const {
        cuda_synchronize();
        packed->copy_back();
    }
};

struct Step {
    int width;
    std::vector<int> valid;
};

// Logical history of one batch row: BF16 K and BF16 V (V is also kept after the FP16 rounding a
// stored or consumed value row undergoes).
struct History {
    std::vector<float> k, v16;
    std::vector<std::uint16_t> k_bits, v_bf16_bits;
};

History make_history(int row, int positions) {
    History h;
    std::vector<float> k(static_cast<std::size_t>(positions) * Hkv * D), v(k.size());
    fill_uniform(k, 31U + row, -0.1f, 0.1f);
    fill_uniform(v, 77U + row, -12.0f, 12.0f);
    round_to_bf16(k);
    round_to_bf16(v);
    h.k_bits      = bf16_bits(k);
    h.v_bf16_bits = bf16_bits(v);
    h.k           = k;
    h.v16         = v;
    for (auto& x : h.v16) x = fp16_round(x);
    return h;
}

} // namespace
