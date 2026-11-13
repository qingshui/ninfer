// Weight representations for the T4 / F2 Linear test. Each representation owns its device
// storage, exposes the public Weight view, and decodes sampled rows with its own code, independent
// of every production decode path.
#pragma once

#include "cuda_fixture.h"

#include "core/buffer.h"
#include "core/weight.h"
#include "ops/direct_bf16_weight.h"
#include "ops/op_tester.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace ninfer::test::t4 {

inline std::uint32_t mix(std::uint32_t h) {
    h ^= h >> 16;
    h *= 0x7feb352dU;
    h ^= h >> 15;
    h *= 0x846ca68bU;
    h ^= h >> 16;
    return h;
}

inline double decode_e4m3(std::uint8_t c) {
    const int sign = c >> 7, exp = (c >> 3) & 15, man = c & 7;
    const double v = exp == 0 ? std::ldexp(man / 8.0, -6) : std::ldexp(1.0 + man / 8.0, exp - 7);
    return sign ? -v : v;
}

struct Rep {
    virtual ~Rep()                                                 = default;
    virtual Weight view() const                                    = 0;
    virtual std::vector<double> decode_row(std::int32_t row) const = 0;
    virtual int verify_preserved(const char* label) const          = 0;
};

struct Fp8RowFp32Rep final : Rep {
    std::int32_t n, k;
    std::vector<std::uint8_t> payload;
    std::size_t scale_offset;
    DeviceBuffer device;

    Fp8RowFp32Rep(std::int32_t n_, std::int32_t k_) : n(n_), k(k_) {
        const std::size_t codes = static_cast<std::size_t>(n) * k;
        scale_offset            = (codes + 255) / 256 * 256;
        payload.assign(scale_offset + static_cast<std::size_t>(n) * 4, 0);
        for (std::int32_t r = 0; r < n; ++r) {
            for (std::int32_t c = 0; c < k; ++c) {
                const std::uint32_t h = mix(static_cast<std::uint32_t>(r) * 0x9e3779b9U ^
                                            static_cast<std::uint32_t>(c) * 0x85ebca6bU ^ 0x77U);
                const std::uint32_t e = 4 + (h % 7), m = (h >> 8) & 7, s = (h >> 12) & 1;
                payload[static_cast<std::size_t>(r) * k + c] =
                    static_cast<std::uint8_t>((s << 7) | (e << 3) | m);
            }
            // Full-mantissa FP32 row scale, not representable in BF16.
            const float scale = std::ldexp(1.0F + static_cast<float>((r * 37) % 101) / 101.0F, -9);
            std::memcpy(payload.data() + scale_offset + static_cast<std::size_t>(r) * 4, &scale, 4);
        }
        device = DeviceBuffer(cuda_environment().context.device(), payload.size());
        device.copy_from_host(payload.data(), payload.size());
    }

    Weight view() const override {
        Weight w{};
        const auto* base          = static_cast<const std::uint8_t*>(device.data());
        const std::int64_t stride = static_cast<std::int64_t>(n) * 4;
        w.qtype                   = QType::FP8_E4M3FN_ROW_FP32;
        w.layout                  = QuantLayout::RowScale;
        w.scale_dtype             = DType::FP32;
        w.device                  = device.device();
        w.payload                 = base;
        w.payload_bytes           = payload.size();
        w.high_plane_bytes        = 0;
        w.qdata                   = base;
        w.qhigh                   = nullptr;
        w.scales                  = base + scale_offset;
        w.group_size              = static_cast<std::uint32_t>(k);
        w.group                   = k;
        w.ndim                    = 2;
        w.n                       = n;
        w.k                       = k;
        for (int i = 0; i < 4; ++i) {
            w.shape[i]        = i == 0 ? n : (i == 1 ? k : 1);
            w.padded_shape[i] = w.shape[i];
            w.scale_ne[i]     = i == 0 ? n : 1;
        }
        w.scale_nb[0] = 4;
        w.scale_nb[1] = w.scale_nb[2] = w.scale_nb[3] = stride;
        return w;
    }

    std::vector<double> decode_row(std::int32_t row) const override {
        float scale;
        std::memcpy(&scale, payload.data() + scale_offset + static_cast<std::size_t>(row) * 4, 4);
        std::vector<double> out(k);
        for (std::int32_t c = 0; c < k; ++c) {
            out[c] = decode_e4m3(payload[static_cast<std::size_t>(row) * k + c]) *
                     static_cast<double>(scale);
        }
        return out;
    }

    int verify_preserved(const char* label) const override {
        std::vector<std::uint8_t> after(payload.size());
        device.copy_to_host(after.data(), after.size());
        if (after == payload) return 0;
        std::cerr << label << ": FP8 weight was modified\n";
        return 1;
    }
};

struct Bf16Rep final : Rep {
    direct_bf16_weight::DeviceWeight weight;

    Bf16Rep(std::int32_t n, std::int32_t k)
        : weight(direct_bf16_weight::make_patterned(n, k, 4242U)) {}

    Weight view() const override { return weight.view(); }

    std::vector<double> decode_row(std::int32_t row) const override {
        const auto& h = weight.host;
        std::vector<double> out(h.k);
        for (std::int32_t c = 0; c < h.k; ++c) {
            out[c] = bf16_to_f32(h.bits[static_cast<std::size_t>(row) * h.k + c]);
        }
        return out;
    }

    int verify_preserved(const char* label) const override { return weight.verify_preserved(label); }
};

} // namespace ninfer::test::t4
