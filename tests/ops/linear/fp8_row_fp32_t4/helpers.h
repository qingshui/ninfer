// Sampling and fixtures for the T4 / F2 Linear test.
#pragma once

#include "reps.h"

#include <cstdint>
#include <vector>

namespace ninfer::test::t4 {

struct Shape {
    const char* name;
    std::int32_t n;
    std::int32_t k;
};

inline constexpr Shape kShapes[] = {
    {"qkv", 3072, 2048},      {"g", 8, 2048},       {"out", 2048, 2048},
    {"gate_up", 13312, 2048}, {"down", 2048, 6656}, {"head", 131072, 2048},
};
inline constexpr std::int32_t kPrefillTokens = 512;
inline constexpr std::int32_t kMaxTokens     = kPrefillTokens;

inline std::vector<std::int32_t> sample_rows(std::int32_t n) {
    std::vector<std::int32_t> rows;
    if (n <= 64) {
        for (std::int32_t r = 0; r < n; ++r) rows.push_back(r);
        return rows;
    }
    for (std::int32_t i = 0; i <= 60; ++i) {
        rows.push_back(static_cast<std::int32_t>(static_cast<std::int64_t>(n - 1) * i / 60));
    }
    return rows;
}

inline std::vector<std::int32_t> sample_tokens(std::int32_t t) {
    std::vector<std::int32_t> tokens;
    if (t <= 16) {
        for (std::int32_t i = 0; i < t; ++i) tokens.push_back(i);
        return tokens;
    }
    for (std::int32_t i = 0; i < 16; ++i) tokens.push_back((t - 1) * i / 15);
    return tokens;
}

// Activation [T][K] BF16 bits (token-major, K contiguous), values (raw-128)/256.
inline std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t tokens) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * tokens);
    for (std::int32_t t = 0; t < tokens; ++t) {
        for (std::int32_t c = 0; c < k; ++c) {
            const std::uint32_t h = mix(static_cast<std::uint32_t>(c) * 0x9e3779b9U ^
                                        static_cast<std::uint32_t>(t) * 0x85ebca6bU ^ 0x5a17U);
            bits[static_cast<std::size_t>(t) * k + c] =
                f32_to_bf16(static_cast<float>(static_cast<int>(h & 0xffU) - 128) / 256.0F);
        }
    }
    return bits;
}

inline bool bf16_is_finite(std::uint16_t bits) { return (bits & 0x7f80U) != 0x7f80U; }

} // namespace ninfer::test::t4
