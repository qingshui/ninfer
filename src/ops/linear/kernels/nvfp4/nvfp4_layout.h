#pragma once
#include "ops/linear/kernels/nvfp4/nvfp4_geometry.h"

namespace ninfer::ops::detail {
inline constexpr int kNvfp4ScaleTileGroups = 16;
// K16M128x4 is the tcgen05 block-scale atom layout shared with the weight format
// block_scale_k16_m128x4_v1: per 128 tokens and 64 K values (four 16-wide groups) one 512-byte
// atom, indexed (token % 32) * 16 + ((token % 128) / 32) * 4 + group % 4, with the atoms of a
// 128-token tile ordered by K. Tiles of 128 tokens follow each other, so a token tile's scales
// are one contiguous (K / 64) * 512-byte run.
enum class Nvfp4ScaleLayout : std::uint8_t { RowMajor, Tiled128, Tiled256, K16M128x4 };
template <Nvfp4ScaleLayout Layout>
inline constexpr int kNvfp4ScaleTileTokens = Layout == Nvfp4ScaleLayout::Tiled128     ? 128
                                             : Layout == Nvfp4ScaleLayout::Tiled256   ? 256
                                             : Layout == Nvfp4ScaleLayout::K16M128x4 ? 128
                                                                                      : 1;

inline constexpr int nvfp4_scale_tile_tokens(Nvfp4ScaleLayout layout) {
    return layout == Nvfp4ScaleLayout::Tiled128      ? 128
           : layout == Nvfp4ScaleLayout::Tiled256    ? 256
           : layout == Nvfp4ScaleLayout::K16M128x4   ? 128
                                                     : 1;
}

inline constexpr int nvfp4_a4_padded_tokens(int tokens,
                                            Nvfp4ScaleLayout layout = Nvfp4ScaleLayout::Tiled256) {
    const int tile = nvfp4_scale_tile_tokens(layout);
    return ((tokens + tile - 1) / tile) * tile;
}
} // namespace ninfer::ops::detail
