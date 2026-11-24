#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/kernels/sm100/gemm_launch.cuh"

#include <type_traits>

namespace ninfer::ops::detail {
namespace {
namespace g  = sm100_gemm;
namespace sm = ::ninfer::ops::common::sm100;

// One tile of `TileN` tokens against 128 weight rows (two such weight tiles for SwiGLU).
template <int TileN, int Parts, class Epilogue>
void launch_tile(const Nvfp4A4Operands& p, const Epilogue& epilogue, int tiles_m,
                 const Sm100GemmPlan& plan, float* partials, cudaStream_t stream) {
    using Cfg = sm::GemmConfig<sm::GemmKind::Nvfp4, TileN, Parts>;
    // Weight codes: rows of k/2 bytes. Activation codes: tokens of k/2 bytes. A 128-byte box
    // inner extent is one 128B-swizzle atom row = 256 E2M1 values.
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(p.k) / 2;
    const CUtensorMap a_map       = sm::make_tensor_map_2d(
        p.codes, CU_TENSOR_MAP_DATA_TYPE_UINT8, row_bytes, p.rows, row_bytes, Cfg::kBoxInner,
        Cfg::kTileM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 nvfp4 gemm: weight codes tensor map");
    const CUtensorMap b_map = sm::make_tensor_map_2d(
        p.x, CU_TENSOR_MAP_DATA_TYPE_UINT8, row_bytes, p.tokens, row_bytes, Cfg::kBoxInner,
        TileN, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 nvfp4 gemm: activation codes tensor map");
    sm::GemmArgs args{};
    args.rows            = p.rows;
    args.tokens          = p.tokens;
    args.k_iters         = p.k / 256;
    args.tiles_m         = tiles_m;
    args.tiles_n         = (p.tokens + TileN - 1) / TileN;
    args.part_row_offset = Parts == 2 ? p.rows / 2 : 0;
    args.scale_k64       = p.k / 64;
    args.a_scales        = p.scales;
    args.b_scales        = p.x_scales;
    if constexpr (std::is_same_v<Epilogue, g::PlainEpilogue>)
        g::launch_plain<Cfg>(a_map, b_map, args, epilogue, plan, partials, stream);
    else
        g::launch_kernel<Cfg>(a_map, b_map, args, epilogue, stream);
}

template <class Epilogue, int Parts>
void dispatch_tile(const Sm100GemmPlan& plan, const Nvfp4A4Operands& p, const Epilogue& epilogue,
                   int tiles_m, float* partials, cudaStream_t stream) {
    switch (plan.tile_tokens) {
    case 16:
        return launch_tile<16, Parts>(p, epilogue, tiles_m, plan, partials, stream);
    case 32:
        return launch_tile<32, Parts>(p, epilogue, tiles_m, plan, partials, stream);
    case 64:
        return launch_tile<64, Parts>(p, epilogue, tiles_m, plan, partials, stream);
    case 128:
        return launch_tile<128, Parts>(p, epilogue, tiles_m, plan, partials, stream);
    }
    g::reject("unsupported token tile");
}
} // namespace

Sm100GemmPlan nvfp4_sm100_plan(int rows, int k, int tokens, std::size_t partial_bytes) {
    // A tile narrower than 128 tokens addresses a single scale atom, so it only serves calls that
    // fit one tile. Wider calls use whole 128-token tiles.
    const int tile = tokens <= 16 ? 16 : tokens <= 32 ? 32 : tokens <= 64 ? 64 : 128;
    return g::plan_gemm(rows / 128, tokens, k / 256, 1, partial_bytes, tile);
}

void launch_sm100_nvfp4_gemm(const Nvfp4A4Operands& p, const Sm100Epilogue& epilogue,
                             __nv_bfloat16* y, cudaStream_t stream, Sm100Partials partials) {
    if (p.rows <= 0 || p.rows % 128 != 0 || p.k <= 0 || p.k % 256 != 0 || p.tokens <= 0)
        g::reject("NVFP4 needs whole 128-row tiles, K a multiple of 256 and tokens > 0");
    if (p.scale_layout != Nvfp4ScaleLayout::K16M128x4)
        g::reject("NVFP4 activations must use the K16M128x4 scale layout");
    if (!g::aligned(p.x, 128) || !g::aligned(p.codes, 128) || !g::aligned(p.x_scales, 128) ||
        !g::aligned(p.scales, 128))
        g::reject("NVFP4 operands must be 128-byte aligned");
    if (!std::isfinite(p.alpha) || p.alpha <= 0.0f) g::reject("alpha must be positive and finite");
    if (partials.bytes != 0 && !g::aligned(partials.data, 16))
        g::reject("split partials must be 16-byte aligned");

    if (epilogue.kind == Sm100Epilogue::SwiGLU) {
        if (p.rows % 256 != 0) g::reject("SwiGLU needs gate and up halves of whole 128-row tiles");
        if (!g::aligned(y, 16)) g::reject("output must be 16-byte aligned");
        const g::SwiGluEpilogue e{p.alpha, y, p.rows / 2, p.tokens};
        dispatch_tile<g::SwiGluEpilogue, 2>(nvfp4_sm100_plan(p.rows, p.k, p.tokens), p, e,
                                            p.rows / 256, nullptr, stream);
        return;
    }
    const g::PlainEpilogue e = g::make_plain_epilogue(epilogue, y, p.alpha, p.rows, p.tokens);
    dispatch_tile<g::PlainEpilogue, 1>(nvfp4_sm100_plan(p.rows, p.k, p.tokens, partials.bytes), p,
                                       e, p.rows / 128, partials.data, stream);
}

} // namespace ninfer::ops::detail
