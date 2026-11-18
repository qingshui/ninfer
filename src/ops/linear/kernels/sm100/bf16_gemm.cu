#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/kernels/sm100/gemm_launch.cuh"

namespace ninfer::ops::detail {
namespace {
namespace g  = sm100_gemm;
namespace sm = ::ninfer::ops::common::sm100;

template <int TileN, int Parts, class Epilogue>
void launch_tile(const Sm100Bf16Operands& p, const Epilogue& epilogue, int tiles_m,
                 cudaStream_t stream) {
    using Cfg = sm::GemmConfig<sm::GemmKind::Bf16, TileN, Parts>;
    // Box inner extent of 64 BF16 values is one 128B-swizzle atom row.
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(p.k) * 2;
    const CUtensorMap a_map       = sm::make_tensor_map_2d(
        p.weight, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.k, p.rows, row_bytes, Cfg::kBoxInner,
        Cfg::kTileM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 bf16 gemm: weight tensor map");
    const CUtensorMap b_map = sm::make_tensor_map_2d(
        p.x, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, p.k, p.tokens, row_bytes, Cfg::kBoxInner, TileN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 bf16 gemm: activation tensor map");
    sm::GemmArgs args{};
    args.rows            = p.rows;
    args.tokens          = p.tokens;
    args.k_iters         = p.k / 64;
    args.tiles_m         = tiles_m;
    args.tiles_n         = (p.tokens + TileN - 1) / TileN;
    args.part_row_offset = Parts == 2 ? p.rows / 2 : 0;
    g::launch_kernel<Cfg>(a_map, b_map, args, epilogue, stream);
}

template <class Epilogue, int Parts>
void dispatch_tile(int tile_tokens, const Sm100Bf16Operands& p, const Epilogue& epilogue,
                   int tiles_m, cudaStream_t stream) {
    switch (tile_tokens) {
    case 16:
        return launch_tile<16, Parts>(p, epilogue, tiles_m, stream);
    case 32:
        return launch_tile<32, Parts>(p, epilogue, tiles_m, stream);
    case 64:
        return launch_tile<64, Parts>(p, epilogue, tiles_m, stream);
    case 128:
        return launch_tile<128, Parts>(p, epilogue, tiles_m, stream);
    case 256:
        return launch_tile<256, Parts>(p, epilogue, tiles_m, stream);
    }
    g::reject("unsupported token tile");
}
} // namespace

Nvfp4Sm100Tuning bf16_sm100_tuning(int rows, int, int tokens) {
    if (tokens <= 16) return {16, 1, 1};
    if (tokens <= 32) return {32, 1, 1};
    if (tokens <= 64) return {64, 1, 1};
    // 256-token tiles cut operand traffic per MAC; they need enough tiles to fill the SMs.
    if (tokens >= 3072 && rows >= 1024) return {256, 1, 1};
    return {128, 1, 1};
}

void launch_sm100_bf16_gemm(const Sm100Bf16Operands& p, const Sm100Epilogue& epilogue,
                            __nv_bfloat16* y, cudaStream_t stream) {
    if (p.rows <= 0 || p.rows % 128 != 0 || p.k <= 0 || p.k % 64 != 0 || p.tokens <= 0)
        g::reject("BF16 needs whole 128-row tiles, K a multiple of 64 and tokens > 0");
    if (!g::aligned(p.x, 16) || !g::aligned(p.weight, 16))
        g::reject("BF16 operands must be 16-byte aligned");

    const Nvfp4Sm100Tuning tuning = bf16_sm100_tuning(p.rows, p.k, p.tokens);
    if (tuning.tile_tokens < 128 && p.tokens > tuning.tile_tokens)
        g::reject("tile narrower than 128 tokens must cover the whole call");

    if (epilogue.kind == Sm100Epilogue::SwiGLU) {
        if (p.rows % 256 != 0) g::reject("SwiGLU needs gate and up halves of whole 128-row tiles");
        if (!g::aligned(y, 16)) g::reject("output must be 16-byte aligned");
        const g::SwiGluEpilogue e{1.0f, y, p.rows / 2, p.tokens};
        dispatch_tile<g::SwiGluEpilogue, 2>(tuning.tile_tokens == 256 ? 128 : tuning.tile_tokens,
                                            p, e, p.rows / 256, stream);
        return;
    }
    const g::PlainEpilogue e = g::make_plain_epilogue(epilogue, y, 1.0f, p.rows, p.tokens);
    dispatch_tile<g::PlainEpilogue, 1>(tuning.tile_tokens, p, e, p.rows / 128, stream);
}

} // namespace ninfer::ops::detail
