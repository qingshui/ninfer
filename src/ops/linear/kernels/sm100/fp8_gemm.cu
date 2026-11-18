#include "ops/linear/kernels/sm100/gemm.h"
#include "ops/linear/kernels/sm100/gemm_launch.cuh"

namespace ninfer::ops::detail {
namespace {
namespace g  = sm100_gemm;
namespace sm = ::ninfer::ops::common::sm100;

template <int TileN>
void launch_tile(const Sm100Fp8Operands& p, const g::PlainEpilogue& epilogue,
                 cudaStream_t stream) {
    using Cfg = sm::GemmConfig<sm::GemmKind::Fp8, TileN>;
    // A box inner extent of 128 E4M3 codes is one 128B-swizzle atom row.
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(p.k);
    const CUtensorMap a_map       = sm::make_tensor_map_2d(
        p.weight, CU_TENSOR_MAP_DATA_TYPE_UINT8, p.k, p.rows, row_bytes, Cfg::kBoxInner,
        Cfg::kTileM, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 fp8 gemm: weight tensor map");
    const CUtensorMap b_map = sm::make_tensor_map_2d(
        p.x, CU_TENSOR_MAP_DATA_TYPE_UINT8, p.k, p.tokens, row_bytes, Cfg::kBoxInner, TileN,
        CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
        "sm100 fp8 gemm: activation tensor map");
    sm::GemmArgs args{};
    args.rows    = p.rows;
    args.tokens  = p.tokens;
    args.k_iters = p.k / Cfg::kBoxInner;
    args.tiles_m = p.rows / Cfg::kTileM;
    args.tiles_n = (p.tokens + TileN - 1) / TileN;
    g::launch_kernel<Cfg>(a_map, b_map, args, epilogue, stream);
}
} // namespace

Nvfp4Sm100Tuning fp8_sm100_tuning(int rows, int, int tokens) {
    if (tokens <= 16) return {16, 1, 1};
    if (tokens <= 32) return {32, 1, 1};
    if (tokens <= 64) return {64, 1, 1};
    if (tokens >= 3072 && rows >= 1024) return {256, 1, 1};
    return {128, 1, 1};
}

void launch_sm100_fp8_gemm(const Sm100Fp8Operands& p, const Sm100Epilogue& epilogue,
                           __nv_bfloat16* y, cudaStream_t stream) {
    if (p.rows <= 0 || p.rows % 128 != 0 || p.k <= 0 || p.k % 128 != 0 || p.tokens <= 0)
        g::reject("FP8 needs whole 128-row tiles, K a multiple of 128 and tokens > 0");
    if (epilogue.kind == Sm100Epilogue::SwiGLU) g::reject("FP8 does not offer the SwiGLU epilogue");
    if (!g::aligned(p.x, 16) || !g::aligned(p.weight, 16))
        g::reject("FP8 operands must be 16-byte aligned");
    if (!g::aligned(p.x_scales, 4) || !g::aligned(p.weight_scales, 4))
        g::reject("FP8 scales must be 4-byte aligned");

    g::PlainEpilogue e = g::make_plain_epilogue(epilogue, y, 1.0f, p.rows, p.tokens);
    e.row_scales       = p.weight_scales;
    e.token_scales     = p.x_scales;
    switch (fp8_sm100_tuning(p.rows, p.k, p.tokens).tile_tokens) {
    case 16:
        return launch_tile<16>(p, e, stream);
    case 32:
        return launch_tile<32>(p, e, stream);
    case 64:
        return launch_tile<64>(p, e, stream);
    case 128:
        return launch_tile<128>(p, e, stream);
    case 256:
        return launch_tile<256>(p, e, stream);
    }
    g::reject("unsupported token tile");
}

} // namespace ninfer::ops::detail
