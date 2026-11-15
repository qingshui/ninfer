#pragma once

// Which execution engine serves an NVFP4 routed-expert call. The decision belongs to the MoE Op:
// shape files and kernel launchers ask here instead of keeping their own list of tcgen05
// geometries.

#include "core/platform.h"

namespace ninfer::ops::detail {

enum class MoeBackend { Tcgen05, MmaSync, Vector };

// Tcgen05 serves the Qwen3.5-122B expert geometry (hidden 3072, 256 experts, top-8) on B200 for
// every token count above one: tokens are grouped by expert and each expert's tokens run as
// swap-AB tcgen05 GEMM tiles. A single token is a weight-bandwidth GEMV on every target (Vector).
// Everything else runs mma.sync.
constexpr MoeBackend moe_backend(int hidden, int experts, int top_k, int tokens) {
    if (tokens == 1) return MoeBackend::Vector;
#if NINFER_TARGET_B200
    if (tokens > 1 && hidden == 3072 && experts == 256 && top_k == 8) return MoeBackend::Tcgen05;
#else
    (void)hidden;
    (void)experts;
    (void)top_k;
#endif
    return MoeBackend::MmaSync;
}

} // namespace ninfer::ops::detail
