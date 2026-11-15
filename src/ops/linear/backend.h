#pragma once

// Which execution engine serves a dense linear call. The decision belongs to the linear Op:
// callers and shape files ask here instead of keeping their own list of tcgen05 shapes.

#include "core/platform.h"
#include "core/weight.h"

namespace ninfer::ops::detail {

enum class LinearBackend { Tcgen05, MmaSync, Gemv };

// Tcgen05 serves the Qwen3.5-122B dense projections on B200 at every token count:
//   NVFP4 (n=3072, k=8192) full-attention o_proj and (n=17408, k=3072) full-attention qkvz;
//   BF16  (n=20480, k=3072) GDN qkvz and (n=3072, k=8192) GDN out.
// Everything else, and every other target, runs mma.sync (or the single-token GEMV).
// A16-only NVFP4 calls never quantize their activations, so they stay off tcgen05; the shape
// file applies that policy.
constexpr LinearBackend linear_backend(QType type, int n, int k, int tokens) {
#if NINFER_TARGET_B200
    if (tokens > 0) {
        if (type == QType::NVFP4 && ((n == 3072 && k == 8192) || (n == 17408 && k == 3072)))
            return LinearBackend::Tcgen05;
        if (type == QType::BF16 && ((n == 20480 && k == 3072) || (n == 3072 && k == 8192)))
            return LinearBackend::Tcgen05;
    }
#else
    (void)type;
    (void)n;
    (void)k;
#endif
    return tokens == 1 ? LinearBackend::Gemv : LinearBackend::MmaSync;
}

} // namespace ninfer::ops::detail
