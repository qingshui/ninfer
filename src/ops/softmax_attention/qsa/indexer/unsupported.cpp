#include "ops/softmax_attention/qsa/indexer/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::qsa {
namespace {
// The sparse QSA indexer kernels are not part of the B200 build; every entry point below is
// the target side of the shared indexer dispatch and reports the family as unavailable.
[[noreturn]] void unsupported() {
    throw std::invalid_argument("qsa: indexer is not available on b200");
}
} // namespace

int select_indexer_window_tokens(int, int) { unsupported(); }

IndexerWindowPlan select_indexer_window(IndexerPlan, int, int, int) { unsupported(); }

void launch_indexer_window(IndexerArguments, IndexerWindowPlan, bool, CudaExecutionView) {
    unsupported();
}

} // namespace ninfer::ops::detail::qsa
