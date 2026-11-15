#include "ops/softmax_attention/qsa/sparse_gqa/plan.h"

#include <stdexcept>

namespace ninfer::ops::detail::qsa {
namespace {
// The sparse GQA kernels are not part of the B200 build; every entry point below is the target
// side of the shared sparse_gqa dispatch and reports the family as unavailable.
[[noreturn]] void unsupported() {
    throw std::invalid_argument("qsa: sparse_gqa is not available on b200");
}
} // namespace

int sparse_gqa_direct_queries(int, int) { unsupported(); }

SparseGqaPlan select_sparse_gqa_plan(KvCacheStorage, int, int, int,
                                     QsaSparseGqaExecutionEnvelope, int) {
    unsupported();
}

SparseGqaLaunchPlan select_sparse_gqa_launch(const ConstPagedKVBatchLayerView&, SparseGqaPlan,
                                             int, int, int) {
    unsupported();
}

void launch_sparse_gqa_instance(SparseGqaArguments, KvCacheStorage, int, SparseGqaPlan,
                                SparseGqaLaunchPlan, int, CudaExecutionView) {
    unsupported();
}

} // namespace ninfer::ops::detail::qsa
