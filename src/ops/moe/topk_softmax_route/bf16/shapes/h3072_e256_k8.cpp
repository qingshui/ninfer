#include "ops/moe/topk_softmax_route/dispatch.h"

namespace ninfer::ops::detail::moe_route {
RouterPlan select_h3072_e256_k8(int tokens) {
    return {tokens == 1 ? RouterAlgorithm::Vector : RouterAlgorithm::Mma};
}
} // namespace ninfer::ops::detail::moe_route
