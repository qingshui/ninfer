#pragma once

#include "models/spark2_5/execution/resident.h"
#include "models/spark2_5/state/kv_cache.h"

#include "ninfer/ops/projection_weights.h"

#include <array>
#include <vector>

namespace ninfer::models::spark2_5 {

using LinearParameters = ops::SingleProjectionWeight;

struct AttentionParameters {
    // Rows q | k | v of the adjacent source parameters.
    ops::ProjectionWeights qkv;
    // Headwise gate logits, [heads, hidden].
    LinearParameters gate;
    LinearParameters output;
};

struct MlpParameters {
    // Rows gate | up, so the product kernel finds a matching pair M rows apart.
    LinearParameters gate_up;
    LinearParameters down;
};

struct BlockParameters {
    Tensor input_norm, post_attention_norm;
    AttentionParameters attention;
    MlpParameters mlp;
};

// Cold native preparation of the fixed model. All weight addresses borrow the ModelShard, which
// outlives this object. Runtime shape choices and scratch stay with the calling Op.
class Parameters {
public:
    explicit Parameters(const ModelShard& source);
    Parameters(const Parameters&)            = delete;
    Parameters& operator=(const Parameters&) = delete;

    const ModelShard& model;
    const TextConfig& config;
    Weight token_embedding;
    // Shares the embedding payload when the checkpoint ties them; its Use sets the policy.
    LinearParameters output_head;
    Tensor final_norm;
    std::vector<BlockParameters> layers;

    // Rows of the fused q|k|v projection, in the order of the output tensors.
    [[nodiscard]] std::array<std::int32_t, 3> qkv_rows() const noexcept;
};

} // namespace ninfer::models::spark2_5
