#pragma once

#include "models/common/weights.h"

#include <vector>

namespace ninfer::models::spark2_5 {

// Handles name logical parameters of the frozen definition. Execution merges query/key/value
// into one qkv projection and gate/up into gate_up from their adjacent parent regions.
struct AttentionWeights {
    WeightId query, key, value;
    WeightId gate, output;
};

struct MlpWeights {
    WeightId gate, up, down;
};

struct LayerWeights {
    WeightId input_norm, post_attention_norm;
    AttentionWeights attention;
    MlpWeights mlp;
};

// With tied embeddings, output_head and token_embedding name the same stored payload.
struct TextWeights {
    WeightId token_embedding, output_head, final_norm;
    std::vector<LayerWeights> layers;
};

struct ModelWeights {
    TextWeights text;
};

} // namespace ninfer::models::spark2_5
