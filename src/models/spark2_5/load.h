#pragma once

#include "models/common/bindings.h"
#include "models/spark2_5/config.h"
#include "models/spark2_5/weights.h"

namespace ninfer::models::spark2_5::loading {

// Declares the text parameters and their Uses against the artifact. A tied output head must
// reuse the embedding's stored payload.
[[nodiscard]] TextWeights bind_text(Bindings&, const TextConfig&);

// Throws when `parameter` does not bind exactly the object ranges of `target`.
void require_shared(const artifact::ParameterReference& parameter,
                    const artifact::ParameterReference& target);

} // namespace ninfer::models::spark2_5::loading
