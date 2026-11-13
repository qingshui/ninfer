#include "models/spark2_5/program/buffers.h"

#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::spark2_5 {

BufferLayout plan_buffers(LayoutBuilder& b, const TextConfig& c, const EngineOptions& o,
                          std::uint32_t chunk, std::uint32_t public_tokens) {
    const auto batch     = static_cast<std::int32_t>(o.max_concurrency);
    const auto positions = std::max(static_cast<std::int32_t>(chunk), batch);
    const auto tensor    = [&](DType dtype, std::initializer_list<std::int32_t> shape,
                            const char* label) { return b.add_tensor(dtype, shape, 256, label); };
    const auto rows      = [&](const char* label) { return tensor(DType::I32, {batch}, label); };
    const bool scoring   = o.purpose == EnginePurpose::CausalScoring;
    BufferLayout out;
    out.max_positions        = positions;
    out.ids                  = tensor(DType::I32, {positions}, "token ids");
    out.positions            = tensor(DType::I32, {positions}, "token ordinals");
    out.valid_columns        = rows("valid columns");
    out.kv_rows              = rows("full KV table rows");
    out.destination_offsets  = tensor(DType::I64, {batch}, "destination State offsets");
    out.sampling_positions   = rows("sampling positions");
    out.count_slots          = rows("token count slots");
    out.count_columns        = rows("token count columns");
    TensorRegion last        = out.count_columns;
    if (!scoring) {
        out.sampling =
            tensor(DType::I32, {static_cast<int>(sizeof(ops::SamplingConfig) / 4), batch},
                   "sampling config");
        last = *out.sampling;
    }
    out.ingress = {out.ids.region.offset,
                   last.region.offset + last.region.bytes - out.ids.region.offset, 256};
    out.hidden =
        tensor(DType::BF16, {dimension(c.hidden), positions}, "final-norm hidden of a traversal");
    if (!scoring) {
        out.logits = tensor(DType::BF16, {dimension(c.vocabulary), batch}, "head logits");
        out.sampled_ids = tensor(DType::I32, {batch}, "generated IDs");
        out.token_counts =
            tensor(DType::I32, {static_cast<std::int32_t>(public_tokens), batch}, "token counts");
        out.grammar_masks = tensor(
            DType::I32, {static_cast<std::int32_t>((public_tokens + 31) / 32), batch}, "grammar masks");
    }
    return out;
}

} // namespace ninfer::models::spark2_5
