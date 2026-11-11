#include "ninfer/ops/sliding_window_attention.h"

#include "core/layout.h"
#include "ops/common/host/operands.h"
#include "ops/softmax_attention/causal_sliding_window/launch.h"
#include "ops/softmax_attention/kernels/causal_sliding_window/constants.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer::ops {
namespace {

constexpr std::int32_t kHeadDim = 256;
constexpr std::int32_t kQHeads  = 8;
constexpr std::int32_t kKVHeads = detail::kCausalSlidingWindowKvHeads;
constexpr float kExpectedScale  = 0.0625f;

void require_profile(AttentionHeadGeometry geometry, std::uint32_t window, const char* op) {
    if (!valid_attention_head_geometry(geometry) || geometry.head_dim != kHeadDim ||
        geometry.query_heads != kQHeads || geometry.kv_heads != kKVHeads) {
        throw std::invalid_argument(std::string(op) + ": unsupported head geometry");
    }
    if (window != static_cast<std::uint32_t>(detail::kCausalSlidingWindow)) {
        throw std::invalid_argument(std::string(op) + ": the supported window is 512");
    }
}

void validate_context(const ConstCyclicKVCacheLayerView& context, std::int32_t batch,
                      const char* op, Device device) {
    common::operand::device(context.device(), device, op, "ring");
    if (context.num_kv_heads != kKVHeads || context.head_dim != kHeadDim ||
        context.capacity != static_cast<std::uint32_t>(detail::kCausalSlidingWindow) ||
        context.padded_capacity < context.capacity || context.k.field.layers != 1 ||
        context.v.field.layers != 1) {
        throw std::invalid_argument(std::string(op) + ": invalid cyclic ring");
    }
    if (context.padded_capacity >
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error(std::string(op) + ": padded capacity exceeds int32");
    }
    const auto padded = static_cast<std::int32_t>(context.padded_capacity);
    common::operand::state_field(context.k, DType::BF16, {kHeadDim, padded, kKVHeads}, 16, op,
                                 "ring K", device);
    common::operand::state_field(context.v, DType::FP16, {kHeadDim, padded, kKVHeads}, 16, op,
                                 "ring V", device);
    common::operand::shape(context.k.records.byte_offsets, {batch, 1, 1, 1}, op, "K offsets");
    common::operand::shape(context.v.records.byte_offsets, {batch, 1, 1, 1}, op, "V offsets");
}

template <class Allocator>
detail::CausalPartialStorage allocate_partials(Allocator& allocator,
                                               const detail::CausalSlidingWindowPlan& plan) {
    return detail::allocate_causal_partials(allocator, kQHeads, plan.tokens, plan.splits,
                                            plan.batch);
}

} // namespace

std::size_t causal_sliding_window_attention_workspace_capacity_bytes(
    AttentionHeadGeometry geometry, std::uint32_t window, std::int32_t batch_size,
    std::int32_t min_tokens, std::int32_t max_tokens, std::int32_t multiprocessor_count) {
    constexpr const char* op = "causal_sliding_window_attention workspace";
    require_profile(geometry, window, op);
    if (batch_size < 1 || batch_size > detail::kCausalSlidingWindowMaxBatch || min_tokens < 1 ||
        max_tokens < min_tokens ||
        (batch_size > 1 && max_tokens > detail::kCausalSlidingWindowMaxBatchedTokens) ||
        multiprocessor_count < 1) {
        throw std::invalid_argument(std::string(op) +
                                    ": invalid batch, token interval or SM count");
    }
    // The plan is constant while a token count stays inside one query tile, and the allocation
    // grows with T there. Evaluate the last T of each tile.
    std::size_t capacity = 0;
    for (std::int64_t tokens = min_tokens; tokens <= max_tokens;) {
        const auto first = detail::select_causal_sliding_window_plan(
            static_cast<std::int32_t>(tokens), batch_size, multiprocessor_count);
        const std::int64_t tile_last =
            (tokens * detail::kCausalSlidingWindowGroup + first.query_rows - 1) / first.query_rows *
            first.query_rows / detail::kCausalSlidingWindowGroup;
        const std::int64_t last = std::min<std::int64_t>(max_tokens, tile_last);
        const auto plan         = detail::select_causal_sliding_window_plan(
            static_cast<std::int32_t>(last), batch_size, multiprocessor_count);
        if (plan.splits > 1) {
            WorkspaceLayoutBuilder layout;
            (void)allocate_partials(layout, plan);
            capacity = std::max(capacity, layout.peak_bytes(1));
        }
        tokens = last + 1;
    }
    return capacity;
}

void causal_sliding_window_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                     const Tensor& positions, const Tensor& valid_columns,
                                     AttentionHeadGeometry geometry, std::uint32_t window,
                                     float scale, const ConstCyclicKVCacheLayerView& context,
                                     MemoryArena& workspace, Tensor& out,
                                     CudaExecutionView execution) {
    constexpr const char* op = "causal_sliding_window_attention";
    if (workspace.device() != execution.device()) {
        throw std::invalid_argument(std::string(op) + ": workspace belongs to another device");
    }
    require_profile(geometry, window, op);
    if (q.dtype != DType::BF16 || k.dtype != DType::BF16 || v.dtype != DType::BF16 ||
        out.dtype != DType::BF16) {
        throw std::invalid_argument(std::string(op) + ": q/k/v/out must be BF16");
    }
    if (positions.dtype != DType::I32 || valid_columns.dtype != DType::I32) {
        throw std::invalid_argument(std::string(op) + ": positions/valid_columns must be I32");
    }
    if (!std::isfinite(scale) || std::abs(scale - kExpectedScale) > 1.0e-6f) {
        throw std::invalid_argument(std::string(op) + ": scale must be 1/sqrt(256)");
    }
    const std::int32_t tokens = q.ne[2];
    const std::int32_t batch  = q.ne[3];
    if (tokens < 1 || batch < 1 || batch > detail::kCausalSlidingWindowMaxBatch ||
        (batch > 1 && tokens > detail::kCausalSlidingWindowMaxBatchedTokens)) {
        throw std::invalid_argument(std::string(op) + ": unsupported B/T domain");
    }
    const Device device = execution.device();
    common::operand::shape(q, {kHeadDim, kQHeads, tokens, batch}, op, "q");
    common::operand::shape(k, {kHeadDim, kKVHeads, tokens, batch}, op, "k");
    common::operand::shape(v, {kHeadDim, kKVHeads, tokens, batch}, op, "v");
    common::operand::shape(positions, {tokens, batch, 1, 1}, op, "positions");
    common::operand::shape(valid_columns, {batch, 1, 1, 1}, op, "valid columns");
    common::operand::shape(out, {kHeadDim, kQHeads, tokens, batch}, op, "out");
    common::operand::contiguous_nonnull(q, op, "q", device);
    common::operand::contiguous_nonnull(k, op, "k", device);
    common::operand::contiguous_nonnull(v, op, "v", device);
    common::operand::contiguous_nonnull(positions, op, "positions", device);
    common::operand::contiguous_nonnull(valid_columns, op, "valid columns", device);
    common::operand::contiguous_nonnull(out, op, "out", device);
    for (const Tensor* tensor : {&q, &k, &v, static_cast<const Tensor*>(&out)}) {
        if (!common::operand::aligned(tensor->data, 16)) {
            throw std::invalid_argument(std::string(op) + ": q/k/v/out must be 16-byte aligned");
        }
    }
    validate_context(context, batch, op, device);

    auto scope = workspace.scope();
    const auto plan =
        detail::select_causal_sliding_window_plan(tokens, batch, execution.multiprocessor_count());
    detail::CausalPartialView partial{};
    if (plan.splits > 1) partial = allocate_partials(workspace, plan).view();
    detail::launch_causal_sliding_window_attention(q, k, v, positions, valid_columns, scale,
                                                   context, plan, partial, out, execution.stream());
}

} // namespace ninfer::ops
