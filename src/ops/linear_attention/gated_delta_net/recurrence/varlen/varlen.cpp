#include "ops/common/host/operands.h"
#include "ninfer/ops/gated_delta_net.h"
#include "core/device.h"
#include "ops/linear_attention/gated_delta_net/recurrence/plan.h"
#include "ops/linear_attention/gated_delta_net/recurrence/varlen/launch.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops::detail::gated_delta_net::varlen {

int chunked_min_tokens(int qk_heads, int value_heads) {
    // The single-segment selector owns the route. It must be a pure length threshold at or below
    // one chunk for the sub-segment walk to reproduce it.
    constexpr int kProbe = 4 * chunked::kChunkSize;
    int threshold        = 0;
    for (int tokens = 1; tokens <= kProbe; ++tokens) {
        const bool chunked = select_recurrence_route(qk_heads, value_heads, tokens) ==
                             RecurrenceRoute::Chunked;
        if (threshold == 0 && chunked) threshold = tokens;
        if (threshold != 0 && !chunked)
            throw std::logic_error("gated_delta_net_varlen: recurrence route is not a threshold");
    }
    if (threshold == 0 || threshold > chunked::kChunkSize)
        throw std::logic_error("gated_delta_net_varlen: unsupported recurrence route threshold");
    return threshold;
}

WorkspaceLayout workspace_layout(int qk_heads, int value_heads, int rows, int tokens,
                                 int boundaries, int chunked_threshold) {
    const std::int64_t sub_segments = static_cast<std::int64_t>(rows) + boundaries;
    // Every chunked sub-segment of length L takes ceil(L/16) <= (L+15)/16 chunks, and at most
    // tokens/threshold sub-segments are long enough to be chunked.
    const std::int64_t chunked_count =
        std::min<std::int64_t>(sub_segments, tokens / chunked_threshold);
    const std::int64_t chunks =
        chunked_count == 0 ? 0 : (tokens + (chunked::kChunkSize - 1) * chunked_count) /
                                     chunked::kChunkSize;
    WorkspaceLayout layout{};
    layout.chunk_capacity = static_cast<std::int32_t>(chunks);
    layout.sub_segments   = static_cast<std::int32_t>(sub_segments);
    LayoutBuilder builder;
    layout.plan = builder.add(sizeof(PlanHeader) + sizeof(SubSegment) * sub_segments +
                                  sizeof(std::int32_t) * (static_cast<std::size_t>(rows) + 1),
                              256, "gdn.varlen.plan");
    if (chunks > 0) {
        layout.qk = builder.add_tensor(
            DType::U8, {int(sizeof(chunked::QkChunk)), layout.chunk_capacity, qk_heads}, 256,
            "gdn.varlen.qk");
        layout.control = builder.add_tensor(
            DType::U8, {int(sizeof(chunked::ControlChunk)), layout.chunk_capacity, value_heads},
            256, "gdn.varlen.control");
    }
    layout.total_bytes = builder.finish(256, "GDN varlen workspace");
    return layout;
}

} // namespace ninfer::ops::detail::gated_delta_net::varlen

namespace ninfer::ops {
namespace {

namespace gdn = detail::gated_delta_net;

constexpr std::int32_t kMaxRows       = gdn::varlen::kMaxRows;
constexpr std::int32_t kMaxBoundaries = gdn::varlen::kMaxBoundaries;

std::string geometry_text(std::int64_t qk_heads, std::int64_t value_heads, std::int64_t rows,
                          std::int64_t boundaries, std::int64_t tokens) {
    return "Hq=" + std::to_string(qk_heads) + " Hv=" + std::to_string(value_heads) +
           " R=" + std::to_string(rows) + " C=" + std::to_string(boundaries) +
           " T=" + std::to_string(tokens);
}

[[noreturn]] void reject(const std::string& what, const std::string& geometry) {
    throw std::invalid_argument("gated_delta_net_varlen: " + what + " (" + geometry + ")");
}

bool supported_profile(std::int64_t qk_heads, std::int64_t value_heads) {
    return qk_heads == 16 && (value_heads == 32 || value_heads == 48 || value_heads == 64);
}

void require_domain(std::int64_t qk_heads, std::int64_t value_heads, std::int64_t rows,
                    std::int64_t boundaries, std::int64_t tokens, bool require_boundary_room) {
    const auto geometry = geometry_text(qk_heads, value_heads, rows, boundaries, tokens);
    if (!supported_profile(qk_heads, value_heads))
        reject("(Hq,Hv) must be one of (16,32), (16,48), (16,64)", geometry);
    if (rows < 1 || rows > kMaxRows) reject("R must be in [1,1024]", geometry);
    if (boundaries < 0 || boundaries > kMaxBoundaries) reject("C must be in [0,1024]", geometry);
    if (tokens < 1) reject("T must be positive", geometry);
    if (require_boundary_room && (tokens < rows || boundaries > tokens - rows))
        reject("every row needs a token and every boundary an extra one: T>=R and C<=T-R",
               geometry);
}

void validate(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
              const Tensor& beta, float scale, const GdnVarlenRows& rows,
              const GdnVarlenBoundaries& boundaries, const StateFieldLayout& checkpoint_field,
              std::int32_t row_count, std::int32_t boundary_count, const Tensor& out,
              Device device) {
    constexpr const char* op = "gated_delta_net_varlen";
    const std::int32_t hq = q.ne[1], hv = v.ne[1], tokens = q.ne[2];
    namespace operand = common::operand;
    operand::dtype(q, DType::BF16, op, "q must be BF16");
    operand::dtype(k, DType::BF16, op, "k must be BF16");
    operand::dtype(v, DType::BF16, op, "v must be BF16");
    operand::dtype(out, DType::BF16, op, "out must be BF16");
    operand::dtype(g, DType::FP32, op, "g must be FP32");
    operand::dtype(beta, DType::FP32, op, "beta must be FP32");
    operand::shape(q, {gdn::kStateDim, hq, tokens, 1}, op, "q");
    operand::shape(k, {gdn::kStateDim, hq, tokens, 1}, op, "k");
    operand::shape(v, {gdn::kStateDim, hv, tokens, 1}, op, "v");
    operand::shape(out, {gdn::kStateDim, hv, tokens, 1}, op, "out");
    operand::shape(g, {hv, tokens, 1, 1}, op, "g");
    operand::shape(beta, {hv, tokens, 1, 1}, op, "beta");
    operand::contiguous_nonnull(q, op, "q", device);
    operand::contiguous_nonnull(k, op, "k", device);
    operand::contiguous_nonnull(v, op, "v", device);
    operand::contiguous_nonnull(g, op, "g", device);
    operand::contiguous_nonnull(beta, op, "beta", device);
    operand::contiguous_nonnull(out, op, "out", device);
    gdn::require_scale(scale, op);

    operand::tensor(rows.token_offsets, DType::I32, {row_count + 1}, alignof(std::int32_t), op,
                    "token offsets", device);
    const std::initializer_list<std::int32_t> field_shape{gdn::kStateDim, gdn::kStateDim, hv};
    operand::state_field(rows.source, DType::FP32, field_shape, 16, op, "source", device);
    operand::state_field(rows.destination, DType::FP32, field_shape, 16, op, "destination",
                         device);
    for (const Tensor* offsets :
         {&rows.source.records.byte_offsets, &rows.destination.records.byte_offsets}) {
        if (offsets->ne[0] != row_count || offsets->ne[1] != 1 || offsets->ne[2] != 1 ||
            offsets->ne[3] != 1)
            throw std::invalid_argument(std::string(op) +
                                        ": state byte offsets must select exactly R records");
    }
    if (rows.source.field.layers != 1 || rows.destination.field.layers != 1)
        throw std::invalid_argument(std::string(op) + ": select one recurrent state layer");

    if (boundary_count == 0) return;
    operand::tensor(boundaries.rows, DType::I32, {boundary_count}, alignof(std::int32_t), op,
                    "boundary rows", device);
    operand::tensor(boundaries.columns, DType::I32, {boundary_count}, alignof(std::int32_t), op,
                    "boundary columns", device);
    operand::tensor(boundaries.destination_offsets, DType::I64, {boundary_count},
                    alignof(std::int64_t), op, "boundary destination offsets", device);
    if (checkpoint_field.layers != 1)
        throw std::invalid_argument(std::string(op) + ": select one checkpoint state layer");
    const StateFieldView checkpoint{rows.destination.records, checkpoint_field};
    operand::state_field(checkpoint, DType::FP32, field_shape, 16, op, "checkpoint field", device,
                         false);
}

} // namespace

std::size_t gated_delta_net_varlen_workspace_capacity_bytes(std::int32_t qk_heads,
                                                            std::int32_t value_heads,
                                                            std::int32_t max_rows,
                                                            std::int32_t max_tokens,
                                                            std::int32_t max_boundaries) {
    require_domain(qk_heads, value_heads, max_rows, max_boundaries, max_tokens, false);
    const int threshold = gdn::varlen::chunked_min_tokens(qk_heads, value_heads);
    return gdn::varlen::workspace_layout(qk_heads, value_heads, max_rows, max_tokens,
                                         max_boundaries, threshold)
        .total_bytes;
}

void gated_delta_net_varlen(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk, GdnVarlenRows rows,
                            GdnVarlenBoundaries boundaries, StateFieldLayout checkpoint_field,
                            std::int32_t row_count, std::int32_t boundary_count, MemoryArena& ws,
                            Tensor& out, CudaExecutionView execution) {
    const std::int32_t hq = q.ne[1], hv = v.ne[1], tokens = q.ne[2];
    const auto geometry = geometry_text(hq, hv, row_count, boundary_count, tokens);
    require_domain(hq, hv, row_count, boundary_count, tokens, true);
    try {
        validate(q, k, v, g, beta, scale, rows, boundaries, checkpoint_field, row_count,
                 boundary_count, out, execution.device());
    } catch (const std::invalid_argument& error) {
        throw std::invalid_argument(std::string(error.what()) + " (" + geometry + ")");
    }
    if (ws.device() != execution.device())
        throw std::invalid_argument("gated_delta_net_varlen: workspace belongs to another device (" +
                                    geometry + ")");

    const int threshold = gdn::varlen::chunked_min_tokens(hq, hv);
    const auto layout =
        gdn::varlen::workspace_layout(hq, hv, row_count, tokens, boundary_count, threshold);
    auto scope         = ws.scope();
    const auto backing = ws.alloc_bytes(layout.total_bytes);

    auto* plan = static_cast<std::byte*>(layout.plan.bind(backing).data);
    gdn::varlen::Workspace workspace{};
    workspace.header = reinterpret_cast<gdn::varlen::PlanHeader*>(plan);
    workspace.sub_segments =
        reinterpret_cast<gdn::varlen::SubSegment*>(plan + sizeof(gdn::varlen::PlanHeader));
    workspace.row_first = reinterpret_cast<std::int32_t*>(
        plan + sizeof(gdn::varlen::PlanHeader) +
        sizeof(gdn::varlen::SubSegment) * static_cast<std::size_t>(layout.sub_segments));
    workspace.chunk_capacity = layout.chunk_capacity;
    if (layout.chunk_capacity > 0) {
        workspace.qk = static_cast<gdn::chunked::QkChunk*>(layout.qk.bind(backing).data);
        workspace.control =
            static_cast<gdn::chunked::ControlChunk*>(layout.control.bind(backing).data);
    }

    const auto* source_base = static_cast<const std::byte*>(rows.source.records.storage.data) +
                              rows.source.field.tensor.region.offset;
    auto* destination_base = static_cast<std::byte*>(rows.destination.records.storage.data) +
                             rows.destination.field.tensor.region.offset;
    auto* checkpoint_base = static_cast<std::byte*>(rows.destination.records.storage.data) +
                            checkpoint_field.tensor.region.offset;
    const gdn::varlen::Operands operands{
        static_cast<const __nv_bfloat16*>(q.data),
        static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data),
        static_cast<const float*>(g.data),
        static_cast<const float*>(beta.data),
        static_cast<__nv_bfloat16*>(out.data),
        static_cast<const std::int32_t*>(rows.token_offsets.data),
        source_base,
        static_cast<const std::int64_t*>(rows.source.records.byte_offsets.data),
        destination_base,
        static_cast<const std::int64_t*>(rows.destination.records.byte_offsets.data),
        checkpoint_base,
        static_cast<const std::int32_t*>(boundaries.rows.data),
        static_cast<const std::int32_t*>(boundaries.columns.data),
        static_cast<const std::int64_t*>(boundaries.destination_offsets.data),
        hq,
        hv,
        tokens,
        row_count,
        boundary_count,
        threshold,
        scale};
    const int tile = gdn::select_value_tile(hq, hv, tokens, execution.multiprocessor_count());
    gdn::varlen::launch(operands, workspace, normalize_qk, tile, execution.stream());
}

} // namespace ninfer::ops
