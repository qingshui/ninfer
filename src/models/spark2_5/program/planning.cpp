#include "models/spark2_5/program/planning.h"

#include "core/platform.h"
#include "models/common/context/storage.h"
#include "models/spark2_5/options.h"
#include "models/spark2_5/prefill_chunk.h"
#include "ninfer/ops/sampling.h"

#include <algorithm>

namespace ninfer::models::spark2_5 {
namespace {

EngineOptions options(const ModelProfile& profile, const EngineOptions& supplied) {
    auto o            = supplied;
    const auto& model = profile.definition->config();
    if (o.max_concurrency == 0 || o.max_concurrency > kMaximumSupportedConcurrency)
        throw std::invalid_argument("Spark2_5 supports max_concurrency in [1," +
                                    std::to_string(kMaximumSupportedConcurrency) + "]");
    if (!o.max_context || o.max_context > model.max_context)
        throw std::invalid_argument("Spark2_5 max_context exceeds the model's context window");
    if (o.speculative.backend != SpeculativeBackend::None || o.speculative.draft_tokens)
        throw std::invalid_argument("Spark2_5 has no speculative decoding");
    if (o.purpose == EnginePurpose::CausalScoring && o.max_concurrency != 1)
        throw std::invalid_argument("scoring requires normalized single-lane text options");
    o.prefill_chunk = resolve_prefill_chunk(o);
    o.kv_cache      = resolve_kv_cache(o);
    return o;
}

PersistentLayout persistent(const ModelProfile& profile, const EngineOptions& o) {
    const auto& config = profile.definition->config();
    LayoutBuilder b;
    PersistentLayout out;
    out.state   = plan_state_image(config);
    out.kv      = plan_full_kv(b, config.full_layers, o.max_context, resolve_kv_cache(o),
                               static_cast<std::int32_t>(o.max_concurrency));
    out.buffers = plan_buffers(b, config, o, o.prefill_chunk, profile.public_tokens);
    out.bytes   = b.finish(256);
    return out;
}

// Largest transient the Program ever asks of its arena: a prefill window, a decode round at
// each batch size and context envelope, the head projection and the sampler.
std::size_t workspace_bytes(const ModelProfile& profile, const EngineOptions& o,
                            DeviceContext& device) {
    const auto& config = profile.definition->config();
    const auto sms     = device.multiprocessor_count();
    const auto kv      = resolve_kv_cache(o);
    std::size_t peak   = 0;
    const auto text    = [&](std::int32_t batch, std::int32_t tokens,
                          ops::CausalAttentionExecutionEnvelope envelope) {
        TextWorkspaceShape shape;
        shape.batch                = batch;
        shape.max_tokens           = tokens;
        shape.envelope             = envelope;
        shape.kv_storage           = kv;
        shape.multiprocessor_count = sms;
        peak = std::max(peak, text_workspace_bytes(config, profile.planning_layers, shape));
    };
    const std::uint32_t chunk = o.prefill_chunk;
    text(1, static_cast<std::int32_t>(chunk), {1, o.max_context});
    if (o.purpose == EnginePurpose::CausalScoring) {
        const auto columns = std::min(kScoreColumns, chunk);
        const auto logits  = std::size_t(config.vocabulary) * columns * 2;
        peak = std::max(peak, output_workspace_bytes(profile.planning_head, 1, columns) +
                                  logits + std::size_t(columns) * 16 + 4096);
        return peak;
    }
    for (std::uint32_t batch = 1; batch <= o.max_concurrency; ++batch)
        for (const auto& envelope : decode_envelopes(o.max_context))
            text(static_cast<std::int32_t>(batch), 1, envelope);
    const auto rows = static_cast<std::int32_t>(o.max_concurrency);
    peak = std::max(peak, output_workspace_bytes(profile.planning_head, 1, rows));
    peak = std::max(peak, ops::sampling_workspace_capacity_bytes(
                              static_cast<std::int32_t>(profile.public_tokens), 1, rows));
    return peak;
}

ExecutionPlan physical_plan(std::shared_ptr<const ModelProfile> profile, const EngineOptions& o,
                            DeviceContext& device) {
    ExecutionPlan out;
    out.shape      = describe_execution_shape(o);
    out.persistent = persistent(*profile, o);
    // Graph capture takes the same arena as the eager path, so one figure covers both.
    out.workspace_bytes    = workspace_bytes(*profile, o, device);
    out.host_context_bytes = context::host_context_capacity(
        o.context_cache.host_capacity_bytes, out.persistent.state.images.image_bytes);
    const auto& buffers  = out.persistent.buffers;
    out.pinned_io_bytes  = buffers.ingress.bytes;
    if (buffers.sampled_ids) {
        out.pinned_sampling_bytes = buffers.grammar_masks->region.bytes;
        out.pinned_output_bytes   = buffers.sampled_ids->region.bytes;
    }
    if (o.purpose == EnginePurpose::CausalScoring)
        out.pinned_output_bytes = std::min(o.prefill_chunk, kScoreColumns) * sizeof(float);
    out.profile = std::move(profile);
    return out;
}

} // namespace

std::vector<std::uint32_t> decode_context_bounds(std::uint32_t capacity) {
    std::vector<std::uint32_t> out;
    for (const auto bound : {128U, 512U, 2048U, 8192U, 32768U, 131072U}) {
        if (bound >= capacity) break;
        out.push_back(bound);
    }
    out.push_back(capacity);
    return out;
}

std::vector<ops::CausalAttentionExecutionEnvelope> decode_envelopes(std::uint32_t capacity) {
    std::vector<ops::CausalAttentionExecutionEnvelope> out;
    std::uint32_t low = 1;
    for (const auto bound : decode_context_bounds(capacity)) {
        out.push_back({low, bound});
        low = bound + 1;
    }
    return out;
}

ExecutionShape describe_execution_shape(const EngineOptions& supplied) {
    // The shape depends on options alone; the profile only bounds them during planning.
    auto o = supplied;
    if (!o.prefill_chunk) o.prefill_chunk = resolve_prefill_chunk(o);
    return {o.max_context,
            o.max_concurrency,
            std::min(o.prefill_chunk, o.max_context),
            resolve_kv_cache(o),
            o.purpose,
            o.context_cache.enabled,
            o.use_cuda_graph,
            o.observability.gpu_timing};
}

std::size_t ExecutionPlan::device_bytes() const {
    return capacity.device_bytes + persistent.bytes + workspace_bytes;
}

std::size_t ExecutionPlan::pinned_context_bytes() const {
    const auto& tables = persistent.kv.tables.spec;
    return 2 * std::size_t(tables.logical_page_capacity) * tables.table_rows * sizeof(std::int64_t);
}

std::size_t ExecutionPlan::pinned_aux_bytes() const {
    return std::size_t(std::max(1024U, persistent.buffers.max_positions)) * sizeof(TokenId);
}

std::size_t ExecutionPlan::pinned_context_counts_bytes() const {
    return persistent.buffers.token_counts
               ? std::size_t(profile->public_tokens) * shape.max_concurrency * sizeof(std::int32_t)
               : 0;
}

std::size_t ExecutionPlan::host_reservation_bytes() const {
    return host_context_bytes + pinned_io_bytes + pinned_output_bytes + pinned_sampling_bytes +
           pinned_context_bytes() + pinned_aux_bytes() + pinned_context_counts_bytes();
}

std::size_t ExecutionPlan::device_reservation_bytes() const {
    return device_bytes() + ((NINFER_SHARED_PHYSICAL_MEMORY != 0) ? host_reservation_bytes() : 0);
}

namespace {
runtime::ContextCapacityPlan capacity_plan(const ExecutionPlan& base, const EngineOptions& o) {
    runtime::ContextCapacityPlan plan;
    const auto& state = base.persistent.state.images;
    const auto& pages = base.persistent.kv.pages;
    plan.records.push_back({state.image_bytes, state.alignment});
    plan.records.push_back({pages.page_bytes, pages.alignment});
    const auto geometry = describe_context_pool(plan.records);
    const auto work     = [&](std::string name, std::uint32_t states, std::uint32_t kv_pages) {
        runtime::ContextStartupWorkingSet set{std::move(name)};
        set.records.push_back({geometry.spec_classes[0], states});
        set.records.push_back({geometry.spec_classes[1], kv_pages});
        return set;
    };
    const auto page_count = [](std::uint64_t tokens) {
        return static_cast<std::uint32_t>((tokens + kPagedKVPageSize - 1) / kPagedKVPageSize);
    };
    auto maximum   = work("maximum context", 1, page_count(o.max_context));
    plan.reference = {o.max_context, 1, geometry.required_bytes(maximum.records)};
    plan.working_sets.push_back(std::move(maximum));
    const auto short_tokens = std::min<std::uint64_t>(o.max_context, std::uint64_t(o.prefill_chunk) + 1);
    plan.working_sets.push_back(work("concurrent requests", o.max_concurrency,
                                     o.max_concurrency * page_count(short_tokens)));
    plan.fixed_device_bytes = base.device_reservation_bytes();
    plan.fixed_host_bytes   = base.host_reservation_bytes();
    plan.metadata           = session_metadata_plan(base, geometry);
    return plan;
}
} // namespace

std::optional<ExecutionPlan> plan_execution(std::shared_ptr<const ModelProfile> profile,
                                            const EngineOptions& supplied, DeviceContext& device,
                                            std::size_t available,
                                            std::size_t available_host_bytes) {
    if (!profile) throw std::invalid_argument("execution planning requires a model profile");
    const auto o       = options(*profile, supplied);
    auto out           = physical_plan(std::move(profile), o, device);
    const auto budget  = capacity_plan(out, o);
    const auto& policy = o.context_capacity;
    const auto minimum = budget.minimum_capacity();
    const auto required =
        policy.mode == ContextCapacityMode::Explicit ? runtime::ContextCapacity{policy.explicit_bytes}
                                                     : minimum;
    const auto reserved = budget.reservation_bytes(required);
    const auto metadata = NINFER_SHARED_PHYSICAL_MEMORY != 0 ? budget.control_bytes(required) : 0;
    if (reserved > available || metadata > available - reserved) return {};
    out.capacity = runtime::resolve_context_capacity(policy, budget, available, available_host_bytes,
                                                     NINFER_SHARED_PHYSICAL_MEMORY != 0, true);
    out.capacity.mode       = o.context_capacity.mode;
    out.checkpoint_interval = o.context_cache.checkpoint_interval;
    return out;
}

ExecutionPlan plan_graph_probe(std::shared_ptr<const ModelProfile> profile,
                               const EngineOptions& supplied, DeviceContext& device,
                               GraphProbeBacking backing) {
    auto o                              = options(*profile, supplied);
    o.context_cache.host_capacity_bytes = 0;
    auto plan           = physical_plan(std::move(profile), o, device);
    plan.graph_probe    = true;
    const auto budget   = capacity_plan(plan, o);
    const auto geometry = describe_context_pool(budget.records);
    std::vector<RecordRequest> records;
    for (const auto id : geometry.spec_classes) records.push_back({id, o.max_concurrency});
    plan.capacity.device_bytes              = backing == GraphProbeBacking::Inputs
                                                  ? geometry.required_bytes(records)
                                                  : budget.minimum_capacity().device_bytes;
    plan.capacity.block_bytes               = geometry.block_bytes;
    plan.capacity.runtime_reservation_bytes = plan.device_reservation_bytes();
    return plan;
}

} // namespace ninfer::models::spark2_5
