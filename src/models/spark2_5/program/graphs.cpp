#include "models/spark2_5/program/graphs.h"

#include "core/startup.h"
#include "core/transfer.h"
#include "models/spark2_5/program/internal.h"

#include <algorithm>

namespace ninfer::models::spark2_5::detail {
namespace {

std::vector<RecordSpec> specs(const ExecutionPlan& plan) {
    const auto& image = plan.persistent.state.images;
    const auto& pages = plan.persistent.kv.pages;
    return {{image.image_bytes, image.alignment}, {pages.page_bytes, pages.alignment}};
}

std::vector<RecordRequest> demand(const ContextPoolGeometry& geometry, std::uint32_t batch) {
    std::vector<RecordRequest> out;
    for (const auto id : geometry.spec_classes) out.push_back({id, batch});
    return out;
}

} // namespace

std::size_t graph_context_bytes(const ExecutionPlan& plan) {
    const auto geometry = describe_context_pool(specs(plan));
    return geometry.required_bytes(demand(geometry, plan.shape.max_concurrency));
}

GraphInputs::GraphInputs(RankProgramImpl& owner) : owner_(owner) {}

void GraphInputs::prepare() {
    auto& p             = owner_;
    auto& r             = p.resources;
    const auto records  = specs(r.plan);
    const auto geometry = describe_context_pool(records);
    const auto requests = demand(geometry, p.shape.max_concurrency);
    const auto bytes    = geometry.required_bytes(requests);
    if (bytes > r.context_backing().bytes)
        throw std::logic_error("Spark2_5 graph records exceed the planned Context working set");
    records_         = std::make_unique<ContextPool>(0, bytes, records);
    auto reservation = records_->reserve(requests);
    if (!reservation.reservation || !reservation.reservation->ready())
        throw std::logic_error("Spark2_5 graph preparation cannot reserve its Context records");
    const auto allocate = [&](std::size_t kind) {
        const auto id = reservation.reservation->materialize(records_->record_class(kind));
        allocations_.push_back(id);
        const auto binding = records_->resolve(id);
        const auto span    = r.context_backing().subspan(binding.offset, binding.bytes);
        CUDA_CHECK(cudaMemsetAsync(span.data, 0, span.bytes, p.cuda.stream()));
        return static_cast<std::int64_t>(binding.offset);
    };
    for (std::uint32_t row = 0; row < p.shape.max_concurrency; ++row)
        states_.push_back(allocate(0));
    auto& tables         = r.kv.tables();
    const auto capacity  = tables.logical_page_capacity();
    const auto row_bytes = std::size_t(capacity) * sizeof(std::int64_t);
    mapping_             = PinnedHostBuffer(row_bytes * p.shape.max_concurrency);
    for (std::uint32_t row = 0; row < p.shape.max_concurrency; ++row) {
        const auto page    = allocate(1);
        const auto staging = mapping_.span().subspan(row * row_bytes, row_bytes);
        auto* offsets      = static_cast<std::int64_t*>(staging.data);
        // Every logical page names the one record, so any history length is addressable.
        std::fill_n(offsets, capacity, page);
        tables.publish(row, 0, {offsets, capacity}, staging, p.cuda);
    }
    p.device.synchronize();
}

GraphInputs::~GraphInputs() {
    // prepare() reports execution failures; unwinding still drains the staging readers.
    (void)cudaStreamSynchronize(owner_.cuda.stream());
    for (const auto id : allocations_) records_->release(id);
}

void GraphInputs::set(std::uint32_t batch, std::uint32_t visible) {
    auto& p = owner_;
    p.device.synchronize();
    const auto& io  = p.resources.plan.persistent.buffers;
    const auto last = std::max(1U, visible) - 1;
    const auto i32  = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(p.host_tensor(r).data);
    };
    for (std::uint32_t b = 0; b < batch; ++b) {
        i32(io.ids)[b]                                              = 0;
        i32(io.positions)[b]                                        = last;
        i32(io.valid_columns)[b]                                    = 1;
        i32(io.kv_rows)[b]                                          = b;
        static_cast<std::int64_t*>(p.host_tensor(io.destination_offsets).data)[b] = states_[b];
        i32(io.sampling_positions)[b]                               = last;
        i32(io.count_slots)[b]                                      = b;
        i32(io.count_columns)[b]                                    = 0;
    }
    auto* configs = static_cast<ops::SamplingConfig*>(p.host_tensor(*io.sampling).data);
    for (std::uint32_t b = 0; b < batch; ++b) configs[b] = {.temperature = 0, .top_k = 1, .top_p = 1};
    p.upload_inputs(batch);
}

Graphs::Graphs(RankProgramImpl& owner)
    : owner_(owner), bounds_(decode_context_bounds(owner.shape.max_context)) {}

std::size_t Graphs::index(std::uint32_t visible) const {
    const auto found = std::lower_bound(bounds_.begin(), bounds_.end(), visible);
    if (found == bounds_.end())
        throw std::logic_error("Spark2_5 decode graph context exceeds its capacity");
    return static_cast<std::size_t>(found - bounds_.begin());
}

void Graphs::prepare(GraphInputs& inputs, GraphPreparation& preparation) {
    auto& p = owner_;
    batches_.resize(p.shape.max_concurrency);
    for (std::uint32_t b = 1; b <= p.shape.max_concurrency; ++b) {
        auto& batch = batches_[b - 1];
        for (const auto& envelope : p.envelopes) {
            inputs.set(b, envelope.max_visible_keys);
            if (!preparation.capturing()) {
                p.decode_round(b, envelope, false, true);
                preparation.checkpoint();
                continue;
            }
            auto& round = batch.rounds.emplace_back();
            p.resources.workspace.reset_peak();
            round.prepare(p.cuda, [&] { p.decode_round(b, envelope, true, true); });
            workspace_peak_ = std::max(workspace_peak_, p.resources.workspace.peak_used());
            round.launch(p.cuda);
            preparation.checkpoint();
        }
        inputs.set(b, 1);
        const auto decision = [&] {
            p.sample(b, false);
            p.copy_result(b);
        };
        if (!preparation.capturing()) {
            decision();
            preparation.checkpoint();
            continue;
        }
        p.resources.workspace.reset_peak();
        batch.decision.prepare(p.cuda, decision);
        workspace_peak_ = std::max(workspace_peak_, p.resources.workspace.peak_used());
        batch.decision.launch(p.cuda);
        preparation.checkpoint();
    }
}

void Graphs::replay(std::uint32_t batch, std::uint32_t visible) {
    batches_.at(batch - 1).rounds.at(index(visible)).launch(owner_.cuda);
}

void Graphs::decide(std::uint32_t batch) { batches_.at(batch - 1).decision.launch(owner_.cuda); }

GraphPreparationSample RankProgramImpl::prepare_graphs(GraphPreparationMode mode) {
    if (!shape.cuda_graph || graphs) return {};
    if (mode == GraphPreparationMode::Complete) {
        prepare_graphs(GraphPreparationMode::Warmup);
        return prepare_graphs(GraphPreparationMode::Capture);
    }
    GraphPreparation observation(device, mode);
    // Inputs outlive a failed preparation until the executor drains the Program.
    graph_inputs = std::make_unique<GraphInputs>(*this);
    graph_inputs->prepare();
    graphs = std::make_unique<Graphs>(*this);
    graphs->prepare(*graph_inputs, observation);
    observation.checkpoint();
    graph_inputs.reset();
    const auto result = observation.sample();
    if (observation.capturing()) {
        graph_memory.preparation_increment_bytes = result.increment_bytes;
        graph_memory.preparation_peak_bytes      = result.peak_bytes;
    } else {
        graphs.reset();
        graph_memory.warmup_increment_bytes = result.increment_bytes;
    }
    return result;
}

void RankProgramImpl::release_graphs() {
    device.synchronize();
    graphs.reset();
    graph_inputs.reset();
}

} // namespace ninfer::models::spark2_5::detail
