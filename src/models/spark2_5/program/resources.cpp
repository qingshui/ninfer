#include "models/spark2_5/program/resources.h"

#include "core/transfer.h"

#include <cstring>

namespace ninfer::models::spark2_5 {
namespace {

ExecutionPlan checked(ExecutionPlan plan) {
    if (!plan.profile || plan.capacity.runtime_reservation_bytes != plan.device_reservation_bytes())
        throw std::invalid_argument("Spark2_5 resources require the finalized execution plan");
    return plan;
}

} // namespace

Resources::Resources(ExecutionPlan input, DeviceContext& device, const StartupObserver& observer)
    : plan(checked(std::move(input))), device_(device),
      context_(device.device(), plan.capacity.device_bytes),
      persistent_(device.device(), plan.persistent.bytes),
      workspace_(device.device(), plan.workspace_bytes), ingress_(plan.pinned_io_bytes),
      egress_(plan.pinned_output_bytes), sampling_(plan.pinned_sampling_bytes),
      compute_aux_(plan.pinned_aux_bytes()), context_counts_(plan.pinned_context_counts_bytes()),
      workspace(workspace_.span()), states(context_.span(), plan.persistent.state),
      kv(context_.span(), persistent_.span(), plan.persistent.kv) {
    if (plan.host_context_bytes) {
        StartupPhaseScope phase(observer, StartupPhase::HostContextPin, StartupProgressUnit::Bytes,
                                plan.host_context_bytes);
        host_context_ = PinnedHostBuffer(plan.host_context_bytes);
        phase.complete(plan.host_context_bytes, plan.host_context_bytes);
    }
    const auto backing = persistent_.span();
    std::memset(ingress_.data(), 0, ingress_.size());
    copy_bytes_async(plan.persistent.buffers.ingress.bind(backing), ingress_.span(),
                     device.execution_view());
    // An unpublished table entry reads as -1, which no execution may address.
    const auto tables = plan.persistent.kv.tables.page_offsets.region.bind(backing);
    CUDA_CHECK(cudaMemsetAsync(tables.data, 255, tables.bytes, device.stream()));
    if (const auto& counts = plan.persistent.buffers.token_counts)
        CUDA_CHECK(
            cudaMemsetAsync(counts->bind(backing).data, 0, counts->region.bytes, device.stream()));
    std::vector<context::ContextTableStorage> pools{{0, &kv.tables()}};
    context = std::make_unique<context::ContextStorage>(context_.span(), host_context_.span(),
                                                        pools, 0);
    const auto mapping_elements = plan.pinned_context_bytes() / 2 / sizeof(std::int64_t);
    compute_actions             = std::make_unique<context::ContextMigration>(
        device, *context, device.execution_view(), mapping_elements);
    context_actions =
        std::make_unique<context::ContextMigration>(device, *context, mapping_elements);
    device.synchronize();
}

Resources::~Resources() {
    try {
        synchronize();
    } catch (const std::logic_error&) { std::terminate(); } catch (...) {
    }
}

void Resources::synchronize() const {
    std::exception_ptr failure;
    const auto attempt = [&](auto&& action) {
        try {
            action();
        } catch (...) {
            if (!failure) failure = std::current_exception();
        }
    };
    if (compute_actions) attempt([&] { compute_actions->drain(); });
    if (context_actions) attempt([&] { context_actions->drain(); });
    attempt([&] { device_.synchronize(); });
    if (failure) std::rethrow_exception(failure);
}

Tensor Resources::tensor(const TensorRegion& region) { return region.bind(persistent()); }

Tensor Resources::packed(const TensorRegion& region, std::initializer_list<std::int32_t> shape) {
    std::uint64_t elements = 1;
    for (const auto n : shape) elements *= dimension(n);
    auto value = tensor(region);
    return value.view({dimension(value.numel())}).slice(0, 0, dimension(elements)).view(shape);
}

} // namespace ninfer::models::spark2_5
