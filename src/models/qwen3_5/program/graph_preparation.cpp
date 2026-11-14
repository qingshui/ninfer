#include "models/qwen3_5/program/graph_preparation.h"

#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/program/graphs.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "core/startup.h"

namespace ninfer::models::qwen3_5 {

GraphPreparationPlan describe_graph_preparation(const EngineOptions& options,
                                                const moe::Plan* moe) {
    GraphPreparationPlan out;
    if (!options.use_cuda_graph || !moe) return out;
    out.segmented = true;
    // One entry per capture bucket of the concurrency, not one per batch size: graph plans and the
    // calibration probes that measure them follow the same bucket sequence the runtime captures.
    for (const auto batch : graph_capture_buckets(options.max_concurrency)) {
        for (const auto& layer : moe->description().layers) {
            out.layers_by_batch.push_back(moe->graph_plan(layer.bank, batch));
            if (options.speculative.backend != SpeculativeBackend::None) {
                const auto width = options.speculative.draft_tokens + 1;
                out.layers_by_batch.push_back(moe->graph_plan(layer.bank, batch * width));
            }
        }
    }
    return out;
}

GraphCalibration calibrate_graphs(const execution::Parameters& parameters, DeviceContext& device,
                                  const EngineOptions& options,
                                  std::shared_ptr<const moe::Plan> moe) {
    GraphCalibration result;
    result.plan = describe_graph_preparation(options, moe.get());
    if (!options.use_cuda_graph) return result;
    // The bucket count rides the calibration phase events so startup observers see the capture
    // granularity alongside its duration; the bucket sequence itself is the public
    // graph_capture_buckets of the resolved concurrency.
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaGraphCalibration,
                            StartupProgressUnit::None,
                            graph_capture_buckets(options.max_concurrency).size());
    auto plan = detail::make_graph_probe_plan(parameters.profile, device, options, moe);
    auto probe =
        std::make_unique<detail::RankProgramImpl>(parameters, *plan, device, StartupObserver{});
    result.memory = begin_graph_calibration(
        device, [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    plan  = detail::make_graph_probe_plan(parameters.profile, device, options, moe,
                                          GraphProbeBacking::StartupWorkingSet);
    probe = std::make_unique<detail::RankProgramImpl>(parameters, *plan, device, StartupObserver{});
    complete_graph_calibration(
        result.memory, device,
        [&](GraphPreparationMode mode) { return probe->initialize_graphs(mode); },
        [&] { probe->release_graphs(); });
    probe.reset();
    device.synchronize();
    phase.complete();
    return result;
}

} // namespace ninfer::models::qwen3_5
