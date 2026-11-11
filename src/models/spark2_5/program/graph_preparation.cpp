#include "models/spark2_5/program/graph_preparation.h"

#include "core/startup.h"
#include "models/spark2_5/options.h"
#include "models/spark2_5/program/planning.h"
#include "models/spark2_5/program/program.h"

namespace ninfer::models::spark2_5 {

GraphPreparationPlan describe_graph_preparation(const EngineOptions& options) {
    GraphPreparationPlan out;
    if (!options.use_cuda_graph || options.purpose == EnginePurpose::CausalScoring) return out;
    out.enabled         = true;
    out.max_context     = options.max_context;
    out.max_concurrency = options.max_concurrency;
    out.prefill_chunk   = options.prefill_chunk;
    out.kv_cache        = resolve_kv_cache(options);
    return out;
}

GraphCalibration calibrate_graphs(const RankParameters& parameters, DeviceContext& device,
                                  const EngineOptions& options) {
    GraphCalibration result;
    result.plan = describe_graph_preparation(options);
    if (!result.plan.enabled) return result;
    StartupPhaseScope phase(options.startup_observer, StartupPhase::CudaGraphCalibration);
    // Each pass owns fresh backing, so the second pass measures the rebuild on a clean device.
    const auto probe = [&](GraphProbeBacking backing, auto&& measure) {
        auto plan  = plan_graph_probe(parameters.profile, options, device, backing);
        auto probe = std::make_unique<RankProgram>(parameters, std::move(plan), device);
        measure(*probe);
        probe.reset();
        device.synchronize();
    };
    probe(GraphProbeBacking::Inputs, [&](RankProgram& program) {
        result.memory = begin_graph_calibration(
            device, [&](GraphPreparationMode mode) { return program.prepare_execution(mode); },
            [&] { program.release_execution(); });
    });
    probe(GraphProbeBacking::StartupWorkingSet, [&](RankProgram& program) {
        complete_graph_calibration(
            result.memory, device,
            [&](GraphPreparationMode mode) { return program.prepare_execution(mode); },
            [&] { program.release_execution(); });
    });
    phase.complete();
    return result;
}

} // namespace ninfer::models::spark2_5
