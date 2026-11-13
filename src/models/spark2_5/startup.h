#pragma once

#include "models/common/startup.h"
#include "models/registry.h"
#include "models/spark2_5/program/graph_preparation.h"
#include "models/spark2_5/runtime_types.h"

namespace ninfer::models::spark2_5 {

// Spark2_5 plans nothing beyond its Program: no offloaded experts, no host tables.
struct ExecutionPreparation {
    friend bool operator==(const ExecutionPreparation&, const ExecutionPreparation&) = default;
};

// Setup contract of the local Engine startup: one resident weight placement on one device.
class Startup {
public:
    using Types         = RuntimeTypes;
    using ExecutionPlan = models::ExecutionPlan<spark2_5::ExecutionPlan, ExecutionPreparation>;
    using Calibration   = GraphCalibration;

    Startup(const artifact::Reader& definition_source, const artifact::Reader& rank_source,
            const EngineOptions&, Device);

    [[nodiscard]] Architecture architecture() const noexcept { return Architecture::Spark2_5; }

    [[nodiscard]] std::optional<ExecutionPlan> plan(DeviceContext&, const EngineOptions&,
                                                    WeightPlacement, std::size_t available,
                                                    std::size_t available_host_bytes) const;
    [[nodiscard]] std::unique_ptr<ModelShard> materialize(DeviceContext&, WeightPlacement,
                                                          const StartupObserver&);
    [[nodiscard]] Frontend frontend(const EngineOptions&) const;
    [[nodiscard]] GraphPreparationPlan graph_plan(const EngineOptions&,
                                                  const ExecutionPreparation&) const;
    [[nodiscard]] Calibration calibrate(const RankParameters&, DeviceContext&,
                                        const EngineOptions&, const ExecutionPreparation&) const;

    [[nodiscard]] std::span<const WeightCandidate> placements() const { return placements_; }

    [[nodiscard]] std::string prefill_signature(const ExecutionPreparation&,
                                                std::uint32_t chunk) const;

    [[nodiscard]] const std::shared_ptr<const ModelDefinition>& definition() const {
        return definition_;
    }

    std::shared_ptr<const ModelProfile> profile;

private:
    std::vector<WeightCandidate> placements_;
    const artifact::Reader& reader_;
    std::shared_ptr<const ModelDefinition> definition_;
};

[[nodiscard]] std::unique_ptr<RankProgram>
create_rank_program(const RankParameters&, ExecutionPlan&&, DeviceContext&, const StartupObserver&,
                    GraphPreparationMode = GraphPreparationMode::Complete);
[[nodiscard]] std::string prefill_signature(const ModelDefinition&, std::uint32_t chunk);

} // namespace ninfer::models::spark2_5
