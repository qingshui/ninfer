#pragma once

#include "models/spark2_5/definition.h"

#include <memory>

namespace ninfer {
struct DeviceContext;
struct StartupObserver;
} // namespace ninfer

namespace ninfer::models::spark2_5 {

// Immutable weights placed on one GPU. Borrowers finish before the backing is released.
class ModelShard {
public:
    ~ModelShard();
    ModelShard(const ModelShard&)            = delete;
    ModelShard& operator=(const ModelShard&) = delete;

    [[nodiscard]] const ModelDefinition& definition() const { return *definition_; }

    [[nodiscard]] const BoundWeight& weight(WeightId id) const { return bound_.at(id.index); }

    [[nodiscard]] ops::WeightInput input(WeightId id) const { return weight_input(bound_, id); }

    [[nodiscard]] std::span<const BoundWeight> weight_data() const { return bound_; }

    [[nodiscard]] const artifact::MaterializationStats& storage_stats() const {
        return backing_.stats();
    }

private:
    friend std::unique_ptr<ModelShard> materialize_resident(const artifact::Reader&,
                                                            std::shared_ptr<const ModelDefinition>,
                                                            DeviceContext&, const StartupObserver*);
    ModelShard(std::shared_ptr<const ModelDefinition>, artifact::MaterializedArtifact,
               std::vector<BoundWeight>);

    std::shared_ptr<const ModelDefinition> definition_;
    artifact::MaterializedArtifact backing_;
    std::vector<BoundWeight> bound_;
};

// Reads every parameter of `definition` from `reader` onto the device of `device`. The Reader
// must be the artifact that produced the definition and may be released once this returns.
[[nodiscard]] std::unique_ptr<ModelShard>
materialize_resident(const artifact::Reader& reader,
                     std::shared_ptr<const ModelDefinition> definition, DeviceContext& device,
                     const StartupObserver* observer = nullptr);

} // namespace ninfer::models::spark2_5
