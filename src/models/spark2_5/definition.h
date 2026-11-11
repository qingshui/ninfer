#pragma once

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "models/load_options.h"
#include "models/spark2_5/config.h"
#include "models/spark2_5/frontend/resources.h"
#include "models/spark2_5/weights.h"

#include <memory>
#include <span>
#include <string>

namespace ninfer::models::spark2_5 {

struct InstanceInfo {
    std::string name;
};

class ModelDefinition {
public:
    ~ModelDefinition();
    ModelDefinition(const ModelDefinition&)            = delete;
    ModelDefinition& operator=(const ModelDefinition&) = delete;

    [[nodiscard]] const TextConfig& config() const;
    [[nodiscard]] const LoadOptions& options() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const InstanceInfo& info() const;
    [[nodiscard]] const artifact::ArtifactId& artifact_id() const;
    [[nodiscard]] const artifact::MaterializationStats& resource_stats() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId) const;
    [[nodiscard]] std::span<const artifact::ParameterReference> parameters() const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId) const;
    [[nodiscard]] std::span<const BoundWeight> weight_data() const;
    [[nodiscard]] ops::WeightInput input(WeightId) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit ModelDefinition(std::unique_ptr<Impl>);
    friend std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader&,
                                                                  LoadOptions);
};

// Spark2_5 has no vision or speculative components; selecting one is rejected here.
[[nodiscard]] std::shared_ptr<const ModelDefinition> read_definition(const artifact::Reader&,
                                                                     LoadOptions = {});

} // namespace ninfer::models::spark2_5
