#include "models/spark2_5/session/planning.h"

#include "models/spark2_5/program/planning.h"
#include "models/spark2_5/session/internal.h"

namespace ninfer::models::spark2_5 {
namespace {
std::uint32_t count(std::uint64_t value) {
    if (value > UINT32_MAX) throw std::overflow_error("Spark2_5 context directory is too large");
    return static_cast<std::uint32_t>(value);
}

constexpr std::uint32_t kRank = 0;
} // namespace

SessionPlan plan_session(const ExecutionPlan& execution) {
    SessionPlan plan{execution.shape};
    plan.checkpoint_interval = execution.checkpoint_interval;
    const auto& image        = execution.persistent.state.images;
    const auto& pages        = execution.persistent.kv.pages;
    const auto capacity      = execution.capacity.device_bytes + execution.host_context_bytes;
    const auto states = count(capacity / image.image_bytes + execution.shape.max_concurrency * 2);
    const auto histories =
        count(std::uint64_t(states) + 3ULL * execution.shape.max_concurrency + 2);
    plan.context.ranks.push_back(
        {kRank, execution.capacity.device_bytes, execution.host_context_bytes});
    plan.context.state = {states, {{0, {{kRank, image.image_bytes, image.alignment}}}}};
    plan.context.pools.push_back({
        .pool_id = 0,
        .logical_pages =
            count(capacity / pages.page_bytes + execution.shape.max_concurrency * 2),
        .histories     = histories,
        .page_capacity = execution.persistent.kv.tables.spec.logical_page_capacity,
        .table_rows    = static_cast<std::uint32_t>(execution.persistent.kv.tables.spec.table_rows),
        .parts         = {{kRank, {pages.page_bytes, pages.alignment}}},
    });
    plan.context.metadata_budget_bytes = execution.capacity.metadata_budget_bytes;
    if (plan.context.metadata_budget_bytes)
        for (auto& pool : plan.context.pools)
            pool.histories = context::HistoryDirectory::capacity_for_metadata(
                plan.context.metadata_budget_bytes);
    return plan;
}

runtime::ContextMetadataPlan session_metadata_plan(const ExecutionPlan& execution,
                                                   const ContextPoolGeometry& geometry) {
    const auto schema = plan_session(execution).context;
    auto metadata     = context::ContextDirectory::metadata_plan(schema, kRank);
    const auto checkpoint_cost =
        detail::Checkpoints::metadata_bytes(1) - detail::Checkpoints::metadata_bytes(0);
    const auto recovery_cost =
        context::RecoveryIndex<>::metadata_bytes(1) - context::RecoveryIndex<>::metadata_bytes(0);
    std::size_t history_cost = 0;
    for (const auto& pool : schema.pools)
        history_cost += context::HistoryDirectory::metadata_bytes(1, 0, pool.page_capacity, 1) -
                        context::HistoryDirectory::metadata_bytes(0, 0, pool.page_capacity, 1);
    metadata.fixed_bytes +=
        TokenCountBuilder::capacity_bytes(execution.shape.purpose == EnginePurpose::Generation
                                              ? execution.profile->public_tokens
                                              : 0) +
        detail::Checkpoints::metadata_bytes(std::size_t(schema.state.logical_capacity)) +
        context::RecoveryIndex<>::metadata_bytes(std::size_t(schema.state.logical_capacity));
    metadata.bytes_per_record.at(geometry.spec_classes[0].index) +=
        checkpoint_cost + recovery_cost + history_cost;
    return metadata;
}

} // namespace ninfer::models::spark2_5
