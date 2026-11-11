#pragma once

#include "models/common/token_input.h"

#include "models/common/context/contract.h"
#include "models/common/context/directory/history.h"
#include "models/common/context/directory/state.h"
#include "models/spark2_5/execution_shape.h"
#include "ninfer/types.h"
#include "runtime/contract/resources.h"
#include "runtime/executor/contract.h"

#include <map>
#include <optional>
#include <span>
#include <variant>
#include <vector>

namespace ninfer::models::spark2_5 {

using StateBindings  = std::map<std::uint32_t, context::StateSelectors>;
using ContextActions = std::map<std::uint32_t, context::ContextActions>;

[[nodiscard]] const context::ContextActions& local_actions(const ContextActions&,
                                                           std::uint32_t rank);

// One sequence's share of an execution. Positions are sequential from main_frontier.
struct WorkRow {
    runtime::ExecutionMember member;
    StateBindings states;
    context::TableRow main_table;
    std::uint32_t main_frontier = 0;
    std::vector<TokenId> ids;
    std::optional<TokenId> control_last_token;
    // Required boundaries within ids, measured from this input window's beginning.
    std::vector<std::uint32_t> window_ends;
    ResolvedSamplingParameters sampling;
    std::uint32_t sampling_position = 0;
    bool constrained                = false;
};

enum class ForwardKind : std::uint8_t { Prefill, Replay, Ordinary, Control };
enum class SampleKind : std::uint8_t { None, FirstToken, NextToken };

struct ForwardWork {
    ForwardKind kind  = ForwardKind::Prefill;
    SampleKind sample = SampleKind::None;
    std::vector<WorkRow> rows;
};

struct ResidencyBinding {
    runtime::ExecutionMember member;
    ResolvedSamplingParameters sampling;
    TokenCounts counts;
};

struct ContextWork {
    std::vector<ResidencyBinding> bindings;
    std::vector<runtime::ResidencyId> releases;
};

struct ScoreWork {
    WorkRow row;
    std::vector<TokenId> targets;
    std::uint32_t first_target_column = 0;
};

struct Work {
    ContextActions actions;
    std::variant<ForwardWork, ContextWork, ScoreWork> operation;
};

struct RowResult {
    std::uint32_t processed = 0;
    // Exactly one sampled token for a sampling row; empty otherwise.
    std::vector<TokenId> tokens;
    std::uint32_t dead_positions = 0;
};

struct ComputeResult {
    std::vector<RowResult> rows;
};

struct ContextResult {
    std::vector<runtime::ContextTransferObservation> transfers;
};

struct ScoreResult {
    std::uint32_t processed = 0;
    std::vector<float> logprobs;
};

using Result = std::variant<ComputeResult, ContextResult, ScoreResult>;

struct CommitRow {
    runtime::ExecutionMember member;
    StateBindings states;
    std::uint32_t accepted = 0;
    bool terminal          = false;
    bool cancelled         = false;
    bool failed            = false;
};

struct Commit {
    std::vector<CommitRow> rows;
    ContextActions actions;
};

struct ExecutionTypes {
    [[nodiscard]] static ExecutionWork describe_work(const spark2_5::Work&);
    using Work   = spark2_5::Work;
    using Result = spark2_5::Result;
    using Commit = spark2_5::Commit;
};

// Value-level execution requirements shared by the coordinator and rank. Residency ownership,
// physical ranges and current device stages are checked by RankProgram.
struct WorkContract {
    ExecutionShape shape;
    std::uint32_t vocabulary, public_tokens;
    std::uint32_t control_token_limit = 0;

    [[nodiscard]] runtime::WorkPayloadCapacity payload_capacity(std::uint32_t rank_count) const;
    void validate(const Work&, runtime::FrameKind) const;
    void validate(const Work&, const Result&, const Commit&) const;
    void validate_commit(const Work&, std::span<const RowResult>, const Commit&) const;
    void validate(const Work&, const runtime::DecisionRequest&,
                  const runtime::DecisionInput&) const;
};

[[nodiscard]] std::span<const WorkRow> work_rows(const Work&);
void validate_decision_rows(const runtime::DecisionInput&, std::uint32_t stage,
                            std::uint32_t vocabulary, std::span<const std::uint32_t> positions);

} // namespace ninfer::models::spark2_5
