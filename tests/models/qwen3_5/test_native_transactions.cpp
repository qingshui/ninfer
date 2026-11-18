#include "core/device.h"
#include "models/common/qwen/frontend/frontend.h"
#include "models/common/qwen/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/execution/profile.h"
#include "native_fixture.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
namespace q35    = models::qwen3_5;
namespace native = ninfer::test::qwen3_5;

constexpr std::uint32_t kCapacity      = 512;
constexpr std::uint32_t kChunk         = 128;
constexpr std::uint32_t kPromptTokens  = 384;
constexpr std::uint32_t kPauseFrontier = 256;

void require(bool condition, std::string_view message) {
    if (!condition) { throw std::runtime_error(std::string(message)); }
}

bool transferred(const q35::ContextProgress& progress, runtime::ContextTransferDirection direction,
                 std::optional<runtime::ContextTransferResource> resource = std::nullopt) {
    return std::any_of(progress.transfers.begin(), progress.transfers.end(),
                       [&](const auto& transfer) {
                           return transfer.direction == direction &&
                                  (!resource || transfer.resource == *resource) &&
                                  transfer.work.payload_bytes != 0;
                       });
}

std::uint64_t live_bytes(const q35::PhysicalUsageSnapshot& usage, ContextMemoryDomain domain) {
    std::uint64_t result = 0;
    for (const auto& pool : usage.pools)
        if (pool.domain == domain) result += pool.live_bytes;
    return result;
}

bool same_ownership(const q35::PhysicalUsageSnapshot& a, const q35::PhysicalUsageSnapshot& b) {
    if (a.state_objects != b.state_objects || a.history_objects != b.history_objects ||
        a.kv_page_groups != b.kv_page_groups || a.pools.size() != b.pools.size() ||
        a.metadata.reserved_bytes != b.metadata.reserved_bytes)
        return false;
    for (std::size_t i = 0; i < a.pools.size(); ++i) {
        const auto& x = a.pools[i];
        const auto& y = b.pools[i];
        if (x.rank != y.rank || x.domain != y.domain || x.live_bytes != y.live_bytes ||
            x.in_use_bytes != y.in_use_bytes || x.cached_bytes != y.cached_bytes ||
            x.reserved_bytes != y.reserved_bytes ||
            x.pending_retirement_bytes != y.pending_retirement_bytes ||
            x.future_reserved_bytes != y.future_reserved_bytes ||
            x.relocation_bytes != y.relocation_bytes)
            return false;
    }
    return true;
}

// Device state a finished prefill leaves for one sequence: the Main KV positions of its page
// table row and the State image selected by its execution row.
struct PrefillImage {
    std::vector<float> k, v, recurrent, conv, hidden;
};

float bf16_value(std::uint16_t word) { return std::bit_cast<float>(std::uint32_t{word} << 16); }

float f16_value(std::uint16_t bits) {
    const int sign     = bits >> 15;
    const int exponent = (bits >> 10) & 0x1f;
    const int fraction = bits & 0x3ff;
    float magnitude    = exponent == 0 ? std::ldexp(static_cast<float>(fraction), -24)
                         : exponent == 31
                             ? (fraction ? std::nanf("") : HUGE_VALF)
                             : std::ldexp(static_cast<float>(fraction | 0x400), exponent - 25);
    return sign ? -magnitude : magnitude;
}

template <class T>
std::vector<T> read_tensor(const Tensor& tensor) {
    require(tensor.is_contiguous() && tensor.bytes() % sizeof(T) == 0, "invalid readback");
    std::vector<T> result(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(result.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return result;
}

// K as BF16 and V as FP16 words of positions [0, tokens) of every Main layer, ordered
// [layer][position][head][dim]. The KV storage of the fixture is BFloat16.
PrefillImage read_prefill_image(native::NativeRankFixture& fixture, const q35::ExecutionRow& row,
                                std::uint32_t tokens) {
    fixture.device().synchronize();
    auto& resources = fixture.rank().resources;
    PrefillImage image;
    auto& text = resources.decoder.text_kv;
    for (std::uint32_t layer = 0; layer < text.layers(); ++layer) {
        const auto view    = text.batch_layer_view(layer);
        const auto offsets = read_tensor<std::int64_t>(
            view.k_pages.records.page_offsets.slice(1, row.main_table.index, 1));
        const std::size_t head_dim = view.head_dim;
        require(!offsets.empty(), "KV table row is empty");
        const auto page_words =
            view.k_pages.tensor(offsets.front()).bytes() / sizeof(std::uint16_t);
        const auto page = static_cast<std::uint32_t>(page_words / (head_dim * view.num_kv_heads));
        for (std::uint32_t position = 0; position < tokens;) {
            const auto page_index = position / page;
            require(page_index < offsets.size(), "KV table is shorter than the sequence");
            const auto kw = read_tensor<std::uint16_t>(view.k_pages.tensor(offsets[page_index]));
            const auto vw = read_tensor<std::uint16_t>(view.v_pages.tensor(offsets[page_index]));
            for (; position < std::min(tokens, (page_index + 1) * page); ++position)
                for (int head = 0; head < view.num_kv_heads; ++head)
                    for (std::size_t d = 0; d < head_dim; ++d) {
                        const auto at = d + head_dim * (position % page + std::size_t(page) * head);
                        image.k.push_back(bf16_value(kw[at]));
                        image.v.push_back(f16_value(vw[at]));
                    }
        }
    }
    const auto offset = row.state_offsets.destination;
    auto& linear      = resources.state_images.linear();
    for (std::uint32_t layer = 0; layer < linear.layer_count(); ++layer) {
        for (float value : read_tensor<float>(linear.recurrent_tensor(layer, offset)))
            image.recurrent.push_back(value);
        for (auto word : read_tensor<std::uint16_t>(linear.conv_tensor(layer, offset)))
            image.conv.push_back(bf16_value(word));
    }
    for (auto word : read_tensor<std::uint16_t>(resources.state_images.continuation_hidden(offset)))
        image.hidden.push_back(bf16_value(word));
    return image;
}

// Relative L2 distance; a vector pair of different size or without elements never matches.
double relative_l2(const std::vector<float>& got, const std::vector<float>& reference) {
    require(got.size() == reference.size() && !got.empty(), "prefill image size mismatch");
    double difference = 0, norm = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = double(got[i]) - double(reference[i]);
        difference += d * d;
        norm += double(reference[i]) * double(reference[i]);
    }
    return std::sqrt(difference / std::max(norm, 1e-300));
}

struct SessionFixture {
    native::NativeRankFixture rank;
    q35::detail::SessionImpl session;

    SessionFixture(std::shared_ptr<const q35::ModelDefinition> definition,
                   std::shared_ptr<const q35::ModelShard> model,
                   std::shared_ptr<const q35::execution::ModelProfile> profile,
                   const EngineOptions& options, native::ContextBudget budget = {})
        : rank(std::move(definition), std::move(model), std::move(profile), options, budget),
          session(rank.definition, std::move(rank.session_plan)) {
        session.bind_executor(rank.executor());
    }
};

class Fixture {
public:
    Fixture(native::NativeRankFixture& rank, q35::detail::SessionImpl& program,
            ninfer::models::qwen::Frontend& frontend)
        : program_(program), frontend_(frontend), empty_(program.physical_usage()),
          driver_(program, rank) {}

    q35::RequestBasePlan request(std::uint32_t tokens, bool history = false,
                                 bool allow_prefix_identity = true) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front() = 1000;
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens    = 8;
        options.allow_prefix_reuse         = history;
        options.sampling.presence_penalty  = 0.5F;
        options.sampling.frequency_penalty = 0.25F;
        return program_.plan_request(
            models::qwen::PreparedPromptAccess::take(
                frontend_.prepare_tokens(std::move(ids), allow_prefix_identity)),
            options);
    }

    void synchronize() { driver_.synchronize(); }

    q35::ContextProgress settle(bool cancelled = false) {
        auto progress = driver_.settle(cancelled);
        driver_.idle();
        require(progress.complete && !program_.has_context_operation(),
                "completed context remains pending");
        return progress;
    }

    q35::SourceCandidate source_for(const q35::RequestBasePlan& base,
                                    const q35::ResumeState* resume = nullptr) {
        auto source =
            program_.inspect_source(base, resume ? resume->snapshot_handle() : std::nullopt,
                                    resume && resume->has_snapshot(), {}, {}, {0}, resume);
        require(source.has_value(), "fresh request has no root source");
        return std::move(*source);
    }

    // The asynchronous binding contract: the lane publishes through a context operation that the
    // caller settles. Synchronous bindings are covered by `bind` and the flight/fork cases.
    q35::BindingReservation start_async(const q35::RequestBasePlan& base, runtime::LaneId lane,
                                        const q35::SourceCandidate& source,
                                        q35::ResumeState* resume = nullptr,
                                        q35::ExecutionUnitKind kind = q35::ExecutionUnitKind::Decode,
                                        std::uint32_t tokens        = 1) {
        return program_.start_binding(base, lane, source, resume, kind, tokens, false);
    }

    q35::SequenceHandle bind(const q35::RequestBasePlan& base, std::uint32_t lane = 0,
                             q35::ResumeState* resume           = nullptr,
                             q35::ExecutionUnitKind resume_kind = q35::ExecutionUnitKind::Decode) {
        const bool needs_replay   = resume && !resume->has_snapshot() && resume->frontier() != 0;
        const auto expected_route = !resume                  ? GenerationRecoveryRoute::None
                                    : needs_replay           ? GenerationRecoveryRoute::Replay
                                    : resume->has_snapshot() ? GenerationRecoveryRoute::Snapshot
                                                             : GenerationRecoveryRoute::Direct;
        const auto target         = resume ? resume->frontier() : 0;
        // A root, or a binding from a Device checkpoint, completes synchronously: the lane is
        // schedulable when start_binding returns and no ContextProgress follows.
        const auto reservation    = program_.start_binding(
            base, {lane}, source_for(base, resume), resume, resume_kind,
            resume_kind == q35::ExecutionUnitKind::Prefill ? 0 : 1);
        require(static_cast<bool>(reservation), "request binding could not reserve its first unit");
        if (reservation.synchronous) {
            require(!resume, "synchronous binding carried a recovery state");
            driver_.idle();
            require(!driver_.poll_context().has_value(),
                    "synchronous binding published a context operation");
            last_binding_ = program_.sequences[lane].residency;
            return program_.sequences[lane].id;
        }
        auto progress = settle();
        require(progress.published && progress.sequence.has_value() && progress.residency,
                "binding did not publish a lane");
        last_binding_ = progress.residency;
        require(progress.recovery.route == expected_route &&
                    progress.recovery.target_frontier == target,
                "binding misreported its restoration source or target");
        if (expected_route == GenerationRecoveryRoute::Direct)
            require(progress.recovery.begin_frontier == 0 && progress.recovery.frontier == 0,
                    "direct restoration reported historical work");
        if (needs_replay) {
            require((progress.recovery.route == GenerationRecoveryRoute::Replay),
                    "snapshot-free binding did not enter replay");
        } else if (resume && !resume->has_snapshot()) {
            require(!(progress.recovery.route == GenerationRecoveryRoute::Replay),
                    "binding entered replay without committed input");
        }
        return *progress.sequence;
    }

    void reserve(q35::SequenceHandle sequence, q35::ExecutionUnitKind kind,
                 std::uint32_t tokens = 0) {
        const std::array<q35::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program_.reserve_units(units)),
                "finite native unit was not reserved");
    }

    void append_forced(q35::SequenceHandle sequence, std::span<const TokenId> tokens) {
        const std::array members{sequence};
        const std::array<std::optional<std::uint32_t>, 1> splits{std::nullopt};
        while (!tokens.empty()) {
            const auto n =
                std::min<std::size_t>(tokens.size(), program_.plan.profile->control_token_limit);
            require(n != 0, "fixture has no prepared control input");
            reserve(sequence, q35::ExecutionUnitKind::Control, n);
            (void)driver_.control(members, tokens.first(n), n, splits);
            tokens = tokens.subspan(n);
        }
    }

    void prefill_prefix(q35::SequenceHandle sequence, std::uint32_t tokens) {
        std::uint32_t processed = 0;
        while (processed < tokens) {
            reserve(sequence, q35::ExecutionUnitKind::Prefill);
            auto step = driver_.prefill(sequence);
            require(!step.complete && !step.pending, "partial prefill sampled before its target");
            require(step.processed_prompt_tokens != 0, "partial prefill made no progress");
            processed += step.processed_prompt_tokens;
        }
        require(processed == tokens, "partial prefill crossed its requested frontier");
    }

    runtime::BeginSummary finish_prefill(q35::SequenceHandle sequence, std::uint32_t remaining,
                                         bool terminal) {
        std::uint32_t processed = 0;
        for (unsigned iteration = 0; iteration < remaining / kChunk + 4; ++iteration) {
            const bool recovering = program_.recovery_pending(sequence);
            reserve(sequence, q35::ExecutionUnitKind::Prefill);
            auto step = driver_.prefill(sequence);
            processed += step.processed_prompt_tokens;
            if (!step.complete) {
                require(step.processed_prompt_tokens != 0,
                        "prefill made no observable progress");
                continue;
            }
            require(step.pending && step.pending->row_count() == 1,
                    "completed prefill did not return one pending row");
            require(!recovering || program_.recovery_pending(sequence),
                    "uncommitted prefill output released recovery protection");
            require(processed == remaining, "resumed prefill repeated or skipped committed input");
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = terminal}}};
            auto result = driver_.commit(std::move(*step.pending), decision);
            require(result.row_count == 1 && result.rows[0].disposition ==
                                                 (terminal ? runtime::CommitDisposition::Finishable
                                                           : runtime::CommitDisposition::Active),
                    "prefill commit returned the wrong request phase");
            return step.summary;
        }
        throw std::runtime_error("prefill exceeded its bounded chunk count");
    }

    void decode_and_finish(q35::SequenceHandle sequence) {
        reserve(sequence, q35::ExecutionUnitKind::Decode, 1);
        const std::array<q35::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budgets{{{.generated_tokens_remaining = 1}}};
        auto pending = driver_.decode(members, budgets);
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        auto committed = driver_.commit(std::move(pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "request could not decode after recovery");
        auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled request did not release its execution resources");
    }

    void consecutive_decode_terminal() {
        auto base           = request(64);
        const auto sequence = bind(base);
        finish_prefill(sequence, 64, false);
        reserve(sequence, q35::ExecutionUnitKind::Decode, 8);
        const std::array<q35::SequenceHandle, 1> members{sequence};
        const std::array<runtime::RoundBudget, 1> budget{{{.generated_tokens_remaining = 8}}};
        auto pending = driver_.decode(members, budget);
        const std::array<runtime::CommitDecision, 1> accepted{
            {{.accepted_tokens = pending.row_counts().empty()
                                     ? 1U
                                     : static_cast<std::uint32_t>(pending.row_counts().front())}}};
        (void)driver_.commit(std::move(pending), accepted);
        decode_and_finish(sequence);
        expect_empty("consecutive decode and terminal append leaked context");
    }

    void rewrite_segments_share_prefill_unit() {
        PromptInput input;
        ChatMessage user;
        user.role = ChatRole::User;
        user.parts.push_back({.kind = MessagePartKind::Text, .text = "Return the word green."});
        input.messages.push_back(std::move(user));
        input.options.enable_thinking = false;
        auto prompt                   = frontend_.prepare(std::move(input));
        const auto& prepared          = ninfer::models::qwen::PreparedPromptAccess::view(prompt);
        const auto tokens             = static_cast<std::uint32_t>(prepared.token_ids.size());
        require(
            tokens <= kChunk &&
                std::any_of(prepared.identity.rewrite_execution_frontiers.begin(),
                            prepared.identity.rewrite_execution_frontiers.end(),
                            [tokens](auto frontier) { return frontier > 0 && frontier < tokens; }),
            "rewrite fixture has no internal execution boundary in one nominal chunk");
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 1;
        options.allow_prefix_reuse      = false;
        auto base                       = program_.plan_request(
            models::qwen::PreparedPromptAccess::take(std::move(prompt)), options);
        const auto sequence = bind(base);
        reserve(sequence, q35::ExecutionUnitKind::Prefill);
        auto step = driver_.prefill(sequence);
        require(step.complete && step.pending && !step.capture_ready &&
                    step.processed_prompt_tokens == tokens,
                "rewrite-only segmentation yielded before the nominal prefill unit completed");
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        const auto committed = driver_.commit(std::move(*step.pending), decision);
        require(committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "segmented prefill did not commit its first output");
        const auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && !finished.checkpoint,
                "history-disabled segmented prefill retained a checkpoint");
        expect_empty("segmented prefill leaked Context resources");
    }

    q35::CheckpointHandle checkpoint(const q35::RequestBasePlan& base) {
        const auto sequence = bind(base);
        finish_prefill(sequence, base.summary().prompt_tokens, true);
        auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed && finished.checkpoint,
                "seed request did not publish its complete checkpoint");
        return *finished.checkpoint;
    }

    void expect_usage(const q35::PhysicalUsageSnapshot& expected, std::string_view label) {
        require(same_ownership(program_.physical_usage(), expected), label);
    }

    std::vector<models::context::ContextReservation>
    fill_capacity(models::context::ContextMemory memory,
                  std::optional<std::uint32_t> history = {}) {
        q35::ContextDemand demand;
        if (history)
            demand.add_history(*history, 1, memory);
        else
            demand.add_state({.count = 1, .memory = memory});
        std::vector<models::context::ContextReservation> held;
        for (;;) {
            auto next = program_.directory.reserve(demand);
            if (!next || !next.reservation->ready()) break;
            held.push_back(std::move(*next.reservation));
        }
        return held;
    }

    std::optional<q35::ContextDemotion>
    demotion(std::span<const q35::CheckpointHandle> allowed,
             std::optional<runtime::ContextTransferResource> resource = {},
             std::span<const q35::CheckpointHandle> excluded          = {}) {
        auto plan = program_.plan_reclaim(allowed, excluded, {});
        for (auto& quote : plan.demotions) {
            const auto host = program_.demotion_host_requirements(quote);
            const bool matches =
                !resource || (resource->is_state() ? host.state_images != 0
                                                   : resource->pool->index < host.kv_pages.size() &&
                                                         host.kv_pages[resource->pool->index] != 0);
            if (matches) return std::move(quote);
        }
        return {};
    }

    void demote_all(std::span<const q35::CheckpointHandle> allowed,
                    std::optional<runtime::ContextTransferResource> resource = {}) {
        while (auto quote = demotion(allowed, resource)) {
            require(program_.start_demote(*quote, {}) == runtime::ContextStart::Started,
                    "valid demotion could not reserve its Host records");
            if (program_.has_context_operation())
                require(settle().published, "demotion did not publish its replicas");
        }
    }

    void expect_empty(std::string_view label) {
        driver_.idle();
        require(!program_.has_context_operation(), "cleanup left a context operation pending");
        expect_usage(empty_, label);
    }

    void next_request() {
        auto base           = request(kChunk);
        const auto sequence = bind(base);
        finish_prefill(sequence, kChunk, false);
        decode_and_finish(sequence);
        expect_empty("next request did not return Context storage to the initial baseline");
    }

    struct MarkerRequest {
        q35::RequestBasePlan base;
        std::vector<std::uint32_t> frontiers;
    };

    // `one_window` keeps the whole prompt inside one prefill chunk, so its only window is final.
    MarkerRequest marker_request(bool one_window = false) {
        // Frontiers come from the real artifact tokenizer and public frontend markers.
        PromptInput input;
        ChatMessage system;
        system.role = ChatRole::System;
        std::string instructions;
        for (unsigned word = 0; word < 20; ++word) { instructions += "a "; }
        system.parts.push_back({.kind = MessagePartKind::Text, .text = instructions});
        input.messages.push_back(std::move(system));
        ChatMessage user;
        user.role = ChatRole::User;
        std::string content;
        for (unsigned word = 0; word < (one_window ? 8U : 350U); ++word) { content += "b "; }
        user.parts.push_back({.kind = MessagePartKind::Text, .text = content});
        input.messages.push_back(std::move(user));
        input.context_cache.allow_engine_automatic_shared_prefixes = false;
        // End before the space: byte-BPE may join that space to the following word.
        for (const auto bytes : {13U, 21U, 29U}) {
            input.context_cache.markers.push_back(
                {.kind                      = PromptCacheMarkerKind::SharedStablePrefix,
                 .evidence                  = SharedCandidateEvidence::ExplicitBoundary,
                 .location                  = PromptCacheMarkerLocation::LeadingInstructionBoundary,
                 .leading_instruction_bytes = bytes});
        }
        auto prompt          = frontend_.prepare(std::move(input));
        const auto& prepared = ninfer::models::qwen::PreparedPromptAccess::view(prompt);
        std::vector<std::uint32_t> frontiers;
        for (const auto& candidate : prepared.context_cache.opportunities) {
            frontiers.push_back(candidate.frontier);
        }
        std::sort(frontiers.begin(), frontiers.end());
        require(frontiers.size() == 3 && frontiers.front() != 0 && frontiers.back() < kChunk &&
                    std::adjacent_find(frontiers.begin(), frontiers.end()) == frontiers.end(),
                "marker fixture did not create three distinct boundaries in one chunk");
        require(one_window ? prepared.token_ids.size() <= kChunk &&
                                 prepared.token_ids.size() > frontiers.back() + 1
                           : prepared.token_ids.size() > frontiers.back() + 2 * kChunk,
                one_window ? "marker fixture does not fit one window after its boundaries"
                           : "marker fixture has insufficient suffix to exercise ordinary chunks");
        runtime::ResolvedExecutionOptions options;
        options.requested_output_tokens = 8;
        options.allow_prefix_reuse      = true;
        return {program_.plan_request(models::qwen::PreparedPromptAccess::take(std::move(prompt)),
                                      options),
                std::move(frontiers)};
    }

    // A flight carries the explicit prefix boundaries of its window, cutting its traversal at each,
    // and publishes them when it retires: no context operation, no quiet Session, at most two
    // checkpoints per request. The caller takes them once the flight retired, and a second take
    // finds nothing. A lane bound from such a checkpoint installs synchronously from the Device
    // image.
    void flight_checkpoints() {
        auto marked         = marker_request();
        const auto sequence = bind(marked.base);
        reserve(sequence, q35::ExecutionUnitKind::Prefill);
        const auto step = driver_.prefill(sequence);
        require(step.processed_prompt_tokens == kChunk && !step.complete && !step.pending &&
                    !step.capture_ready,
                "a window with checkpoints stopped at a boundary or sampled");
        require(!program_.has_context_operation(), "a flight checkpoint opened a context operation");
        const auto taken = program_.take_flight_checkpoints(sequence);
        require(taken.published.size() == 2 && taken.skipped == 0,
                "the flight did not publish the first and the last marked boundary");
        require(program_.take_flight_checkpoints(sequence).published.empty(),
                "a flight checkpoint was taken twice");
        std::vector<std::uint32_t> published;
        for (const auto point : taken.published) {
            const auto summary = program_.checkpoint_summary(point);
            require(summary.role == runtime::CheckpointRole::SharedPrefix,
                    "a flight checkpoint is not a shared prefix");
            published.push_back(summary.frontier);
        }
        require(published == std::vector<std::uint32_t>{marked.frontiers.front(),
                                                        marked.frontiers.back()},
                "the flight checkpointed boundaries other than the first and the last");

        const auto source = program_.inspect_source(marked.base, taken.published.back());
        require(source.has_value(), "a published flight checkpoint is not a binding source");
        const auto reservation = program_.start_binding(marked.base, {1}, *source);
        require(static_cast<bool>(reservation) && reservation.synchronous &&
                    reservation.sequence.has_value() && !program_.has_context_operation(),
                "a Device checkpoint source did not bind synchronously");
        driver_.idle();
        require(!driver_.poll_context().has_value(),
                "a synchronous checkpoint binding published a context operation");
        reserve(*reservation.sequence, q35::ExecutionUnitKind::Prefill);
        const auto resumed = driver_.prefill(*reservation.sequence);
        require(resumed.processed_prompt_tokens != 0 && !resumed.complete,
                "a lane bound from a checkpoint did not continue its prefill");
        require(driver_.abort(*reservation.sequence).status == runtime::ConsumeStatus::Consumed &&
                    driver_.abort(sequence).status == runtime::ConsumeStatus::Consumed,
                "checkpoint fixture retained its lanes");
        for (const auto point : taken.published)
            require(program_.release_checkpoint(point), "a flight checkpoint retained a reader");
        expect_empty("flight checkpoints leaked Context resources");
    }

    // The window that ends a request's prefill carries the checkpoints inside it like any other
    // window. Its flight publishes them when it retires, which is after the commit of its token.
    void final_window_checkpoints() {
        auto marked         = marker_request(true);
        const auto sequence = bind(marked.base);
        reserve(sequence, q35::ExecutionUnitKind::Prefill);
        auto step = driver_.prefill(sequence);
        require(step.complete && step.pending && !step.capture_ready &&
                    step.processed_prompt_tokens == marked.base.summary().prompt_tokens,
                "the only window of a request did not end its prefill");
        const auto early = program_.take_flight_checkpoints(sequence);
        require(early.published.empty() && early.skipped == 0,
                "a flight reported checkpoints before it retired");
        const std::array<runtime::CommitDecision, 1> decision{
            {{.accepted_tokens = 1, .terminal = true}}};
        const auto committed = driver_.commit(std::move(*step.pending), decision);
        require(committed.rows.size() == 1 &&
                    committed.rows[0].disposition == runtime::CommitDisposition::Finishable,
                "the final window did not commit its first output");
        const auto taken = program_.take_flight_checkpoints(sequence);
        require(taken.published.size() == 2 && taken.skipped == 0,
                "the final window did not publish the first and the last marked boundary");
        std::vector<std::uint32_t> published;
        for (const auto point : taken.published)
            published.push_back(program_.checkpoint_summary(point).frontier);
        require(published == std::vector<std::uint32_t>{marked.frontiers.front(),
                                                        marked.frontiers.back()},
                "the final window checkpointed boundaries other than the first and the last");
        const auto finished = driver_.finish(sequence);
        require(finished.status == runtime::ConsumeStatus::Consumed,
                "the final window fixture did not finish");
        if (finished.checkpoint)
            require(program_.release_checkpoint(*finished.checkpoint),
                    "the final window continuation retained a reader");
        for (const auto point : taken.published)
            require(program_.release_checkpoint(point), "a flight checkpoint retained a reader");
        expect_empty("final window checkpoints leaked Context resources");
    }

    void private_move() {
        auto base            = request(65, true);
        const auto source    = checkpoint(base);
        const auto candidate = program_.inspect_source(base, source, true);
        require(candidate && (candidate->state_action == q35::StateBindingAction::Move) &&
                    candidate->move_history,
                "exclusive continuation was not eligible for Move");
        require(static_cast<bool>(start_async(base, {0}, *candidate)),
                "exclusive continuation required a second State record");
        const auto bound = settle();
        require(bound.sequence && !program_.valid_checkpoint(source) &&
                    bound.retired_checkpoints == std::vector<q35::CheckpointHandle>{source} &&
                    bound.operations.state_moves == 1 && bound.operations.state_forks == 0 &&
                    bound.operations.partial_tail_cow_pages == 0 &&
                    !transferred(bound, runtime::ContextTransferDirection::DeviceToDevice),
                "private Move copied resources or failed to retire its consumed endpoint");
        finish_prefill(*bound.sequence, 0, true);
        const auto finished = driver_.finish(*bound.sequence);
        require(finished.checkpoint && program_.release_checkpoint(*finished.checkpoint),
                "private Move did not publish a releasable successor");
        expect_empty("private Move retained its consumed owner");
    }

    void cancel_gpu_binding() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        // An independent active writer makes the next binding require a real private tail.
        auto writer_candidate = program_.inspect_source(base, source);
        require(writer_candidate && start_async(base, {1}, *writer_candidate),
                "COW cancellation fixture could not bind its first writer");
        const auto writer = settle();
        require(writer.published && writer.sequence,
                "COW fixture did not publish its first writer");
        const auto baseline = program_.physical_usage();
        for (const bool submit_copy : {false, true}) {
            auto candidate = program_.inspect_source(base, source);
            require(candidate && candidate->reused_tokens == 65, "seed checkpoint is not reusable");
            require(static_cast<bool>(start_async(base, {0}, *candidate)),
                    "checkpoint binding did not reserve its destinations");
            std::optional<q35::ContextProgress> completed;
            if (submit_copy) {
                synchronize();
                if (auto progress = driver_.poll_context()) completed.emplace(std::move(*progress));
            }
            const auto cancelled = completed ? std::move(*completed) : settle(true);
            if (completed) {
                require(cancelled.published && cancelled.sequence &&
                            driver_.abort(*cancelled.sequence).status ==
                                runtime::ConsumeStatus::Consumed,
                        "cancellation lost a binding which completed before it arrived");
            } else {
                require(!cancelled.published && !cancelled.sequence,
                        "cancelled binding published a live sequence");
            }
            if (submit_copy) {
                require(transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                    runtime::ContextTransferResource{q35::kTextKvPool}),
                        "post-copy cancellation did not exercise a real KV tail copy");
            }
            expect_usage(baseline, "cancelled binding retained a lane, state or KV reservation");
            require(program_.checkpoint_summary(source).frontier == 65,
                    "binding cancellation destroyed its source checkpoint");
        }
        require(driver_.abort(*writer.sequence).status == runtime::ConsumeStatus::Consumed,
                "COW cancellation fixture retained its first writer");
        require(program_.release_checkpoint(source), "cancelled binding left its source pinned");
        expect_empty("binding cancellation leaked Context resources");
        next_request();
    }

    void raw_identity_opt_out() {
        auto reusable     = request(65, true);
        const auto source = checkpoint(reusable);
        require(program_.inspect_source(reusable, source).has_value(),
                "raw identity opt-out fixture has no reusable checkpoint");
        auto disabled = request(65, true, false);
        require(!program_.inspect_source(disabled, source),
                "raw identity opt-out still accepts a matching checkpoint");
        require(program_.inspect_source(disabled, std::nullopt).has_value() &&
                    !disabled.summary().publish_continuation,
                "raw identity opt-out must preserve root execution without publishing history");
        require(program_.release_checkpoint(source),
                "raw identity fixture retained its checkpoint");
        expect_empty("raw identity opt-out leaked Context resources");
    }

    void cancel_host_restore() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array<q35::CheckpointHandle, 1> allowed{source};
        demote_all(allowed);
        const auto baseline = program_.physical_usage();
        auto candidate      = program_.inspect_source(base, source);
        require(candidate.has_value(), "Host checkpoint is not a valid recovery source");
        require(static_cast<bool>(start_async(base, {1}, *candidate)),
                "Host restore did not reserve its destinations");
        // start_binding has submitted H2D. Cancellation stays asserted while poll settles it.
        const auto cancelled = settle(true);
        require(!cancelled.published && !cancelled.sequence &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextTransferResource{}) &&
                    transferred(cancelled, runtime::ContextTransferDirection::HostToDevice,
                                runtime::ContextTransferResource{q35::kTextKvPool}),
                "restore cancellation did not settle its submitted State/KV copies");
        expect_usage(baseline, "cancelled H2D restore retained unpublished destinations");
        require(program_.release_checkpoint(source),
                "cancelled H2D restore retained source leases");
        expect_empty("cancelled H2D restore leaked Context resources");
        next_request();
    }

    void cancel_and_destroy_paused_prefill() {
        for (const bool cancel_transfer : {true, false}) {
            auto base           = request(kPromptTokens);
            const auto sequence = bind(base);
            prefill_prefix(sequence, kPauseFrontier);
            require(program_.start_pause(sequence, {runtime::PauseKind::HostSnapshot, {0}}),
                    "prefill pause was not accepted");
            auto paused = settle(cancel_transfer);
            require(transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextTransferResource{}) &&
                        transferred(paused, runtime::ContextTransferDirection::DeviceToHost,
                                    runtime::ContextTransferResource{q35::kTextKvPool}),
                    "pause did not submit its complete State/KV snapshot");
            if (cancel_transfer) {
                require(!paused.published && !paused.paused,
                        "cancelled pause published a durable snapshot");
            } else {
                require(paused.published && paused.paused && paused.paused->has_snapshot() &&
                            paused.paused->frontier() == kPauseFrontier,
                        "prefill pause lost its exact recovery frontier");
                // Engine-wide active cleanup must not steal externally owned snapshot content.
                program_.fail_all_cleanup();
                require(paused.paused->has_snapshot(), "active cleanup revoked an owned snapshot");
                paused.paused.reset();
            }
            expect_empty("cancelled or destroyed paused prefill retained physical resources");
            if (cancel_transfer) next_request();
        }
    }

    void replay_pause_and_resume() {
        auto base                     = request(kPromptTokens);
        const auto original           = bind(base);
        const auto original_residency = *last_binding_;
        prefill_prefix(original, kPauseFrontier);
        require(program_.start_pause(original, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "replay-only pause was not accepted");
        auto first = settle();
        require(first.paused && !first.paused->has_snapshot() &&
                    first.paused->frontier() == kPauseFrontier,
                "snapshot-free prefill pause lost its replay target");
        expect_empty("snapshot-free pause did not release its execution resources");

        const auto replay = bind(base, 1, &*first.paused, q35::ExecutionUnitKind::Prefill);
        const auto replay_residency = *last_binding_;
        first.paused.reset();
        reserve(replay, q35::ExecutionUnitKind::Replay);
        const auto partial = driver_.replay(replay);
        require(partial.processed_tokens == kChunk && !partial.complete,
                "replay did not yield after its first finite chunk");
        require(program_.start_pause(replay, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "partially replayed request could not pause again");
        auto second = settle();
        require(second.paused && !second.paused->has_snapshot() &&
                    second.paused->frontier() == kPauseFrontier,
                "repeated replay pause replaced the original target with its partial cursor");
        expect_empty("repeated replay pause retained execution resources");

        const auto resumed = bind(base, 0, &*second.paused, q35::ExecutionUnitKind::Prefill);
        second.paused.reset();
        require(original == replay && replay == resumed && *last_binding_ != original_residency &&
                    *last_binding_ != replay_residency &&
                    last_binding_->lane == original_residency.lane &&
                    last_binding_->generation != original_residency.generation,
                "recovery changed the sequence identity or reused a retired residency");
        std::uint32_t replayed = 0;
        bool complete          = false;
        for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
            reserve(resumed, q35::ExecutionUnitKind::Replay);
            const auto progress = driver_.replay(resumed);
            replayed += progress.processed_tokens;
            require(progress.frontier == replayed, "replay progress lost its completed frontier");
            if (progress.complete) {
                complete = true;
                break;
            }
        }
        require(complete && replayed == kPauseFrontier,
                "second replay did not recover the entire original target");
        const auto summary = finish_prefill(resumed, kPromptTokens - kPauseFrontier, false);
        require(summary == runtime::BeginSummary{kPromptTokens, 0, PrefixReusePath::Root},
                "replayed prefill changed its original admission summary");
        decode_and_finish(resumed);
        expect_empty("replayed request retained physical resources after finishing");
        next_request();

        // Cancellation owns the paused request by destroying its ResumeState; no request output
        // session or resumed lane is needed merely to reclaim a snapshot-free replay record.
        const auto to_cancel = bind(base);
        prefill_prefix(to_cancel, kPauseFrontier);
        require(program_.start_pause(to_cancel, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "cancel fixture could not pause prefill");
        auto saved = settle();
        require(saved.paused.has_value(), "cancel fixture lost its paused prefill record");
        const auto rebuilding = bind(base, 1, &*saved.paused, q35::ExecutionUnitKind::Prefill);
        saved.paused.reset();
        reserve(rebuilding, q35::ExecutionUnitKind::Replay);
        require(!driver_.replay(rebuilding).complete, "cancel fixture replay completed early");
        require(program_.start_pause(rebuilding, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "cancel fixture could not pause replay");
        auto cancelled = settle();
        require(cancelled.paused && cancelled.paused->frontier() == kPauseFrontier,
                "paused replay cancellation lost its request target");
        cancelled.paused.reset();
        expect_empty("destroyed paused replay retained Context resources");
        next_request();
    }

    void pause_before_first_prefill_chunk() {
        auto base           = request(kPromptTokens);
        const auto original = bind(base);
        require(program_.start_pause(original, {runtime::PauseKind::ReleaseAndReplay, {0}}),
                "unstarted prefill pause was not accepted");
        auto paused = settle();
        require(paused.paused && !paused.paused->has_snapshot() && paused.paused->frontier() == 0,
                "unstarted prefill pause did not preserve its empty frontier");
        expect_empty("unstarted prefill pause retained execution resources");
        const auto resumed = bind(base, 1, &*paused.paused, q35::ExecutionUnitKind::Prefill);
        paused.paused.reset();
        require(resumed == original && program_.recovery_pending(resumed),
                "unstarted prefill resume changed identity or lost recovery protection");
        prefill_prefix(resumed, kChunk);
        require(!program_.recovery_pending(resumed),
                "first resumed prefill chunk did not release recovery protection");
        const auto summary = finish_prefill(resumed, kPromptTokens - kChunk, false);
        require(summary == runtime::BeginSummary{kPromptTokens, 0, PrefixReusePath::Root},
                "unstarted prefill resume changed initial prompt accounting");
        decode_and_finish(resumed);
        expect_empty("unstarted prefill resume retained resources after finishing");
        next_request();
    }

    void fork_capacity_retry() {
        auto prefix      = request(kCapacity - 1, true);
        const auto point = checkpoint(prefix);
        auto full        = request(kCapacity, true);
        const auto first = program_.inspect_source(full, point);
        require(first && start_async(full, {1}, *first),
                "fork fixture could not claim its first writer");
        const auto writer = settle();
        const auto source = program_.inspect_source(full, point);
        require(writer.sequence && source && !source->move_history,
                "fork fixture consumed the immutable source");
        auto pressure      = fill_capacity(models::context::ContextMemory::Device);
        auto host_pressure = fill_capacity(models::context::ContextMemory::Host);
        const auto before  = program_.physical_usage();
        const auto blocked = start_async(full, {0}, *source);
        require(!blocked && blocked.source_valid && !blocked.assessment.ready() &&
                    program_.valid_checkpoint(point),
                "blocked fork changed its retained source");
        expect_usage(before, "blocked fork leaked its partial reservation");
        pressure.clear();
        host_pressure.clear();
        require(static_cast<bool>(start_async(full, {0}, *source)),
                "fork failed after the competing reservation was released");
        const auto resumed = settle();
        require(resumed.sequence &&
                    driver_.abort(*resumed.sequence).status == runtime::ConsumeStatus::Consumed &&
                    driver_.abort(*writer.sequence).status == runtime::ConsumeStatus::Consumed &&
                    program_.release_checkpoint(point),
                "fork retry retained owners");
        expect_empty("fork capacity retry leaked resources");
    }

    void recovery_capacity_and_progress() {
        for (const bool snapshot : {false, true}) {
            auto base           = request(kPromptTokens);
            const auto original = bind(base);
            prefill_prefix(original, kPauseFrontier);
            require(program_.start_pause(original, {snapshot ? runtime::PauseKind::HostSnapshot
                                                             : runtime::PauseKind::ReleaseAndReplay,
                                                    {0}}),
                    "recovery fixture could not pause");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() == snapshot,
                    "recovery fixture did not retain the selected recovery mode");

            auto pressure     = fill_capacity(models::context::ContextMemory::Device, 0);
            const auto before = program_.physical_usage();
            const auto refused =
                start_async(base, {0}, source_for(base, &*paused.paused),
                                       &*paused.paused, q35::ExecutionUnitKind::Prefill, 0);
            require(!refused && !refused.assessment.ready() &&
                        paused.paused->has_snapshot() == snapshot,
                    "recovery admitted work without complete progress capacity");
            expect_usage(before, "failed recovery changed ownership");
            pressure.clear();
            const auto recovered = bind(base, 0, &*paused.paused, q35::ExecutionUnitKind::Prefill);
            paused.paused.reset();
            require(program_.recovery_pending(recovered),
                    "resumed binding lost its recovery permit");
            auto remainder             = fill_capacity(models::context::ContextMemory::Device, 0);
            const auto check_protected = [&] {
                require(program_.recovery_pending(recovered),
                        "recovery permit was released before new progress");
            };
            if (!snapshot) {
                std::uint32_t replayed = 0;
                bool complete          = false;
                for (unsigned step = 0; step < kPauseFrontier / kChunk + 1; ++step) {
                    reserve(recovered, q35::ExecutionUnitKind::Replay);
                    const auto progress = driver_.replay(recovered);
                    replayed += progress.processed_tokens;
                    check_protected();
                    if (progress.complete) {
                        complete = true;
                        break;
                    }
                }
                require(complete && replayed == kPauseFrontier,
                        "recovery failed to reconstruct its saved frontier");
            }
            finish_prefill(recovered, kPromptTokens - kPauseFrontier, false);
            require(!program_.recovery_pending(recovered),
                    "committed new prefill output did not release recovery protection");
            remainder.clear();
            decode_and_finish(recovered);
            expect_empty("completed recovery retained protected resources");

            const auto partial = bind(base);
            prefill_prefix(partial, kChunk);
            require(program_.start_pause(partial, {snapshot ? runtime::PauseKind::HostSnapshot
                                                            : runtime::PauseKind::ReleaseAndReplay,
                                                   {0}}),
                    "partial recovery could not pause");
            auto saved           = settle();
            const auto continued = bind(base, 0, &*saved.paused, q35::ExecutionUnitKind::Prefill);
            saved.paused.reset();
            if (!snapshot) {
                reserve(continued, q35::ExecutionUnitKind::Replay);
                require(driver_.replay(continued).complete,
                        "partial recovery did not reconstruct its single saved chunk");
            }
            require(program_.recovery_pending(continued),
                    "reconstructing old progress prematurely released recovery protection");
            prefill_prefix(continued, kChunk);
            require(!program_.recovery_pending(continued),
                    "new partial prefill did not release recovery protection");
            require(driver_.abort(continued).status == runtime::ConsumeStatus::Consumed,
                    "partial recovery did not release its writer");
            expect_empty("partial recovery retained resource reservations");
            next_request();
        }
    }

    void restore_own_snapshot() {
        for (const auto kind : {runtime::PauseKind::HostSnapshot, runtime::PauseKind::KeepDevice}) {
            constexpr std::uint32_t frontier = 500;
            auto base                        = request(frontier);
            const auto sequence              = bind(base);
            finish_prefill(sequence, frontier, false);
            require(program_.start_pause(sequence, {kind, {0}}),
                    "snapshot request could not save its snapshot");
            auto paused = settle();
            require(paused.paused && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier,
                    "snapshot request fell back to replay instead of saving its snapshot");
            const auto saved = program_.physical_usage();
            {
                require(static_cast<bool>(start_async(
                            base, {1}, source_for(base, &*paused.paused), &*paused.paused)),
                        "own snapshot could not reserve its complete restore");
                const auto cancelled = settle(true);
                require(
                    !cancelled.published && !cancelled.sequence && paused.paused->has_snapshot() &&
                        paused.paused->frontier() == frontier &&
                        cancelled.operations.partial_tail_cow_pages == 0 &&
                        !transferred(cancelled, runtime::ContextTransferDirection::DeviceToDevice,
                                     runtime::ContextTransferResource{q35::kTextKvPool}),
                    "cancelled own restore consumed its snapshot or copied an unnecessary tail");
                expect_usage(saved, "cancelled own restore retained unpublished records");
                for (const auto& pool : program_.physical_usage().pools)
                    require(pool.reserved_bytes == 0 && pool.pending_retirement_bytes == 0,
                            "cancelled own restore retained destination reservations");
            }
            const auto resumed = bind(base, 0, &*paused.paused);
            paused.paused.reset();
            decode_and_finish(resumed);
            expect_empty("owned snapshot restore retained its transferred checkpoint ownership");
            next_request();
        }
    }

    void independent_physical_demotions() {
        auto first_base = request(64, true), second_base = request(128, true);
        const auto first = checkpoint(first_base), second = checkpoint(second_base);
        const std::array first_owner{first}, second_owner{second};
        const auto before = program_.physical_usage();
        const auto first_quote =
            demotion(first_owner, runtime::ContextTransferResource{q35::kTextKvPool});
        const auto second_quote =
            demotion(second_owner, runtime::ContextTransferResource{q35::kTextKvPool});
        require(first_quote && second_quote, "independent owners offered no KV demotion");
        auto host_pressure        = fill_capacity(models::context::ContextMemory::Host);
        const auto blocked_before = program_.physical_usage();
        require(program_.start_demote(*first_quote, {}) != runtime::ContextStart::Started &&
                    !program_.has_context_operation() && program_.valid_checkpoint(first),
                "failed Host placement consumed its Device source");
        expect_usage(blocked_before, "failed Host placement retained a partial migration");
        host_pressure.clear();
        require(program_.start_demote(*first_quote, {}) == runtime::ContextStart::Started,
                "demotion could not retry after Host capacity was freed");
        const auto cancelled = settle(true);
        require(!cancelled.published && program_.valid_checkpoint(first) &&
                    transferred(cancelled, runtime::ContextTransferDirection::DeviceToHost,
                                runtime::ContextTransferResource{q35::kTextKvPool}),
                "cancelled D2H demotion removed its source or skipped the transfer");
        expect_usage(before, "cancelled D2H demotion retained unpublished replicas");
        const auto first_bytes = program_.directory.released_bytes(first_quote->released);
        require(program_.start_demote(*first_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "first independent demotion failed");
        require(live_bytes(before, ContextMemoryDomain::Device) ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Device) +
                        first_bytes,
                "demotion changed records outside its selected owner");
        require(program_.start_demote(*first_quote, {}) != runtime::ContextStart::Started &&
                    !program_.has_context_operation(),
                "retired demotion quote remained executable");
        require(program_.start_demote(*second_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "unrelated demotion invalidated another owner's quote");
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "demoted checkpoints retained references");
        expect_empty("independent demotions leaked resources");
    }

    void demotion_holder_changes() {
        for (const std::uint32_t tokens : {64U, 65U}) {
            auto seed         = request(tokens, true);
            const auto source = checkpoint(seed);
            const std::array source_set{source};
            const auto old_quote =
                demotion(source_set, runtime::ContextTransferResource{q35::kTextKvPool});
            require(old_quote.has_value(), "holder fixture offered no original demotion");
            std::optional<q35::SequenceHandle> writer;
            if (tokens % 64) {
                const auto first = program_.inspect_source(seed, source);
                require(first && start_async(seed, {1}, *first),
                        "holder fixture could not obtain its independent tail writer");
                const auto bound = settle();
                require(bound.sequence.has_value(), "tail writer did not publish");
                writer = bound.sequence;
            }
            auto extension    = request(tokens + 64, true);
            const auto choice = program_.inspect_source(extension, source);
            require(choice && start_async(extension, {0}, *choice),
                    "prefix fork failed");
            const auto bound = settle();
            require(bound.sequence.has_value(), "fork did not publish");
            finish_prefill(*bound.sequence, 64, true);
            const auto finished = driver_.finish(*bound.sequence);
            require(finished.checkpoint.has_value(), "fork did not retain its descendant");
            if (writer)
                require(driver_.abort(*writer).status == runtime::ConsumeStatus::Consumed,
                        "holder fixture retained its independent tail writer");
            const auto before = program_.physical_usage();
            const auto result = program_.start_demote(*old_quote, {});
            if (result == runtime::ContextStart::Started) {
                // A private COW tail remains valid; a newly co-owned full page does not.
                require(tokens % 64 != 0 && settle().published,
                        "old demotion ignored a new holder of its shared full page");
            } else {
                expect_usage(before, "stale holder rejection changed ownership");
            }
            const std::array family{source, *finished.checkpoint};
            const auto plan = program_.plan_releases(family, {}, {});
            auto selection  = plan.inventory->select();
            q35::ContextReleaseSet individual;
            for (const auto point : family) {
                const std::array one{point};
                auto probe = selection.probe(one);
                individual.merge(selection.released());
            }
            auto all            = selection.probe(family);
            const auto complete = selection.released();
            require(complete.records.size() > individual.records.size(),
                    "shared prefix became releasable before its last checkpoint holder");
            {
                auto duplicates = selection.probe(family);
                require(selection.released() == complete,
                        "duplicate checkpoint holders double-counted shared records");
            }
            const auto bytes          = program_.directory.released_bytes(complete);
            const auto before_release = program_.physical_usage();
            require(program_.release_checkpoint(source) &&
                        program_.release_checkpoint(*finished.checkpoint),
                    "holder fixture retained checkpoints");
            const auto after_release = program_.physical_usage();
            require(live_bytes(before_release, ContextMemoryDomain::Device) +
                            live_bytes(before_release, ContextMemoryDomain::Host) ==
                        live_bytes(after_release, ContextMemoryDomain::Device) +
                            live_bytes(after_release, ContextMemoryDomain::Host) + bytes,
                    "family's physical release disagreed with the unique record set");
            expect_empty("holder changes leaked resources");
        }
    }

    void release_set_union() {
        auto first_base = request(64, true), second_base = request(192, true);
        const auto first = checkpoint(first_base), second = checkpoint(second_base);
        const std::array family{first, second};
        const auto plan = program_.plan_releases(family, {}, {});
        auto selected   = plan.inventory->select();
        q35::ContextReleaseSet union_set;
        for (const auto point : family) {
            const std::array one{point};
            auto probe = selected.probe(one);
            union_set.merge(selected.released());
            probe.commit();
        }
        require(union_set == selected.released(),
                "release selection double-counted physical records");
        const auto complete = selected.released();
        {
            auto duplicate = selected.probe(family);
            require(selected.released() == complete, "repeated holders changed release accounting");
        }
        const auto bytes  = program_.directory.released_bytes(complete);
        const auto before = live_bytes(program_.physical_usage(), ContextMemoryDomain::Device);
        require(program_.release_checkpoint(first) && program_.release_checkpoint(second),
                "release batch retained checkpoint owners");
        require(before ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Device) + bytes,
                "committed release set disagrees with physical live bytes");
        expect_empty("release union leaked resources");
    }

    void physical_facts() {
        auto base         = request(65, true);
        const auto source = checkpoint(base);
        const std::array source_set{source};
        demote_all(source_set);
        const auto host_bytes = program_.host_bytes_released(source_set);
        require(host_bytes > 0, "Host source has no releasable records");
        const auto candidate = program_.inspect_source(base, source);
        require(candidate && start_async(base, {1}, *candidate),
                "Host source could not bind");
        require(program_.host_bytes_released(source_set) == 0 &&
                    program_.plan_releases(source_set, {}, {}).releases.empty() &&
                    !program_.release_checkpoint(source),
                "pending source reader became reclaimable");
        require(!settle(true).published && program_.valid_checkpoint(source),
                "cancelled restore removed its immutable source");
        require(program_.host_bytes_released(source_set) == host_bytes,
                "retirement did not release the source borrow");
        require(program_.release_checkpoint(source), "Host source remained pinned");
        expect_empty("source-borrow facts leaked resources");

        auto seed           = request(64, true);
        const auto ancestor = checkpoint(seed);
        const std::array ancestor_set{ancestor};
        const auto main_quote =
            demotion(ancestor_set, runtime::ContextTransferResource{q35::kTextKvPool});
        require(main_quote &&
                    program_.start_demote(*main_quote, {}) == runtime::ContextStart::Started &&
                    settle().published,
                "shared Host fixture could not back up its prefix");
        const auto shared_host_bytes = program_.host_bytes_released(ancestor_set);
        require(shared_host_bytes == main_quote->host_bytes && shared_host_bytes != 0,
                "Host backup disagreed with its allocation quote");
        auto extension    = request(128, true);
        const auto choice = program_.inspect_source(extension, ancestor);
        require(choice && start_async(extension, {1}, *choice),
                "shared Host fixture could not bind a descendant");
        const auto bound = settle();
        require(bound.sequence && program_.host_bytes_released(ancestor_set) == 0 &&
                    program_.release_redundant_host(ancestor_set) == 0,
                "active descendant or exclusion failed to protect its Host prefix");
        finish_prefill(*bound.sequence, 64, true);
        const auto finished = driver_.finish(*bound.sequence);
        require(finished.checkpoint.has_value(), "shared Host descendant was not retained");
        const auto descendant = *finished.checkpoint;
        const std::array family{ancestor, descendant};
        const std::array repeated{ancestor, descendant, ancestor};
        const std::array descendant_set{descendant};
        require(program_.host_bytes_released(ancestor_set) == 0 &&
                    program_.host_bytes_released(family) == shared_host_bytes &&
                    program_.host_bytes_released(repeated) == shared_host_bytes,
                "Host accounting double-counted a shared or repeated checkpoint holder");
        program_.recovery_index.synchronize(models::qwen::RecoveryStoreView{program_.checkpoints});
        auto recovery = program_.recovery_index.evaluate(ancestor_set);
        require(recovery.loss(descendant_set) == 64 && recovery.loss(family) == 128 &&
                    recovery.loss(descendant_set) == 64,
                "recovery loss depended on a previous hypothetical release");
        const auto protected_plan = program_.plan_reclaim(family, ancestor_set, {});
        require(std::all_of(
                    protected_plan.demotions.begin(), protected_plan.demotions.end(),
                    [&](const auto& quote) { return quote.sources == std::vector{descendant}; }),
                "an excluded ancestor was exposed through a descendant alias");

        const auto baseline = program_.physical_usage();
        const auto plan     = program_.plan_reclaim(family, {}, {});
        const auto redundant =
            std::find_if(plan.demotions.begin(), plan.demotions.end(), [&](const auto& quote) {
                return quote.sources.size() == family.size() && quote.host_bytes == 0;
            });
        require(redundant != plan.demotions.end() &&
                    program_.start_demote(*redundant, {}) == runtime::ContextStart::Started,
                "current Host coverage did not permit releasing its redundant Device replica");
        if (program_.has_context_operation())
            require(settle().published, "redundant replica release failed");
        require(live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) ==
                        shared_host_bytes &&
                    program_.checkpoint_matches(ancestor, seed) &&
                    program_.checkpoint_matches(descendant, extension),
                "redundant Device release invalidated complete Host-backed histories");
        const auto mixed = program_.inspect_source(extension, descendant);
        require(mixed && std::count_if(
                             mixed->transfers.begin(), mixed->transfers.end(),
                             [](const auto& transfer) {
                                 return transfer.resource ==
                                            runtime::ContextTransferResource{q35::kTextKvPool} &&
                                        transfer.direction ==
                                            runtime::ContextTransferDirection::HostToDevice &&
                                        transfer.page_groups == 1;
                             }) == 1,
                "mixed Device/Host recovery did not request exactly its missing prefix record");
        require(static_cast<bool>(start_async(extension, {0}, *mixed)),
                "mixed Device/Host checkpoint could not restore");
        const auto restored = settle();
        require(restored.sequence &&
                    driver_.abort(*restored.sequence).status == runtime::ConsumeStatus::Consumed,
                "mixed Device/Host restore retained its writer");
        expect_usage(baseline, "mixed restore retained an extra destination");
        require(program_.release_checkpoint(ancestor) &&
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) ==
                        shared_host_bytes &&
                    program_.host_bytes_released(descendant_set) == shared_host_bytes &&
                    program_.release_checkpoint(descendant),
                "shared Host record was not owned through its last checkpoint holder");
        expect_empty("shared Host accounting leaked resources");

        auto active_base  = request(kPromptTokens);
        const auto active = bind(active_base);
        prefill_prefix(active, kPauseFrontier);
        require(program_.start_pause(active, {runtime::PauseKind::HostSnapshot, {0}}),
                "snapshot facts could not pause");
        auto paused = settle();
        require(paused.paused && paused.paused->snapshot_handle(), "pause lost snapshot identity");
        const auto handle = *paused.paused->snapshot_handle();
        const std::array handles{handle};
        const auto paused_bytes  = program_.host_bytes_released(handles);
        const auto before_revoke = live_bytes(program_.physical_usage(), ContextMemoryDomain::Host);
        require(paused_bytes != 0, "paused snapshot lost its Host release footprint");
        require(program_.revoke_snapshot(*paused.paused) && !paused.paused->snapshot_handle(),
                "revocation retained snapshot ownership");
        require(before_revoke ==
                    live_bytes(program_.physical_usage(), ContextMemoryDomain::Host) + paused_bytes,
                "paused snapshot release footprint disagrees with revocation");
        paused.paused.reset();
        expect_empty("snapshot revocation leaked resources");
    }

    void grammar_row_failure() {

        auto base        = request(32);
        const auto first = bind(base, 0);
        finish_prefill(first, 32, false);
        const auto second = bind(base, 1);
        finish_prefill(second, 32, false);
        const std::array<q35::SequenceHandle, 2> members{first, second};
        const std::array<q35::ExecutionUnit, 2> units{
            {{first, q35::ExecutionUnitKind::Decode, 1},
             {second, q35::ExecutionUnitKind::Decode, 1}}};
        require(static_cast<bool>(program_.reserve_units(units)), "grammar mixed unit reservation");
        const std::array<runtime::RoundBudget, 2> budgets{{{1}, {1}}};
        const std::array constrained{true, false};
        auto pending = driver_.decode(
            members, budgets, constrained, [](const runtime::DecisionRequest& request) {
                runtime::DecisionInput input{.stage = request.stage};
                const auto stride = (request.vocabulary + 31U) / 32U;
                for (const auto& row : request.rows) {
                    runtime::DecisionInput::Row mask;
                    mask.member         = row.member;
                    mask.positions      = row.drafts.size() + 1;
                    mask.dead_positions = 1;
                    mask.words.resize(stride * mask.positions, 0);
                    for (std::size_t column = 0; column < mask.positions; ++column)
                        mask.words[column * stride] = 1;
                    input.rows.push_back(std::move(mask));
                }
                return input;
            });
        require(pending.constraint_failed(0) && !pending.constraint_failed(1),
                "grammar failure lost its row");
        const std::array<runtime::CommitDecision, 2> decisions{
            {{.terminal = true, .failed = true}, {.accepted_tokens = 1}}};
        auto committed = driver_.commit(std::move(pending), decisions);
        require(committed.rows[0].disposition == runtime::CommitDisposition::FailedReleased &&
                    committed.rows[1].disposition == runtime::CommitDisposition::Active,
                "grammar failure contaminated a healthy row");
        require(driver_.abort(second).status == runtime::ConsumeStatus::Consumed,
                "healthy row was released by another row's grammar error");
        expect_empty("grammar row failure leaked physical resources");
    }

    // A Prefill unit capped by ExecutionUnit::tokens covers at most that many prompt positions,
    // and the KV and State it leaves equal an uncapped continuous execution of the same prompt
    // within the numerical effect of cutting a sequence into different windows (the packed
    // prefill test holds the same budget). A wrong window, position or State would be of order one.
    void capped_prefill_matches_uncapped(native::NativeRankFixture& rank) {
        constexpr std::uint32_t kCap  = 40; // neither a chunk nor a GDN block multiple
        constexpr double kRecutBudget = 0.12;

        struct Run {
            std::vector<std::uint32_t> windows;
            PrefillImage image;
        };

        const auto execute = [&](std::uint32_t lane, std::uint32_t cap) {
            // Distinct token ids: a repeated token makes every position alike and the comparison
            // insensitive to a misplaced window.
            std::vector<TokenId> ids(kPromptTokens);
            for (std::uint32_t i = 0; i < kPromptTokens; ++i)
                ids[i] = static_cast<TokenId>(1000 + (i * 40503ULL + i / 7 * 977ULL) % 200000);
            runtime::ResolvedExecutionOptions options;
            options.requested_output_tokens = 1;
            options.allow_prefix_reuse      = false;
            // No sibling shares this prompt, so its prefill runs to the end.
            const auto base                 = program_.plan_request(
                models::qwen::PreparedPromptAccess::take(frontend_.prepare_tokens(ids, false)),
                options);
            const auto sequence = bind(base, lane);
            Run run;
            std::optional<q35::ExecutionRow> row;
            rank.observer().submitted = [&](const auto& submitted) {
                const auto* work = std::get_if<q35::ForwardWork>(&submitted.work.operation);
                if (!work || work->purpose != q35::ForwardPurpose::Prefill) return;
                require(work->rows.size() == 1 && work->inputs.size() == 1,
                        "single prefill submitted several segments");
                row = work->rows.front();
                run.windows.push_back(
                    static_cast<std::uint32_t>(work->inputs.front().tokens.size()));
            };
            std::uint32_t processed = 0;
            q35::PrefillRowProgress step;
            do {
                reserve(sequence, q35::ExecutionUnitKind::Prefill, cap);
                step = driver_.prefill(sequence);
                require(step.processed_prompt_tokens != 0 &&
                            (cap == 0 || step.processed_prompt_tokens <= cap),
                        "capped prefill unit processed more than its cap or nothing");
                processed += step.processed_prompt_tokens;
                require(processed <= kPromptTokens, "prefill crossed the prompt end");
            } while (!step.complete);
            rank.observer().submitted = nullptr;
            require(processed == kPromptTokens && row && step.pending &&
                        step.pending->row_count() == 1,
                    "prefill did not cover the prompt exactly once");
            run.image = read_prefill_image(rank, *row, kPromptTokens);
            const std::array<runtime::CommitDecision, 1> decision{
                {{.accepted_tokens = 1, .terminal = true}}};
            (void)driver_.commit(std::move(*step.pending), decision);
            require(driver_.finish(sequence).status == runtime::ConsumeStatus::Consumed,
                    "finished prefill was not consumed");
            expect_empty("capped prefill leaked Context resources");
            return run;
        };
        const auto continuous = execute(0, 0);
        const auto capped     = execute(1, kCap);
        require(continuous.windows.size() == kPromptTokens / kChunk &&
                    std::all_of(continuous.windows.begin(), continuous.windows.end(),
                                [](auto tokens) { return tokens == kChunk; }),
                "uncapped prefill did not run whole chunks");
        std::uint32_t covered = 0;
        for (const auto tokens : capped.windows) {
            require(tokens != 0 && tokens <= kCap, "executed window exceeds its cap");
            covered += tokens;
        }
        require(covered == kPromptTokens && capped.windows.size() >= kPromptTokens / kCap,
                "capped windows do not tile the prompt");
        const std::pair<const char*,
                        std::pair<const std::vector<float>&, const std::vector<float>&>>
            fields[]{{"K", {capped.image.k, continuous.image.k}},
                     {"V", {capped.image.v, continuous.image.v}},
                     {"recurrent State", {capped.image.recurrent, continuous.image.recurrent}},
                     {"conv State", {capped.image.conv, continuous.image.conv}},
                     {"last-token hidden", {capped.image.hidden, continuous.image.hidden}}};
        bool within = true;
        for (const auto& [name, pair] : fields) {
            const auto distance = relative_l2(pair.first, pair.second);
            std::cout << "  capped prefill " << name << ": rel_l2=" << distance << '\n';
            within = within && distance <= kRecutBudget;
        }
        require(within, "capped prefill diverged from continuous execution");
    }

private:
    q35::detail::SessionImpl& program_;
    ninfer::models::qwen::Frontend& frontend_;
    q35::PhysicalUsageSnapshot empty_;
    native::EventDriver<q35::detail::SessionImpl> driver_;
    std::optional<runtime::ResidencyId> last_binding_;
};

void replay_sampling_counts(std::shared_ptr<const q35::ModelDefinition> definition,
                            std::shared_ptr<const q35::ModelShard> model,
                            std::shared_ptr<const q35::execution::ModelProfile> profile,
                            ninfer::models::qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 128;
    options.max_concurrency                   = 2;
    options.context_cache.enabled             = false;
    options.context_cache.host_capacity_bytes = 0;
    native::NativeRankFixture rank(definition, model, profile, options);
    q35::detail::SessionImpl program(definition, std::move(rank.session_plan));
    program.bind_executor(rank.executor());
    native::EventDriver driver(program, rank);

    // This exact oracle counts committed output events, independently of the execution route.
    // The prompt token appears many times but contributes no occurrences of its own.
    const std::vector<TokenId> prompt(32, 200);
    auto prepared = frontend.prepare_tokens(prompt);
    runtime::ResolvedExecutionOptions request;
    request.requested_output_tokens    = 8;
    request.allow_prefix_reuse         = false;
    request.sampling.presence_penalty  = 0.5F;
    request.sampling.frequency_penalty = 0.25F;
    auto base                          = program.plan_request(
        ninfer::models::qwen::PreparedPromptAccess::take(std::move(prepared)), request);

    const auto settle = [&]() { return driver.settle(); };
    const auto bind   = [&](std::uint32_t lane, q35::ResumeState* resume) {
        auto source = program.inspect_source(base, std::nullopt);
        require(source && static_cast<bool>(program.start_binding(
                              base, {lane}, *source, resume, q35::ExecutionUnitKind::Decode, 1, false)),
                  "sampling-count fixture could not bind its lane");
        auto progress = settle();
        require(progress.published && progress.sequence,
                  "sampling-count fixture did not publish a sequence");
        require(!resume || (progress.recovery.route == GenerationRecoveryRoute::Replay),
                "sampling-count resume did not enter replay");
        return *progress.sequence;
    };
    const auto reserve = [&](q35::SequenceHandle sequence, q35::ExecutionUnitKind kind,
                             std::uint32_t tokens = 0) {
        const std::array<q35::ExecutionUnit, 1> units{{{sequence, kind, tokens}}};
        require(static_cast<bool>(program.reserve_units(units)),
                "sampling-count fixture could not reserve a finite unit");
    };

    const auto initial = bind(0, nullptr);
    reserve(initial, q35::ExecutionUnitKind::Prefill);
    auto begin = driver.prefill(initial);
    require(begin.complete && begin.pending && begin.pending->tokens().size() == 1,
            "sampling-count fixture did not produce exactly one Begin token");
    const TokenId first = begin.pending->tokens().front();
    const std::array<runtime::CommitDecision, 1> accepted{{{.accepted_tokens = 1}}};
    const auto committed = driver.commit(std::move(*begin.pending), accepted);
    require(committed.rows[0].disposition == runtime::CommitDisposition::Active,
            "sampling-count fixture did not commit Begin");

    const std::array<TokenId, 3> forced{198, 198, 199};
    const std::array<q35::SequenceHandle, 1> members{initial};
    const std::array<std::optional<std::uint32_t>, 1> splits{2U};
    reserve(initial, q35::ExecutionUnitKind::Control, forced.size());
    (void)driver.control(members, forced, forced.size(), splits);

    const auto vocabulary = static_cast<std::int32_t>(profile->public_tokens);
    std::vector<std::int32_t> expected_counts(static_cast<std::size_t>(vocabulary), 0);
    ++expected_counts.at(static_cast<std::size_t>(first));
    for (const TokenId token : forced) { ++expected_counts.at(static_cast<std::size_t>(token)); }
    std::vector<TokenId> expected_ledger = prompt;
    expected_ledger.push_back(first);
    expected_ledger.insert(expected_ledger.end(), forced.begin(), forced.end());
    const std::uint32_t expected_frontier = static_cast<std::uint32_t>(expected_ledger.size() - 1U);
    const std::array<std::uint32_t, 1> expected_splits{
        static_cast<std::uint32_t>(prompt.size() + 1U + *splits.front())};

    const auto check = [&](std::uint32_t lane, std::uint32_t state_frontier,
                           std::string_view phase) {
        std::vector<std::int32_t> counts(expected_counts.size());
        const auto source = rank.rank()
                                .resources.token_counts.slice(1, static_cast<std::int32_t>(lane), 1)
                                .view({vocabulary});
        rank.device().synchronize();
        CUDA_CHECK(cudaMemcpy(counts.data(), source.data, source.bytes(), cudaMemcpyDeviceToHost));
        require(counts == expected_counts,
                std::string(phase) + ": sampling counts differ from committed output occurrences");
        const auto& sequence     = program.sequences[lane];
        const auto actual_splits = sequence.prefix_identity.execution_frontiers();
        require(sequence.ledger == expected_ledger &&
                    sequence.continuation.metadata.execution_frontier == state_frontier &&
                    sequence.continuation.metadata.text_kv_valid == state_frontier &&
                    sequence.prefix_identity.size() == expected_ledger.size() &&
                    actual_splits.size() == expected_splits.size() &&
                    std::equal(actual_splits.begin(), actual_splits.end(), expected_splits.begin()),
                std::string(phase) +
                    ": replay changed committed ledger, E/S or forced execution split");
    };
    check(0, expected_frontier, "after forced control");
    require(program.start_pause(initial, {runtime::PauseKind::ReleaseAndReplay, {0}}),
            "sampling-count fixture could not pause");
    auto paused = settle();
    require(paused.paused && !paused.paused->has_snapshot() &&
                paused.paused->frontier() == expected_frontier,
            "sampling-count fixture did not retain its committed replay target");
    const auto resumed = bind(1, &*paused.paused);
    paused.paused.reset();
    check(1, 0, "after lane rebinding");
    bool recovered         = false;
    std::uint32_t replayed = 0;
    for (unsigned step = 0; step < 3; ++step) {
        reserve(resumed, q35::ExecutionUnitKind::Replay);
        const auto progress = driver.replay(resumed);
        replayed += progress.processed_tokens;
        check(1, replayed, "after replay chunk");
        if (progress.complete) {
            recovered = true;
            break;
        }
    }
    require(recovered && replayed == expected_frontier,
            "sampling-count fixture did not finish replaying its execution frontier");
    require(driver.abort(resumed).status == runtime::ConsumeStatus::Consumed,
            "sampling-count fixture could not release its restored lane");
    const auto released = program.physical_usage();
    require(released.state_objects == 0 && released.history_objects == 0 &&
                std::all_of(released.pools.begin(), released.pools.end(),
                            [](const auto& pool) {
                                return pool.live_bytes == 0 && pool.reserved_bytes == 0 &&
                                       pool.pending_retirement_bytes == 0;
                            }),
            "sampling-count fixture leaked Context resources");
}

// Final windows pack with other segments and mix with decode rows; a root binding installs
// synchronously without publishing a context operation, and a sample fork makes the parent branch 0
// and installs children that share the rested parent's whole-page KV, all decoding the last prompt
// token from the same flight.
void sample_fork_and_packed_finals(std::shared_ptr<const q35::ModelDefinition> definition,
                                   std::shared_ptr<const q35::ModelShard> model,
                                   std::shared_ptr<const q35::execution::ModelProfile> profile,
                                   ninfer::models::qwen::Frontend& frontend, EngineOptions options) {
    options.max_context                       = 256;
    options.max_concurrency                   = 4;
    options.context_cache.enabled             = false;
    options.context_cache.host_capacity_bytes = 0;
    native::NativeRankFixture rank(definition, model, profile, options);
    q35::detail::SessionImpl program(definition, std::move(rank.session_plan));
    program.bind_executor(rank.executor());
    native::EventDriver driver(program, rank);
    const auto empty = program.physical_usage();
    const auto stage = [](const char* label) { std::cerr << "stage: " << label << '\n'; };

    const auto plan = [&](std::uint32_t tokens, float penalty, std::uint32_t samples = 1) {
        std::vector<TokenId> ids(tokens, 198);
        ids.front()                              = 1000;
        runtime::ResolvedExecutionOptions request;
        request.requested_output_tokens          = 8;
        request.sampling.presence_penalty        = penalty;
        request.sampling.frequency_penalty       = penalty;
        request.sample_count                     = samples;
        return program.plan_request(
            ninfer::models::qwen::PreparedPromptAccess::take(
                frontend.prepare_tokens(std::move(ids), true)),
            request);
    };
    // Every root binding completes synchronously, with or without penalties: its installation and
    // State clearing ride the stream order of the lane's first flight.
    const auto bind_root = [&](const q35::RequestBasePlan& base, std::uint32_t lane) {
        auto source = program.inspect_source(base, std::nullopt);
        require(source.has_value(), "fork fixture has no root source");
        auto reservation = program.start_binding(base, runtime::LaneId{lane}, *source, nullptr,
                                                 q35::ExecutionUnitKind::Prefill, 0);
        require(static_cast<bool>(reservation), "root binding did not reserve its lane");
        require(reservation.synchronous && reservation.sequence.has_value(),
                "root binding was asynchronous");
        require(!program.has_context_operation(), "synchronous root binding opened a context operation");
        driver.idle();
        require(!driver.poll_context().has_value(),
                "synchronous root binding published a context operation");
        return program.sequences[lane].id;
    };
    const auto reserve = [&](std::span<const q35::ExecutionUnit> units) {
        require(static_cast<bool>(program.reserve_units(units)),
                "fork fixture could not reserve its execution units");
    };
    const auto prefill_window = [&](q35::SequenceHandle sequence, std::uint32_t tokens) {
        const std::array<q35::ExecutionUnit, 1> units{{{sequence, q35::ExecutionUnitKind::Prefill}}};
        reserve(units);
        auto step = driver.prefill(sequence);
        require(!step.complete && !step.pending && step.processed_prompt_tokens == tokens,
                "partial prefill did not cover exactly its requested window");
        driver.idle();
    };

    // The parent of a sample group rests at the fork frontier instead of sampling, with or without
    // penalties: penalty counts cover generated tokens only, so every branch starts empty.
    stage("penalized parent");
    const std::uint32_t prompt = 64;
    const auto unshared_base  = plan(prompt, 0.5F);
    require(unshared_base.fork_frontier() == 0, "a prompt nobody shares gained a fork frontier");
    const auto parent_base    = plan(prompt, 0.5F, 3);
    require(parent_base.fork_frontier() == prompt - 1,
            "sample group member lost its fork frontier");
    require(program.sample_fork_limit() == 4, "sample fork limit does not cover every free lane");
    const auto parent = bind_root(parent_base, 0);
    prefill_window(parent, prompt - 1);
    const std::array<q35::ExecutionUnit, 1> rested{{{parent, q35::ExecutionUnitKind::Prefill}}};
    require(!static_cast<bool>(program.reserve_units(rested)),
            "fork parent reserved a window past its resting frontier");
    // Forking a parent that has not rested is rejected before any child lane changes.
    const auto early_base = plan(prompt, 0.0F, 3);
    const auto early      = bind_root(early_base, 1);
    bool rejected         = false;
    try {
        const std::array<q35::ForkGroupRequest, 1> early_group{
            {{.parent = early, .children = {runtime::LaneId{2}, runtime::LaneId{3}}}}};
        (void)program.fork_batch(early_group);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "fork accepted a parent that did not rest at its frontier");
    require(driver.abort(early).status == runtime::ConsumeStatus::Consumed,
            "rejected fork fixture retained its lane");

    // The parent is branch 0: the group decodes three rows, the parent and both children, from
    // the same flight, with no context operation and no helper lane.
    stage("fork batch");
    const std::array<q35::ForkGroupRequest, 1> group{
        {{.parent = parent, .children = {runtime::LaneId{2}, runtime::LaneId{3}}}}};
    const auto forked = program.fork_batch(group);
    require(forked.size() == 1 && forked[0].reserved && forked[0].children.size() == 2,
            "fork did not reserve the requested children");
    require(!program.has_context_operation(), "fork opened a context operation");
    driver.idle();
    require(!driver.poll_context().has_value(), "fork published a context operation");
    const std::array<q35::SequenceHandle, 3> members{parent, forked[0].children[0],
                                                     forked[0].children[1]};
    const std::array<q35::ExecutionUnit, 3> decode_units{
        {{members[0], q35::ExecutionUnitKind::Decode, 1},
         {members[1], q35::ExecutionUnitKind::Decode, 1},
         {members[2], q35::ExecutionUnitKind::Decode, 1}}};
    reserve(decode_units);
    const std::array<runtime::RoundBudget, 3> budgets{{{.generated_tokens_remaining = 1},
                                                       {.generated_tokens_remaining = 1},
                                                       {.generated_tokens_remaining = 1}}};
    auto pending = driver.decode(members, budgets);
    require(pending.row_count() == 3 && pending.tokens().size() == 3,
            "fork branches did not decode one token each");
    const std::array<runtime::CommitDecision, 3> terminal3{{{.accepted_tokens = 1, .terminal = true},
                                                            {.accepted_tokens = 1, .terminal = true},
                                                            {.accepted_tokens = 1, .terminal = true}}};
    const auto committed = driver.commit(std::move(pending), terminal3);
    require(committed.row_count == 3 &&
                std::all_of(committed.rows.begin(), committed.rows.end(),
                            [](const auto& row) {
                                return row.disposition == runtime::CommitDisposition::Finishable;
                            }),
            "fork branches did not reach a finishable phase");
    for (const auto branch : members) {
        const auto finished = driver.finish(branch);
        require(finished.status == runtime::ConsumeStatus::Consumed,
                "fork branch did not release its execution resources");
    }
    require(same_ownership(program.physical_usage(), empty),
            "sample fork leaked Context resources");

    // A group without children only moves its parent to decoding.
    stage("fork without children");
    const auto lone_base = plan(prompt, 0.0F, 2);
    const auto lone      = bind_root(lone_base, 0);
    prefill_window(lone, prompt - 1);
    const std::array<q35::ForkGroupRequest, 1> lone_group{{{.parent = lone}}};
    const auto lone_result = program.fork_batch(lone_group);
    require(lone_result.size() == 1 && lone_result[0].reserved && lone_result[0].children.empty(),
            "a group without children was not accepted");
    require(driver.abort(lone).status == runtime::ConsumeStatus::Consumed,
            "childless fork parent retained its lane");
    require(same_ownership(program.physical_usage(), empty),
            "childless fork leaked Context resources");

    const std::array<runtime::CommitDecision, 2> terminal2{
        {{.accepted_tokens = 1, .terminal = true}, {.accepted_tokens = 1, .terminal = true}}};
    // Two final windows pack into one prefill; each final segment carries one decision line.
    stage("packed finals");
    const std::uint32_t long_prompt = 200, short_prompt = 64;
    const auto long_base             = plan(long_prompt, 0.5F);
    const auto short_base            = plan(short_prompt, 0.5F);
    const auto pack_long             = bind_root(long_base, 0);
    const auto pack_short            = bind_root(short_base, 1);
    prefill_window(pack_long, 128);
    const std::array<q35::SequenceHandle, 2> pack{pack_long, pack_short};
    const std::array<q35::ExecutionUnit, 2> pack_units{
        {{pack_long, q35::ExecutionUnitKind::Prefill},
         {pack_short, q35::ExecutionUnitKind::Prefill}}};
    reserve(pack_units);
    const auto packed = program.start_prefill(pack, {});
    require(packed.has_value() && packed->segments.size() == 2 &&
                packed->segments[0].produces_token && packed->segments[1].produces_token,
            "two final windows did not pack with one decision line each");
    auto packed_progress = driver.wait<q35::PrefillProgress>(packed->ticket);
    require(packed_progress.rows.size() == 2 && packed_progress.rows[0].complete &&
                packed_progress.rows[1].complete,
            "packed final windows did not complete together");
    require(packed_progress.rows[0].pending &&
                packed_progress.rows[0].pending->row_count() == 2 &&
                !packed_progress.rows[1].pending.has_value(),
            "packed prefill did not carry one decision line per final segment");
    const std::array<runtime::CommitDecision, 2> accepted{{{.accepted_tokens = 1},
                                                            {.accepted_tokens = 1}}};
    const auto packed_commit = driver.commit(std::move(*packed_progress.rows[0].pending), accepted);
    require(packed_commit.row_count == 2, "packed prefill commit lost a decision row");

    // A final window mixes with decode rows; the batch carries the decode row first, then the
    // producing segment.
    stage("mixed final window");
    const auto mixed_base = plan(long_prompt, 0.5F);
    const auto mixed_seq  = bind_root(mixed_base, 2);
    prefill_window(mixed_seq, 128);
    const std::array<q35::ExecutionUnit, 2> mixed_units{
        {{mixed_seq, q35::ExecutionUnitKind::Prefill},
         {pack_short, q35::ExecutionUnitKind::Decode, 1}}};
    reserve(mixed_units);
    const std::array<q35::SequenceHandle, 1> mixed_prefill{mixed_seq};
    const std::array<q35::SequenceHandle, 1> mixed_decode{pack_short};
    const std::array<runtime::RoundBudget, 1> mixed_budgets{{{.generated_tokens_remaining = 1}}};
    const auto mixed =
        program.start_mixed(mixed_prefill, {}, mixed_decode, mixed_budgets, {});
    require(mixed.has_value() && mixed->segments.size() == 1 &&
                mixed->segments[0].produces_token && mixed->decode_rows == 1,
            "final window did not mix with decode rows");
    auto mixed_progress = driver.wait<q35::MixedProgress>(mixed->ticket);
    require(mixed_progress.prefill.rows.size() == 1 && mixed_progress.prefill.rows[0].complete,
            "mixed prefill segment did not complete");
    require(mixed_progress.decode.row_count() == 2,
            "mixed batch lost the producing segment or the decode row");
    const auto mixed_commit =
        driver.commit(std::move(mixed_progress.decode),
                      std::span<const runtime::CommitDecision>{terminal2});
    require(mixed_commit.row_count == 2, "mixed commit lost a decision row");
    for (const auto sequence : {mixed_seq, pack_short}) {
        require(driver.finish(sequence).status == runtime::ConsumeStatus::Consumed,
                "mixed execution did not finish its sequences");
    }
    require(driver.abort(pack_long).status == runtime::ConsumeStatus::Consumed,
            "packed fixture retained an idle lane");
    require(same_ownership(program.physical_usage(), empty),
            "packed finals or mixed execution leaked Context resources");
}

void fresh_scoring(const char* artifact) {
    EngineOptions options;
    options.purpose        = EnginePurpose::CausalScoring;
    options.enable_vision  = false;
    options.use_cuda_graph = false;
    options.max_context = options.prefill_chunk = 128;
    options.context_cache                       = {.enabled = false, .host_capacity_bytes = 0};
    DeviceContext device;
    artifact::Reader reader(artifact);
    auto definition = q35::read_definition(reader, models::load_options(options));
    auto profile    = q35::execution::describe_model(*definition);
    std::shared_ptr<const q35::ModelShard> model =
        q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
    auto frontend = models::qwen::make_frontend(definition->resources(),
                                                {.vision_enabled = false, .max_context = 128}, {});
    native::NativeRankFixture rank(definition, model, profile, options);
    q35::Session session(q35::prepare_session(definition, std::move(rank.session_plan)),
                         rank.executor());
    native::EventDriver driver(session, rank);
    unsigned starts = 0, completed = 0;
    bool written          = false;
    const auto read_image = [&](std::int64_t offset) {
        const auto image = rank.rank().resources.state_images.storage().image(offset);
        std::vector<std::byte> bytes(image.bytes);
        CUDA_CHECK(cudaMemcpy(bytes.data(), image.data, image.bytes, cudaMemcpyDeviceToHost));
        return bytes;
    };
    rank.observer().submitted = [&](const auto& request) {
        const auto* score = std::get_if<q35::ScoreWork>(&request.work.operation);
        if (!score) return;
        require(starts == completed && starts < 2, "unexpected scoring execution window");
        require(score->row.main_frontier == 0 && score->row.backend_frontier == 0 &&
                    score->row.rope_delta == 0,
                "fresh scoring exposed a prior KV history");
        if (starts) require(written, "first scoring run did not exercise StateImage writes");
        const auto image = read_image(score->row.state_offsets.source);
        require(std::all_of(image.begin(), image.end(),
                            [](auto value) { return value == std::byte{}; }),
                "fresh scoring inherited the previous recurrent state");
        ++starts;
    };
    rank.observer().observed = [&](const auto& event) {
        const auto* ready = std::get_if<runtime::ResultReady<q35::ExecutionTypes>>(&event);
        if (!ready || !std::holds_alternative<q35::ScoreResult>(ready->result.output)) return;
        const auto& work =
            std::get<q35::ScoreWork>(rank.rank().slots[0]->compute->request.work.operation);
        const auto image = read_image(work.row.state_offsets.destination);
        written          = std::any_of(image.begin(), image.end(),
                                       [](auto value) { return value != std::byte{}; });
        ++completed;
    };
    std::vector<TokenId> ids(66);
    for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = 100 + i % 27;
    const auto empty = session.physical_usage();
    for (unsigned i = 0; i < 2; ++i) {
        const auto ticket = session.start_score(frontend.prepare_tokens(ids), 1);
        require(ticket.has_value(), "fresh scoring could not acquire a root binding");
        const auto result = driver.wait<q35::ScoreComplete>(ticket);
        require(result.logprobs.size() == ids.size() - 1 &&
                    same_ownership(session.physical_usage(), empty) &&
                    !rank.executor().has_pending_work(),
                "scoring left context ownership or omitted targets");
    }
    require(starts == 2 && completed == 2, "fresh scoring observation is incomplete");
}

SpeculativeBackend selected_backend(std::string_view name) {
    if (name == "none") { return SpeculativeBackend::None; }
    if (name == "mtp") { return SpeculativeBackend::Mtp; }
    if (name == "dflash") { return SpeculativeBackend::DFlash; }
    if (name == "dflash2") { return SpeculativeBackend::DFlash2; }
    throw std::invalid_argument("backend must be none, mtp, dflash or dflash2");
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "SKIP: set NINFER_TEST_ARTIFACT to an explicit .ninfer artifact\n";
        return 77;
    }
    try {
        require(argc <= 2,
                "usage: ninfer_qwen3_5_native_transactions_test [none|mtp|dflash|dflash2]");
        const auto backend = selected_backend(argc == 2 ? argv[1] : "none");
        DeviceContext device;
        models::LoadOptions selected;
        selected.speculative = backend;
        artifact::Reader reader(artifact);
        auto definition = q35::read_definition(reader, selected);
        auto profile    = q35::execution::describe_model(*definition);
        std::shared_ptr<const q35::ModelShard> model =
            q35::materialize_model(q35::resolve_load(reader, definition, device.device()), device);
        auto frontend = ninfer::models::qwen::make_frontend(
            definition->resources(), {.vision_enabled = false, .max_context = kCapacity},
            q35::default_sampling(definition->config().text.architecture));
        EngineOptions options;
        options.max_context              = kCapacity;
        options.prefill_chunk            = kChunk;
        options.max_concurrency          = 2;
        options.kv_cache                 = KvCacheStorage::Fp8E4m3fnGroup128Ue8m0;
        options.use_cuda_graph           = false;
        options.speculative.backend      = backend;
        options.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0U : 3U;
        const auto capacity_plan         = q35::make_sequence_planner(profile, device, options);
        // Two retained images and both independent KV history classes may coexist on Host.
        options.context_cache.host_capacity_bytes =
            4 * describe_context_pool(capacity_plan.capacity_plan().records).block_bytes;
        {
            const auto stage = [](const char* label) {
                std::cerr << "fixture stage: " << label << '\n';
            };
            stage("open");
            SessionFixture instance(definition, model, profile, options, {.retained_states = 4});
            Fixture fixture(instance.rank, instance.session, frontend);
            auto run = [&](const char* label, void (Fixture::*case_)()) {
                stage(label);
                (fixture.*case_)();
            };
            run("flight_checkpoints", &Fixture::flight_checkpoints);
            run("final_window_checkpoints", &Fixture::final_window_checkpoints);
            run("consecutive_decode_terminal", &Fixture::consecutive_decode_terminal);
            run("rewrite_segments_share_prefill_unit", &Fixture::rewrite_segments_share_prefill_unit);
            run("raw_identity_opt_out", &Fixture::raw_identity_opt_out);
            run("cancel_gpu_binding", &Fixture::cancel_gpu_binding);
            run("cancel_host_restore", &Fixture::cancel_host_restore);
            run("replay_pause_and_resume", &Fixture::replay_pause_and_resume);
            run("pause_before_first_prefill_chunk", &Fixture::pause_before_first_prefill_chunk);
            run("independent_physical_demotions", &Fixture::independent_physical_demotions);
            run("demotion_holder_changes", &Fixture::demotion_holder_changes);
            run("release_set_union", &Fixture::release_set_union);
            run("physical_facts", &Fixture::physical_facts);
            run("grammar_row_failure", &Fixture::grammar_row_failure);
            run("cancel_and_destroy_paused_prefill", &Fixture::cancel_and_destroy_paused_prefill);
        }
        {
            std::cerr << "fixture stage: rewind block\n";
            SessionFixture instance(definition, model, profile, options, {.retained_states = 2});
            Fixture fixture(instance.rank, instance.session, frontend);
            fixture.fork_capacity_retry();
            fixture.recovery_capacity_and_progress();
            fixture.restore_own_snapshot();
            fixture.private_move();
        }
        {
            std::cerr << "fixture stage: bf16 geometry\n";
            auto geometry     = options;
            geometry.kv_cache = KvCacheStorage::BFloat16;
            SessionFixture instance(definition, model, profile, geometry);
            Fixture fixture(instance.rank, instance.session, frontend);
            fixture.capped_prefill_matches_uncapped(instance.rank);
        }
        std::cerr << "fixture stage: replay_sampling_counts\n";
        replay_sampling_counts(definition, model, profile, frontend, options);
        if (backend == SpeculativeBackend::None) {
            sample_fork_and_packed_finals(definition, model, profile, frontend, options);
            model.reset();
            fresh_scoring(artifact);
        }
        std::cout << "OK native transaction cancellation, paused ownership and replay recovery\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL native transactions: " << error.what() << '\n';
        return 1;
    }
}
