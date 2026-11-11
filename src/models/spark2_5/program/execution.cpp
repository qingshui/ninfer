#include "models/spark2_5/program/internal.h"

#include "core/transfer.h"
#include "ninfer/ops/indexing.h"
#include "ninfer/ops/target_logprobs.h"

#include <algorithm>

namespace ninfer::models::spark2_5::detail {
namespace {

constexpr std::uint32_t kRank = 0;

} // namespace

ops::SamplingConfig sampling_config(const ResolvedSamplingParameters& p) {
    return {.temperature       = p.temperature,
            .top_k             = p.top_k,
            .top_p             = p.top_p,
            .min_p             = p.min_p,
            .presence_penalty  = p.presence_penalty,
            .frequency_penalty = p.frequency_penalty,
            .seed              = p.seed,
            .mask              = {}};
}

Tensor RankProgramImpl::device_tensor(const TensorRegion& r) { return resources.tensor(r); }

Tensor RankProgramImpl::row(const TensorRegion& r, std::int32_t rows) {
    return resources.tensor(r).slice(0, 0, rows);
}

Tensor RankProgramImpl::host_tensor(const TensorRegion& region) {
    auto local = region;
    local.region.offset -= resources.plan.persistent.buffers.ingress.offset;
    return local.bind(resources.ingress());
}

std::size_t RankProgramImpl::envelope_index(std::uint32_t visible) const {
    for (std::size_t i = 0; i < envelopes.size(); ++i)
        if (visible <= envelopes[i].max_visible_keys) return i;
    throw std::logic_error("Spark2_5 decode context exceeds its capacity");
}

TextInputs RankProgramImpl::inputs(std::int32_t tokens, std::int32_t batch) {
    const auto& io = resources.plan.persistent.buffers;
    return {resources.packed(io.ids, {tokens, batch}), resources.packed(io.positions, {tokens, batch}),
            row(io.valid_columns, batch), row(io.kv_rows, batch),
            row(io.destination_offsets, batch)};
}

void RankProgramImpl::prepare_inputs(std::span<const WorkRow> rows, std::uint32_t width) {
    auto& f        = *compute;
    const auto& io = resources.plan.persistent.buffers;
    const auto i32 = [&](const TensorRegion& r) {
        return static_cast<std::int32_t*>(host_tensor(r).data);
    };
    const auto i64 = [&](const TensorRegion& r) {
        return static_cast<std::int64_t*>(host_tensor(r).data);
    };
    f.visible = 1;
    for (std::size_t b = 0; b < rows.size(); ++b) {
        const auto& r      = rows[b];
        const auto count   = r.ids.size();
        i32(io.valid_columns)[b]       = count;
        i32(io.kv_rows)[b]             = r.main_table.index;
        i64(io.destination_offsets)[b] = r.states.at(kRank).destination;
        i32(io.sampling_positions)[b]  = r.sampling_position;
        i32(io.count_slots)[b]         = r.member.residency.lane.value;
        for (std::uint32_t t = 0; t < width; ++t) {
            i32(io.ids)[b * width + t] = t < count ? r.ids[t] : 0;
            i32(io.positions)[b * width + t] =
                r.main_frontier + std::min<std::size_t>(t, count ? count - 1 : 0);
        }
        f.visible = std::max<std::uint32_t>(f.visible, r.main_frontier + count);
    }
    if (io.sampling) {
        auto* configs = static_cast<ops::SamplingConfig*>(host_tensor(*io.sampling).data);
        for (std::size_t b = 0; b < rows.size(); ++b) {
            configs[b] = sampling_config(rows[b].sampling);
            if (configs[b].presence_penalty || configs[b].frequency_penalty)
                configs[b].token_counts =
                    static_cast<const int*>(device_tensor(*io.token_counts)
                                                .slice(1, rows[b].member.residency.lane.value, 1)
                                                .data);
            if (rows[b].constrained)
                configs[b].mask = {static_cast<const std::uint32_t*>(
                                       device_tensor(*io.grammar_masks).slice(1, b, 1).data),
                                   static_cast<int>((parameters.profile->public_tokens + 31) / 32)};
        }
    }
    if (f.tokens) {
        ++f.attention.forwards;
        f.attention.rows += rows.size();
        f.attention.scheduled_positions += std::uint64_t(width) * rows.size();
        for (const auto& r : rows) f.attention.valid_positions += r.ids.size();
    }
    upload_inputs(width * rows.size());
}

void RankProgramImpl::upload_inputs(std::uint32_t columns) {
    const auto& io = resources.plan.persistent.buffers;
    const auto bytes = std::size_t(columns) * sizeof(std::int32_t);
    for (const auto* region : {&io.ids, &io.positions}) {
        const auto source = host_tensor(*region);
        const auto target = device_tensor(*region);
        copy_bytes_async({target.data, bytes, cuda.device()},
                         {source.data, bytes, Device::cpu(), true}, cuda);
    }
    const auto& last = io.sampling ? *io.sampling : io.count_columns;
    const auto begin = io.valid_columns.region.offset;
    const auto size  = last.region.offset + last.region.bytes - begin;
    copy_bytes_async(resources.persistent().subspan(begin, size),
                     resources.ingress().subspan(begin - io.ingress.offset, size), cuda);
}

void RankProgramImpl::run_text(std::int32_t tokens, std::int32_t batch,
                               ops::CausalAttentionExecutionEnvelope envelope, Tensor hidden) {
    TextExecutionView view{parameters, resources.workspace, cuda, resources.kv,
                           resources.states.window()};
    forward_text(view, inputs(tokens, batch), envelope, hidden);
}

void RankProgramImpl::decode_round(std::int32_t batch,
                                   ops::CausalAttentionExecutionEnvelope envelope, bool captured,
                                   bool head) {
    const auto& io    = resources.plan.persistent.buffers;
    const auto width  = dimension(parameters.config.hidden);
    auto hidden       = resources.packed(io.hidden, {width, batch});
    timer.begin_target(captured);
    run_text(1, batch, envelope, hidden);
    ops::scatter(hidden, resources.states.sample_hidden(row(io.destination_offsets, batch)), cuda);
    timer.end_target(captured);
    if (head) project_head(hidden);
}

void RankProgramImpl::project_head(const Tensor& hidden) {
    const auto& io = resources.plan.persistent.buffers;
    auto logits    = resources.packed(*io.logits, {dimension(parameters.config.vocabulary), hidden.ne[1]});
    output_logits(parameters, hidden, logits, resources.workspace, cuda);
}

void RankProgramImpl::sample(std::int32_t batch, bool first_token) {
    const auto& io = resources.plan.persistent.buffers;
    auto logits    = resources.packed(*io.logits, {dimension(parameters.config.vocabulary), batch});
    auto ids       = resources.packed(*io.sampled_ids, {batch});
    auto positions = row(io.sampling_positions, batch);
    const auto* configs =
        static_cast<const ops::SamplingConfig*>(device_tensor(*io.sampling).data);
    auto scope = resources.workspace.scope();
    ops::sample(logits, ids, parameters.profile->public_tokens, configs, positions,
                first_token ? ops::kSamplePurposePrefill : ops::kSamplePurposeDecode,
                resources.workspace, cuda);
}

void RankProgramImpl::copy_result(std::int32_t batch) {
    const auto ids = resources.packed(*resources.plan.persistent.buffers.sampled_ids, {batch});
    copy_bytes_async(resources.egress().subspan(0, ids.bytes()),
                     {ids.data, ids.bytes(), cuda.device()}, cuda);
}

void RankProgramImpl::select_window() {
    auto& f              = *compute;
    const auto& work     = std::get<ForwardWork>(f.request.work.operation);
    const auto& original = work.rows.at(f.control_row);
    auto end = std::min<std::size_t>(original.ids.size(), f.control_offset + shape.prefill_chunk);
    for (const auto boundary : original.window_ends)
        if (boundary > f.control_offset) {
            end = std::min<std::size_t>(end, boundary);
            break;
        }
    f.window = original;
    auto& w  = *f.window;
    w.ids.assign(original.ids.begin() + f.control_offset, original.ids.begin() + end);
    w.control_last_token.reset();
    w.window_ends.clear();
    w.main_frontier += f.control_offset;
    // Later windows continue from the image the first one produced.
    if (f.control_offset)
        for (auto& [rank, selectors] : w.states) selectors.source = selectors.destination;
}

void RankProgramImpl::forward() {
    auto& f             = *compute;
    const auto& work    = f.request.work;
    auto rows           = work_rows(work);
    const auto* forward = std::get_if<ForwardWork>(&work.operation);
    const bool control  = forward && forward->kind == ForwardKind::Control &&
                         rows.front().ids.size() > 1;
    f.sequence = std::holds_alternative<ScoreWork>(work.operation) || control ||
                 (forward &&
                  (forward->kind == ForwardKind::Prefill || forward->kind == ForwardKind::Replay));
    if (control) {
        select_window();
        rows = {&*f.window, 1};
    }
    f.tokens = rows.front().ids.size();
    // The rings update in place, so a row writing a different image starts from a copy.
    for (const auto& r : rows) {
        const auto s = r.states.at(kRank);
        if (s.source != s.destination)
            resources.states.storage().copy(s.source, s.destination, cuda);
    }
    const auto& io       = resources.plan.persistent.buffers;
    const auto batch     = static_cast<std::int32_t>(rows.size());
    const auto width     = std::max(1U, f.tokens);
    const auto hidden_by = dimension(parameters.config.hidden);
    prepare_inputs(rows, width);
    f.hidden = f.sequence ? device_tensor(io.hidden).slice(1, 0, width)
                          : resources.packed(io.hidden, {hidden_by, batch});
    for (std::size_t i = 0; i < rows.size(); ++i)
        f.result.rows[control ? f.control_row : i].processed += rows[i].ids.size();
    const bool sampled = forward && forward->sample != SampleKind::None;
    if (!f.tokens) {
        // An exact-frontier restore samples from the hidden its image kept.
        const auto kept = resources.states.sample_hidden(rows[0].states.at(kRank).destination);
        copy_bytes_async({f.hidden.data, f.hidden.bytes(), cuda.device()},
                         {kept.data, kept.bytes(), cuda.device()}, cuda);
    } else if (!f.sequence) {
        const auto index = envelope_index(f.visible);
        f.graph_round    = graphs && forward->kind == ForwardKind::Ordinary;
        if (f.graph_round)
            graphs->replay(batch, f.visible);
        else
            decode_round(batch, envelopes[index], false, sampled);
        f.head_ready = sampled;
    } else {
        f.timed_prefill = prefill_timer.has_value();
        if (f.timed_prefill) prefill_timer->start();
        run_text(dimension(f.tokens), 1, {1, shape.max_context}, f.hidden);
        auto last = f.hidden.slice(1, f.tokens - 1, 1);
        auto kept = resources.states.sample_hidden(rows[0].states.at(kRank).destination);
        copy_bytes_async({kept.data, kept.bytes(), cuda.device()},
                         {last.data, last.bytes(), cuda.device()}, cuda);
        if (f.timed_prefill) prefill_timer->record_stop();
    }
    if (sampled) {
        f.phase = ComputePhase::Decision;
        decide();
        return;
    }
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitForward;
}

void RankProgramImpl::finish_forward() {
    auto& f = *compute;
    if (f.timed_prefill) {
        f.gpu.add_prefill(
            static_cast<std::uint64_t>(double(prefill_timer->elapsed_ms()) * 1e6 + 0.5));
        f.timed_prefill = false;
    }
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (f.window) {
        f.control_offset += f.tokens;
        if (f.control_offset == forward->rows[f.control_row].ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
        f.window.reset();
        if (f.control_row < forward->rows.size()) {
            f.phase = ComputePhase::Forward;
            return;
        }
    }
    if (std::holds_alternative<ScoreWork>(f.request.work.operation)) {
        f.phase = ComputePhase::Score;
        return;
    }
    if (forward->kind == ForwardKind::Control) {
        // Forced tokens update the same per-residency penalty ledger as sampled ones.
        f.control_row    = 0;
        f.control_offset = 0;
        f.counts_pending = true;
        control_counts();
        return;
    }
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::request_decision() {
    auto& f = *compute;
    if (f.decision_requested || f.decision) return;
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (!forward || forward->sample == SampleKind::None) return;
    runtime::DecisionRequest request;
    request.vocabulary = parameters.profile->public_tokens;
    for (std::uint32_t i = 0; i < forward->rows.size(); ++i)
        if (forward->rows[i].constrained) request.rows.push_back({i, {}});
    if (request.rows.empty()) {
        f.decision.emplace();
        return;
    }
    f.decision_requested = true;
    events.emplace_back(runtime::DecisionNeeded{f.request.ticket, std::move(request)});
}

void RankProgramImpl::upload_sampling() {
    auto& f            = *compute;
    const auto& io     = resources.plan.persistent.buffers;
    const auto rows    = work_rows(f.request.work);
    auto host_region   = *io.grammar_masks;
    host_region.region.offset = 0;
    auto mask_host     = host_region.bind(resources.sampling_ingress());
    auto mask_device   = device_tensor(*io.grammar_masks);
    runtime::DecisionConsumed consumed{f.request.ticket, f.decision->stage, {}};
    for (std::size_t b = 0; b < rows.size(); ++b) {
        if (!rows[b].constrained) continue;
        const auto mask = std::find_if(f.decision->rows.begin(), f.decision->rows.end(),
                                       [&](const auto& r) { return r.member == b; });
        if (mask == f.decision->rows.end())
            throw std::logic_error("Spark2_5 is missing a decision mask");
        auto dst = mask_device.slice(1, b, 1), src = mask_host.slice(1, b, 1);
        std::copy(mask->words.begin(), mask->words.end(), static_cast<std::uint32_t*>(src.data));
        const auto bytes = mask->words.size() * sizeof(std::uint32_t);
        copy_bytes_async({dst.data, bytes, cuda.device()}, {src.data, bytes, Device::cpu(), true},
                         cuda);
        f.result.rows[b].dead_positions = mask->dead_positions;
        consumed.rows.push_back({static_cast<std::uint32_t>(b), bytes});
    }
    if (!consumed.rows.empty()) events.emplace_back(std::move(consumed));
}

void RankProgramImpl::decide() {
    auto& f = *compute;
    if (!f.decision) return;
    upload_sampling();
    const auto& work = std::get<ForwardWork>(f.request.work.operation);
    const auto b     = static_cast<std::int32_t>(work.rows.size());
    if (f.graph_round)
        graphs->decide(b);
    else {
        if (!f.head_ready) {
            auto hidden = f.sequence ? f.hidden.slice(1, f.tokens ? f.tokens - 1 : 0, 1) : f.hidden;
            project_head(hidden);
        }
        sample(b, work.sample == SampleKind::FirstToken);
        copy_result(b);
    }
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::result() {
    auto& f = *compute;
    if (f.timed_prefill) {
        f.gpu.add_prefill(
            static_cast<std::uint64_t>(double(prefill_timer->elapsed_ms()) * 1e6 + 0.5));
        f.timed_prefill = false;
    }
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample != SampleKind::None) {
        const auto* ids = static_cast<const TokenId*>(resources.egress().data);
        for (std::size_t b = 0; b < f.result.rows.size(); ++b) f.result.rows[b].tokens = {ids[b]};
    }
    timer.observe_result(f.gpu);
    if (ExecutionTypes::describe_work(f.request.work).kind == ExecutionKind::Decode)
        timer.observe_decode(f.gpu, false);
    if (resources.compute_actions->occupied()) resources.compute_actions->finish();
    events.emplace_back(runtime::ResultReady<ExecutionTypes>{f.request.ticket, Result(f.result)});
    f.phase = ComputePhase::PendingCommit;
}

void RankProgramImpl::score() {
    auto& f          = *compute;
    const auto& work = std::get<ScoreWork>(f.request.work.operation);
    if (f.score_at == work.targets.size()) {
        if (f.timed_prefill) {
            f.gpu.add_prefill(
                static_cast<std::uint64_t>(double(prefill_timer->elapsed_ms()) * 1e6 + 0.5));
            f.timed_prefill = false;
        }
        timer.observe_result(f.gpu);
        if (resources.compute_actions->occupied()) resources.compute_actions->finish();
        events.emplace_back(runtime::ResultReady<ExecutionTypes>{
            f.request.ticket, Result(ScoreResult{static_cast<std::uint32_t>(work.row.ids.size()),
                                                 std::move(f.logprobs)})});
        f.phase = ComputePhase::PendingCommit;
        return;
    }
    const auto n      = static_cast<std::int32_t>(
        std::min<std::size_t>(kScoreColumns, work.targets.size() - f.score_at));
    const auto vocabulary = dimension(parameters.config.vocabulary);
    auto scope            = resources.workspace.scope();
    auto logits           = resources.workspace.alloc(DType::BF16, {vocabulary, n});
    auto targets          = resources.workspace.alloc(DType::I32, {n});
    auto output           = resources.workspace.alloc(DType::FP32, {n});
    auto* staging         = static_cast<TokenId*>(resources.compute_aux().data);
    std::copy_n(work.targets.begin() + f.score_at, n, staging);
    copy_bytes_async({targets.data, targets.bytes(), cuda.device()},
                     {staging, targets.bytes(), Device::cpu(), true}, cuda);
    const auto hidden = f.hidden.slice(1, work.first_target_column + f.score_at, n);
    output_logits(parameters, hidden, logits, resources.workspace, cuda);
    ops::target_logprobs(logits, targets, parameters.profile->public_tokens, output, cuda);
    copy_bytes_async(resources.egress().subspan(0, output.bytes()),
                     {output.data, output.bytes(), cuda.device()}, cuda);
    f.score_count = n;
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitScore;
}

void RankProgramImpl::control_counts() {
    auto& f          = *compute;
    const auto& rows = std::get<ForwardWork>(f.request.work.operation).rows;
    const auto& io   = resources.plan.persistent.buffers;
    auto device_ids  = device_tensor(io.ids);
    auto* staging    = static_cast<TokenId*>(resources.compute_aux().data);
    const auto cap   = static_cast<std::size_t>(device_ids.numel());
    std::size_t used = 0;
    while (f.control_row < rows.size() && used < cap) {
        const auto& r = rows[f.control_row];
        const auto n  = std::min<std::size_t>(cap - used, r.ids.size() - f.control_offset);
        if (n) {
            ControlTokens{r.ids, *r.control_last_token}.copy({staging + used, n}, f.control_offset);
            auto input = device_ids.slice(0, used, n);
            copy_bytes_async({input.data, input.bytes(), cuda.device()},
                             {staging + used, input.bytes(), Device::cpu(), true}, cuda);
            auto counts = device_tensor(*io.token_counts).slice(1, r.member.residency.lane.value, 1);
            ops::increment_token_counts(input, counts, cuda);
            f.control_offset += n;
            used += n;
        }
        if (f.control_offset == r.ids.size()) {
            ++f.control_row;
            f.control_offset = 0;
        }
    }
    f.counts_pending = f.control_row < rows.size();
    timer.record_result();
    compute_done.record(cuda.stream());
    f.phase = ComputePhase::AwaitResult;
}

void RankProgramImpl::commit(runtime::ExecutionTicket ticket, runtime::CommandSequence command,
                             Commit&& commit) {
    prepare_commit(ticket, commit);
    auto& f = *compute;
    timer.resolve();
    const auto* forward = std::get_if<ForwardWork>(&f.request.work.operation);
    if (forward && forward->sample != SampleKind::None) {
        const auto& io = resources.plan.persistent.buffers;
        const auto b   = static_cast<std::int32_t>(commit.rows.size());
        auto* values   = static_cast<std::int32_t*>(host_tensor(io.count_columns).data);
        bool any       = false;
        for (std::int32_t i = 0; i < b; ++i) {
            const auto& sampling = forward->rows[i].sampling;
            values[i]            = sampling.presence_penalty || sampling.frequency_penalty
                                       ? commit.rows[i].accepted
                                       : 0;
            any |= values[i] != 0;
        }
        if (any) {
            const auto columns = row(io.count_columns, b);
            copy_bytes_async({columns.data, columns.bytes(), cuda.device()},
                             {values, columns.bytes(), Device::cpu(), true}, cuda);
            auto counts = device_tensor(*io.token_counts);
            ops::increment_token_counts_batch(resources.packed(*io.sampled_ids, {1, b}), columns,
                                              row(io.count_slots, b), counts, cuda);
        }
    }
    resources.compute_actions->submit(local_actions(commit.actions, kRank));
    timer.record_retirement();
    compute_done.record(cuda.stream());
    f.retirement = command;
    f.phase      = ComputePhase::Retiring;
}

void RankProgramImpl::advance_compute() {
    if (!compute) return;
    auto& f = *compute;
    switch (f.phase) {
    case ComputePhase::Forward:
        forward();
        break;
    case ComputePhase::AwaitForward:
        if (compute_done.ready()) finish_forward();
        break;
    case ComputePhase::Decision:
        decide();
        break;
    case ComputePhase::AwaitResult:
        if (compute_done.ready()) {
            if (f.counts_pending)
                control_counts();
            else
                result();
        }
        break;
    case ComputePhase::Score:
        score();
        break;
    case ComputePhase::AwaitScore:
        if (compute_done.ready()) {
            const auto* words = static_cast<const float*>(resources.egress().data);
            f.logprobs.insert(f.logprobs.end(), words, words + f.score_count);
            f.score_at += f.score_count;
            f.phase = ComputePhase::Score;
        }
        break;
    case ComputePhase::PendingCommit:
        break;
    case ComputePhase::Retiring:
        if (compute_done.ready()) {
            if (resources.compute_actions->occupied()) resources.compute_actions->finish();
            auto observed = observation();
            observed.path = f.graph_round ? ExecutionPath::Graph : ExecutionPath::Eager;
            timer.observe_retirement(f.gpu);
            observed.gpu       = f.gpu;
            observed.attention = {ExecutionTypes::describe_work(f.request.work).kind, f.attention};
            const auto ticket  = f.request.ticket;
            const auto command = f.retirement;
            compute.reset();
            events.emplace_back(runtime::DeviceRetired{ticket, command, std::move(observed)});
        }
        break;
    }
}

} // namespace ninfer::models::spark2_5::detail
