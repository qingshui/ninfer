// Device side of the Spark2_5 execution tests: one loaded model, the KV and window state of a
// single sequence, chunked execution, and the FP64 attention oracle behind AttentionProbe.
#pragma once

#include "reference.h"

#include "artifact/reader.h"
#include "core/buffer.h"
#include "core/device.h"
#include "models/spark2_5/execution/text.h"
#include "models/spark2_5/execution/workspace.h"
#include "ops/float_codec.h"
#include "ops/op_check.h"

#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <variant>

namespace spark_test {

// The Op suites bound the worst element by a few 1e-3 of the largest reference for unit-RMS
// fixtures. Model activations reach |x| > 4, where one BF16 half-ulp of the output alone is 2^-8
// of the maximum, so the worst-element bound is one BF16 ulp (2^-7) of the largest reference.
inline constexpr test::ReductionCriterion kAttentionCriterion{2.8e-3, 1.0e-3, 0x1p-7};

inline void compare(const std::string& label, const std::vector<double>& got,
                    const std::vector<double>& reference, const test::ReductionCriterion& criterion,
                    bool expect_pass = true) {
    require(got.size() == reference.size() && !got.empty(), label + ": size mismatch");
    const auto stats = test::compute_reduction_stats(got.data(), reference.data(), got.size());
    const bool pass  = test::reduction_passes(stats, got.size(), criterion);
    std::cout << "  " << label << ": rel_l2=" << stats.relative_l2
              << " max_abs=" << stats.maximum_absolute_error
              << " max_ref=" << stats.maximum_absolute_reference
              << (expect_pass ? "" : " (mutation)") << (pass ? " within" : " outside")
              << " budget\n";
    require(pass == expect_pass, label + (expect_pass ? ": outside budget"
                                                      : ": a deliberate error stayed within budget"));
}

template <class T>
std::vector<T> download(const Tensor& tensor) {
    require(tensor.is_contiguous() && tensor.bytes() % sizeof(T) == 0, "download: layout");
    std::vector<T> out(tensor.bytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

inline std::vector<std::byte> download_bytes(ConstMemorySpan span) {
    std::vector<std::byte> out(span.bytes);
    CUDA_CHECK(cudaMemcpy(out.data(), span.data, span.bytes, cudaMemcpyDeviceToHost));
    return out;
}

inline double fp16_of_bf16(std::uint16_t word) {
    using namespace test::float_codec;
    return f16_bits_to_f32(f32_to_f16_bits(bf16_to_float(word)));
}

// A loaded artifact placed on the device.
struct Loaded {
    std::unique_ptr<artifact::Reader> reader;
    std::shared_ptr<const spark::ModelDefinition> definition;
    std::unique_ptr<spark::ModelShard> shard;
    std::unique_ptr<spark::Parameters> parameters;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
};

inline Loaded load(DeviceContext& device, const std::filesystem::path& path) {
    Loaded out;
    out.reader     = std::make_unique<artifact::Reader>(path);
    out.definition = spark::read_definition(*out.reader);
    out.shard      = spark::materialize_resident(*out.reader, out.definition, device);
    out.parameters = std::make_unique<spark::Parameters>(*out.shard);
    // The recipe's policy is what its Uses permit; the criterion follows it, not the case.
    for (std::size_t i = 0; i < out.definition->weight_data().size(); ++i)
        for (const auto& use : out.definition->uses({i}))
            if (ops::allows_a8(use.policy)) out.policy = ops::LinearPolicy::AllowA8;
    return out;
}

// Logits criteria of the 2-layer fixture, selected by the recipe's LinearPolicy.
inline test::ReductionCriterion fixture_logits_criterion(ops::LinearPolicy policy) {
    return policy == ops::LinearPolicy::A16Only ? test::ReductionCriterion{1.0e-2, 0.0, 0x1p-5}
                                                : test::ReductionCriterion{4.0e-2, 0.0, 6.0e-2};
}

// True when every 2-D weight (the linear projections and the embedding) is stored as BF16.
inline bool linear_weights_are_bf16(const Loaded& loaded) {
    for (const auto& parameter : loaded.definition->parameters()) {
        if (parameter.shape.size() != 2) continue;
        for (const auto& part : parameter.binding.parts)
            if (loaded.reader->geometry(part.object).format != QType::BF16) return false;
    }
    return true;
}

// Model-level parity against the HF fp32 forward: every compared position must reach cosine_min,
// and the share of positions with the same argmax must reach top1_min. `basis` names the case.
struct HfParityCriterion {
    double cosine_min;
    double top1_min;
    std::string basis;
};

// The activations of the reference are never quantized, so a w8a8 run or a quantized weight
// representation adds error the reference does not contain and gets the looser bounds.
inline HfParityCriterion hf_parity_criterion(const Loaded& loaded) {
    if (loaded.policy == ops::LinearPolicy::A16Only && linear_weights_are_bf16(loaded))
        return {0.999, 0.98, "bf16 weights, A16Only"};
    return {0.99, 0.95, "quantized representation / w8a8"};
}

// KV and window state of one sequence (table row 0, ring record 0) plus the buffers of a chunk.
class Rig {
public:
    Rig(DeviceContext& device, const spark::Parameters& parameters, std::uint32_t capacity,
        std::int32_t max_chunk)
        : device_(device), parameters_(parameters), capacity_(capacity), max_chunk_(max_chunk) {
        const auto& c = parameters.config;
        LayoutBuilder fixed_builder;
        const auto layout = spark::plan_full_kv(fixed_builder, c.full_layers, capacity,
                                                KvCacheStorage::BFloat16, 1);
        pages_            = 1 + (capacity - 1) / kPagedKVPageSize;
        page_bytes_       = layout.pages.page_bytes;
        context_          = DeviceBuffer(device.device(), pages_ * page_bytes_);
        fixed_            = DeviceBuffer(device.device(), fixed_builder.finish(256));
        cache_.emplace(context_.span(), fixed_.span(), layout);
        std::vector<std::int64_t> offsets(pages_);
        for (std::uint32_t i = 0; i < pages_; ++i) offsets[i] = std::int64_t(i) * page_bytes_;
        CUDA_CHECK(cudaMemcpy(cache_->tables().row(0).data, offsets.data(),
                              offsets.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice));

        StateImageBuilder state_builder;
        const auto window = spark::plan_window_state(state_builder, c.sliding_layers);
        state_layout_     = state_builder.finish();
        state_            = DeviceBuffer(device.device(), state_layout_.image_bytes);
        storage_.emplace(state_.span(), state_layout_);
        window_.emplace(*storage_, window);
        offsets_ = DeviceBuffer(device.device(), 256);
        const std::int64_t zero = 0;
        CUDA_CHECK(cudaMemcpy(offsets_.data(), &zero, sizeof(zero), cudaMemcpyHostToDevice));

        spark::TextWorkspaceShape shape;
        shape.batch               = 1;
        shape.min_tokens          = 1;
        shape.max_tokens          = max_chunk;
        shape.envelope            = {1, capacity};
        shape.kv_storage          = KvCacheStorage::BFloat16;
        shape.multiprocessor_count = device.multiprocessor_count();
        planned_ = std::max(spark::text_workspace_bytes(parameters, shape),
                            spark::output_workspace_bytes(parameters, 1, max_chunk));
        workspace_ = DeviceBuffer(device.device(), planned_);
        arena_.emplace(workspace_.span());
        hidden_ = DeviceBuffer(device.device(), std::size_t(c.hidden) * max_chunk * 2);
        logits_ = DeviceBuffer(device.device(), std::size_t(c.vocabulary) * max_chunk * 2);
        stride_ = (std::size_t(max_chunk) * 4 + 255) / 256 * 256;
        meta_   = DeviceBuffer(device.device(), 2 * stride_ + 512);
        reset();
    }

    // Empties the sequence; the page and ring bytes are scrubbed so stale data cannot pass.
    void reset() {
        CUDA_CHECK(cudaMemset(context_.data(), 0xa5, context_.size()));
        CUDA_CHECK(cudaMemset(state_.data(), 0, state_.size()));
        CUDA_CHECK(cudaDeviceSynchronize());
        position_ = 0;
    }

    [[nodiscard]] std::uint32_t position() const { return position_; }

    [[nodiscard]] std::size_t planned_workspace() const { return planned_; }

    [[nodiscard]] std::size_t peak_workspace() const { return arena_->peak_used(); }

    [[nodiscard]] std::int64_t window_offset() const { return 0; }

    [[nodiscard]] spark::TextExecutionView view() {
        return {parameters_, *arena_, device_.execution_view(), *cache_, *window_};
    }

    // Uploads ids as the next chunk of the sequence. The inputs borrow the rig's metadata buffer
    // and stay valid until the next upload; advance() commits the chunk to the position counter.
    spark::TextInputs upload(const std::vector<int>& ids) {
        const auto tokens = static_cast<std::int32_t>(ids.size());
        require(tokens > 0 && tokens <= max_chunk_ && position_ + tokens <= capacity_,
                "chunk exceeds the rig capacity");
        std::vector<std::int32_t> ids32(ids.begin(), ids.end()), positions(tokens);
        for (int i = 0; i < tokens; ++i) positions[i] = static_cast<std::int32_t>(position_) + i;
        auto* base        = static_cast<std::uint8_t*>(meta_.data());
        const auto upload = [&](std::size_t offset, const void* data, std::size_t bytes) {
            CUDA_CHECK(cudaMemcpy(base + offset, data, bytes, cudaMemcpyHostToDevice));
        };
        const std::int32_t valid = tokens, row = 0;
        upload(0, ids32.data(), tokens * 4);
        upload(stride_, positions.data(), tokens * 4);
        upload(2 * stride_, &valid, 4);
        upload(2 * stride_ + 256, &row, 4);
        const auto device = device_.device();
        const auto at     = [&](std::size_t offset, std::int32_t extent, DType dtype, std::size_t bytes) {
            return Tensor(MemorySpan(base + offset, bytes, device), dtype, {extent});
        };
        spark::TextInputs inputs;
        inputs.ids = Tensor(MemorySpan(base, tokens * 4, device), DType::I32, {tokens, 1});
        inputs.positions =
            Tensor(MemorySpan(base + stride_, tokens * 4, device), DType::I32, {tokens, 1});
        inputs.valid_columns  = at(2 * stride_, 1, DType::I32, 4);
        inputs.kv_table_rows  = at(2 * stride_ + 256, 1, DType::I32, 4);
        inputs.window_offsets = Tensor(MemorySpan(offsets_.data(), 8, device), DType::I64, {1});
        return inputs;
    }

    void advance(std::uint32_t tokens) { position_ += tokens; }

    // The ring of sliding layer `sliding_layer`, for the one sequence of the rig.
    [[nodiscard]] ConstCyclicKVCacheLayerView ring(std::uint32_t sliding_layer,
                                                   const spark::TextInputs& inputs) {
        return window_->layer_view(sliding_layer, inputs.window_offsets);
    }

    // Executes ids as the next chunk of the sequence; returns BF16 logits; column t (contiguous
    // vocabulary words) belongs to token t.
    std::vector<std::uint16_t> chunk(const std::vector<int>& ids, spark::AttentionProbe* probe) {
        const auto tokens = static_cast<std::int32_t>(ids.size());
        const auto inputs = upload(ids);
        const auto device = device_.device();
        const auto hidden_width = std::int32_t(parameters_.config.hidden);
        Tensor hidden(MemorySpan(hidden_.data(), std::size_t(hidden_width) * tokens * 2, device),
                      DType::BF16, {hidden_width, tokens});
        const auto vocabulary = std::int32_t(parameters_.config.vocabulary);
        Tensor logits(MemorySpan(logits_.data(), std::size_t(vocabulary) * tokens * 2, device),
                      DType::BF16, {vocabulary, tokens});
        const auto exec = device_.execution_view();
        spark::forward_text(view(), inputs, {position_ + 1, position_ + std::uint32_t(tokens)},
                            hidden, probe);
        spark::output_logits(parameters_, hidden, logits, *arena_, exec);
        CUDA_CHECK(cudaStreamSynchronize(exec.stream()));
        advance(tokens);
        return download<std::uint16_t>(logits);
    }

    DeviceContext& device_;
    const spark::Parameters& parameters_;
    std::uint32_t capacity_;
    std::int32_t max_chunk_;
    std::uint32_t pages_ = 0, position_ = 0;
    std::size_t page_bytes_ = 0, planned_ = 0, stride_ = 0;
    DeviceBuffer context_, fixed_, state_, offsets_, workspace_, hidden_, logits_, meta_;
    StateImageLayout state_layout_;
    std::optional<spark::FullKVCache> cache_;
    std::optional<StateImageStorage> storage_;
    std::optional<CyclicKVCacheView> window_;
    std::optional<MemoryArena> arena_;
};

// Keys and values of committed positions [first, first + count), [position][kv head][channel],
// as the cache represents them: BF16 K and FP16 V.
struct History {
    std::int64_t first = 0, count = 0;
    std::vector<double> k, v;
};

inline History read_ring(const ConstCyclicKVCacheLayerView& ring, std::int64_t record_offset,
                         std::int64_t length) {
    require(ring.capacity == 512 && ring.num_kv_heads == 2 && ring.head_dim == 256, "ring geometry");
    History out;
    out.first = std::max<std::int64_t>(0, length - ring.capacity);
    out.count = length - out.first;
    const auto kb = download_bytes(ring.k.memory(record_offset)),
               vb = download_bytes(ring.v.memory(record_offset));
    const auto& stride = ring.k.field.byte_strides;
    out.k.resize(out.count * 2 * 256);
    out.v.resize(out.k.size());
    for (std::int64_t p = out.first; p < length; ++p)
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < 256; ++d) {
                const std::int64_t at = d * stride[0] + (p % ring.capacity) * stride[1] + h * stride[2];
                std::uint16_t k, v;
                std::memcpy(&k, kb.data() + at, 2);
                std::memcpy(&v, vb.data() + at, 2);
                const auto index = ((p - out.first) * 2 + h) * 256 + d;
                out.k[index]     = bf16_to_float(k);
                out.v[index]     = test::float_codec::f16_bits_to_f32(v);
            }
    return out;
}

inline History read_pages(const ConstPagedKVBatchLayerView& cache, std::int64_t length) {
    require(cache.storage == KvCacheStorage::BFloat16 && cache.num_kv_heads == 2 &&
                cache.head_dim == 256,
            "paged cache geometry");
    History out;
    out.first = 0;
    out.count = length;
    const auto pages = static_cast<std::int32_t>((length + kPagedKVPageSize - 1) / kPagedKVPageSize);
    std::vector<std::int64_t> offsets(pages);
    // Table row 0: entries [0, pages) of the I64 [logical pages, rows] matrix.
    CUDA_CHECK(cudaMemcpy(offsets.data(), cache.k_pages.records.page_offsets.data,
                          pages * sizeof(std::int64_t), cudaMemcpyDeviceToHost));
    out.k.resize(length * 2 * 256);
    out.v.resize(out.k.size());
    for (std::int32_t page = 0; page < pages; ++page) {
        const auto kb = download_bytes(cache.k_pages.memory(offsets[page]));
        const auto vb = download_bytes(cache.v_pages.memory(offsets[page]));
        for (std::int64_t p = page * std::int64_t{kPagedKVPageSize};
             p < std::min<std::int64_t>(length, (page + 1) * std::int64_t{kPagedKVPageSize}); ++p)
            for (int h = 0; h < 2; ++h)
                for (int d = 0; d < 256; ++d) {
                    const std::int64_t at = h * cache.k_pages.head_stride_bytes() +
                                            (p % kPagedKVPageSize) * cache.k_pages.row_stride_bytes() + d * 2;
                    std::uint16_t k, v;
                    std::memcpy(&k, kb.data() + at, 2);
                    std::memcpy(&v, vb.data() + at, 2);
                    const auto index = (p * 2 + h) * 256 + d;
                    out.k[index]     = bf16_to_float(k);
                    out.v[index]     = test::float_codec::f16_bits_to_f32(v);
                }
    }
    return out;
}

// FP64 causal attention of one sequence: committed history plus the call's own columns (V
// crossing the same BF16 -> FP16 boundary as a stored row). A key at distance <= max_distance is
// visible; max_distance < 0 means every earlier key.
inline std::vector<double> attention_reference(const std::vector<std::uint16_t>& q,
                                               const std::vector<std::uint16_t>& k,
                                               const std::vector<std::uint16_t>& v,
                                               const std::vector<std::int32_t>& positions,
                                               const History& history, int max_distance) {
    const std::int64_t tokens = positions.size(), group = 4;
    std::vector<double> out(static_cast<std::size_t>(tokens * 8 * 256));
    parallel_for(tokens * 8, [&](std::int64_t index) {
        const std::int64_t t = index / 8, h = index % 8, kvh = h / group;
        const std::int64_t pi = positions[t];
        struct Key {
            const double* k;
            const double* v;
            std::int64_t at;
        };
        std::vector<Key> keys;
        std::vector<double> live_k_store((t + 1) * 256), live_v_store((t + 1) * 256);
        for (std::int64_t j = 0; j <= t; ++j)
            for (int d = 0; d < 256; ++d) {
                live_k_store[j * 256 + d] = bf16_to_float(k[(j * 2 + kvh) * 256 + d]);
                live_v_store[j * 256 + d] = fp16_of_bf16(v[(j * 2 + kvh) * 256 + d]);
            }
        for (std::int64_t p = history.first; p < history.first + history.count; ++p)
            keys.push_back({&history.k[(((p - history.first) * 2) + kvh) * 256],
                            &history.v[(((p - history.first) * 2) + kvh) * 256], p});
        for (std::int64_t j = 0; j <= t; ++j)
            keys.push_back({&live_k_store[j * 256], &live_v_store[j * 256], positions[j]});
        std::vector<Key> visible;
        for (const auto& key : keys)
            if (key.at <= pi && (max_distance < 0 || pi - key.at <= max_distance)) visible.push_back(key);
        std::vector<double> score(visible.size());
        double peak = -1e300;
        for (std::size_t j = 0; j < visible.size(); ++j) {
            double dot = 0;
            for (int d = 0; d < 256; ++d) dot += bf16_to_float(q[(t * 8 + h) * 256 + d]) * visible[j].k[d];
            score[j] = dot / 16.0;
            peak     = std::max(peak, score[j]);
        }
        double total = 0;
        for (auto& s : score) total += (s = std::exp(s - peak));
        double* dst = &out[(t * 8 + h) * 256];
        for (std::size_t j = 0; j < visible.size(); ++j)
            for (int d = 0; d < 256; ++d) dst[d] += score[j] / total * visible[j].v[d];
    });
    return out;
}

// Checks the attention output of every layer against the FP64 oracle fed with the probe's own
// inputs, so each layer is judged on what it received, not on the error accumulated before it.
class OracleProbe final : public spark::AttentionProbe {
public:
    OracleProbe(DeviceContext& device, const spark::TextConfig& config, std::int64_t record_offset)
        : device_(device), config_(config), record_offset_(record_offset) {}

    void on_attention(std::uint32_t layer, spark::LayerKind kind,
                      const spark::AttentionCall& call) override {
        CUDA_CHECK(cudaStreamSynchronize(device_.execution_view().stream()));
        require(layer < config_.layers.size() && config_.layers[layer].kind == kind,
                "probe: layer kind differs from layer_types");
        const auto positions = download<std::int32_t>(call.positions);
        const auto q = download<std::uint16_t>(call.q), k = download<std::uint16_t>(call.k),
                   v = download<std::uint16_t>(call.v), out = download<std::uint16_t>(call.out);
        const std::int64_t committed = positions.front();
        const bool sliding = kind == spark::LayerKind::Sliding;
        require(std::holds_alternative<ConstCyclicKVCacheLayerView>(call.context) == sliding,
                "probe: context kind differs from the layer kind");
        History history;
        if (sliding)
            history = read_ring(std::get<ConstCyclicKVCacheLayerView>(call.context), record_offset_, committed);
        else
            history = read_pages(std::get<ConstPagedKVBatchLayerView>(call.context), committed);
        const auto reference = attention_reference(q, k, v, positions, history, sliding ? 511 : -1);
        std::vector<double> actual(out.size());
        for (std::size_t i = 0; i < out.size(); ++i) actual[i] = bf16_to_float(out[i]);
        compare("layer " + std::to_string(layer) + (sliding ? " sliding" : " full") + " attention [" +
                    std::to_string(committed) + "," + std::to_string(committed + positions.size()) + ")",
                actual, reference, kAttentionCriterion);
        calls.push_back(layer);
    }

    std::vector<std::uint32_t> calls;

private:
    DeviceContext& device_;
    const spark::TextConfig& config_;
    std::int64_t record_offset_;
};

} // namespace spark_test
