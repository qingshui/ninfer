// Spark2_5 text execution against independent references.
//
//   layers --variant bf16|fp8   2-layer fixture (layer 0 sliding, layer 3 full of the checkpoint),
//                               600 positions in three chunks, per-layer attention oracle through
//                               AttentionProbe, final logits against the full-model reference,
//                               and the rope / gate / head mutation self-checks
//   lengths --variant bf16|fp8  the same fixture at seq in {1, 510, 511, 512, 513, 2048}, chunks of
//                               at most 512: per-layer oracle, the last 64 positions' logits, and the
//                               window / rope / gate mutants where they can matter
//   window                      constructed q/k/v through sliding_attention across a ring wrap
//   hf --variant bf16|fp8       a full converted checkpoint against the HF fp32 logits
//                               (NINFER_SPARK2_5_HF_REFERENCE, NINFER_SPARK2_5_HF_ARTIFACTS)
//   real                        a converted checkpoint named by NINFER_SPARK2_5_ARTIFACT
//
// The fixture directory comes from NINFER_SPARK2_5_FIXTURE (make_fixture.py). A missing fixture,
// reference or artifact skips the case with exit code 77.
#include "harness.h"

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string_view>

namespace {
using namespace spark_test;

struct Skip {
    std::string reason;
};

std::uint64_t next_random(std::uint64_t& state) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return state >> 33;
}

std::vector<int> random_ids(std::size_t count, std::uint64_t seed, std::uint32_t vocabulary) {
    std::vector<int> ids(count);
    for (auto& id : ids) id = static_cast<int>(1000 + next_random(seed) % (vocabulary - 2000));
    return ids;
}

std::filesystem::path fixture_path(const std::string& name) {
    const char* root = std::getenv("NINFER_SPARK2_5_FIXTURE");
    if (!root || !*root) throw Skip{"NINFER_SPARK2_5_FIXTURE is not set"};
    const std::filesystem::path dir(root);
    if (!std::filesystem::is_regular_file(dir / "FIXTURE_OK"))
        throw Skip{"the Spark2_5 fixture was not built (checkpoints absent)"};
    return dir / name;
}

// The source lm_head of the fixture: BF16 [vocabulary, hidden], deliberately not the embedding.
std::shared_ptr<Matrix> read_source_head(const std::filesystem::path& path,
                                         const spark::TextConfig& config) {
    auto head  = std::make_shared<Matrix>();
    head->rows = config.vocabulary;
    head->cols = config.hidden;
    head->code.resize(std::size_t(head->rows) * head->cols);
    std::ifstream in(path, std::ios::binary);
    std::vector<std::uint16_t> words(head->code.size());
    in.read(reinterpret_cast<char*>(words.data()), std::streamsize(words.size() * 2));
    require(bool(in), "lm_head.bin is truncated");
    for (std::size_t i = 0; i < words.size(); ++i) head->code[i] = bf16_to_float(words[i]);
    return head;
}

// Logits at `emit` positions from the chunked production run, [emit][vocabulary].
std::vector<double> select(const std::vector<std::vector<std::uint16_t>>& chunks,
                           const std::vector<int>& starts, const std::vector<int>& emit,
                           std::size_t vocabulary) {
    std::vector<double> out;
    for (const int position : emit) {
        std::size_t c = 0;
        while (c + 1 < starts.size() && starts[c + 1] <= position) ++c;
        const std::size_t column = std::size_t(position - starts[c]);
        for (std::size_t v = 0; v < vocabulary; ++v)
            out.push_back(bf16_to_float(chunks[c][column * vocabulary + v]));
    }
    return out;
}

// Runs `ids` as chunks of the given sizes with the per-layer oracle armed; returns the logits of
// every chunk.
std::vector<std::vector<std::uint16_t>> run_chunks(Rig& rig, const Loaded& loaded,
                                                   DeviceContext& device,
                                                   const std::vector<int>& ids,
                                                   const std::vector<int>& sizes,
                                                   std::size_t expected_layers) {
    OracleProbe probe(device, loaded.parameters->config, rig.window_offset());
    std::vector<std::vector<std::uint16_t>> logits;
    std::size_t at = 0;
    for (const int size : sizes) {
        const std::vector<int> part(ids.begin() + at, ids.begin() + at + size);
        logits.push_back(rig.chunk(part, &probe));
        at += size;
    }
    require(at == ids.size(), "chunk sizes do not cover the sequence");
    require(probe.calls.size() == sizes.size() * expected_layers,
            "the probe was not called once per layer and chunk");
    for (std::size_t i = 0; i < probe.calls.size(); ++i)
        require(probe.calls[i] == i % expected_layers, "probe calls are out of layer order");
    return logits;
}

void layers_case(const std::string& variant) {
    DeviceContext device(0);
    device.bind_to_current_thread();
    const auto loaded = load(device, fixture_path(variant + ".ninfer"));
    const auto& config = loaded.parameters->config;
    require(config.layers.size() == 2 && config.layers[0].kind == spark::LayerKind::Sliding &&
                config.layers[1].kind == spark::LayerKind::Full,
            "the fixture must be sliding then full");
    const auto reference = load_reference(*loaded.reader, *loaded.definition);
    const auto criterion = fixture_logits_criterion(loaded.policy);
    std::cout << variant << ": " << (loaded.policy == ops::LinearPolicy::A16Only ? "A16Only" : "AllowA8")
              << " logits criterion rel_l2<=" << criterion.relative_l2
              << " gross<=" << criterion.gross_relative_to_max_reference << " x max\n";

    constexpr int kLength = 600;
    const auto ids        = random_ids(kLength, 0x5eed, config.vocabulary);
    const std::vector<int> sizes{256, 256, 88}, starts{0, 256, 512};
    const std::vector<int> emit{0, 1, 37, 255, 256, 400, 510, 511, 512, 513, 598, 599};
    Rig rig(device, *loaded.parameters, 640, 256);
    const auto logits = run_chunks(rig, loaded, device, ids, sizes, config.layers.size());
    require(rig.peak_workspace() <= rig.planned_workspace(),
            "the workspace peak exceeded the planned bytes");
    const auto got = select(logits, starts, emit, config.vocabulary);

    compare("logits", got, reference_logits(reference, ids, emit), criterion);
    Mutation rope;
    rope.full_rotation = true;
    compare("mutation: full-channel rope", got, reference_logits(reference, ids, emit, rope),
            criterion, false);
    Mutation elementwise;
    elementwise.gate = Mutation::Gate::Elementwise;
    compare("mutation: elementwise gate", got, reference_logits(reference, ids, emit, elementwise),
            criterion, false);
    Mutation raw;
    raw.gate = Mutation::Gate::Raw;
    compare("mutation: gate without sigmoid", got, reference_logits(reference, ids, emit, raw),
            criterion, false);
    Mutation head;
    head.head = read_source_head(fixture_path(variant + "-lm_head.bin"), config);
    compare("mutation: source lm_head", got, reference_logits(reference, ids, emit, head),
            criterion, false);
}

// Sliding-window boundary, HF semantics: position i sees [i-511, i], distances 0..511. Each query
// of position >= 512 aims at one key whose logit leads every other key within 512 positions by
// >= 16 and whose value is its own spike channel. Even positions aim at distance 512, which must
// stay invisible (a q-k <= 512 window moves the output by about the spike height). Odd positions
// aim at distance 511, which must be visible (the Spec's literal [i-510, i] hides it).
struct WindowData {
    static constexpr int kLength = 600, kFirst = 400, kSpike = 4;
    std::vector<std::uint16_t> q, k, v;
    std::vector<std::int32_t> positions;
};

double normal(std::uint64_t& state) {
    double sum = -6.0;
    for (int i = 0; i < 12; ++i) sum += double(next_random(state) & 0xffffff) / double(1 << 24);
    return sum;
}

WindowData make_window_data() {
    WindowData data;
    constexpr int L = WindowData::kLength;
    std::uint64_t state = 0xc0ffee;
    data.k.resize(std::size_t(L) * 2 * 256);
    data.v.assign(data.k.size(), 0);
    data.q.resize(std::size_t(L) * 8 * 256);
    data.positions.resize(L);
    for (int p = 0; p < L; ++p) {
        data.positions[p] = p;
        for (int h = 0; h < 2; ++h) {
            std::vector<double> row(256);
            double norm = 0;
            for (auto& x : row) norm += (x = normal(state)) * x;
            for (int d = 0; d < 256; ++d)
                data.k[(std::size_t(p) * 2 + h) * 256 + d] =
                    float_to_bf16(float(row[d] / std::sqrt(norm) * 16.0));
            // The spike channels of the two KV heads differ, and neighbours within 256 differ.
            data.v[(std::size_t(p) * 2 + h) * 256 + (p + 64 * h) % 256] =
                float_to_bf16(float(WindowData::kSpike));
        }
    }
    // The query of position p aims at the key 512 (even p) or 511 (odd p) positions back; its own
    // key for p < 512.
    constexpr double kGain = 28.0;
    for (int p = 0; p < L; ++p)
        for (int h = 0; h < 8; ++h) {
            const int target = p >= 512 ? p - (p % 2 == 0 ? 512 : 511) : p;
            for (int d = 0; d < 256; ++d)
                data.q[(std::size_t(p) * 8 + h) * 256 + d] = float_to_bf16(float(
                    kGain / 16.0 * bf16_to_float(data.k[(std::size_t(target) * 2 + h / 4) * 256 + d])));
        }
    return data;
}

// Smallest lead, in logits, of the aimed key over every other key within 512 positions.
double smallest_lead(const WindowData& data) {
    double lead = 1e300;
    for (int p = 512; p < WindowData::kLength; ++p)
        for (int h = 0; h < 8; ++h) {
            const auto logit = [&](int key) {
                double dot = 0;
                for (int d = 0; d < 256; ++d)
                    dot += double(bf16_to_float(data.q[(std::size_t(p) * 8 + h) * 256 + d])) *
                           bf16_to_float(data.k[(std::size_t(key) * 2 + h / 4) * 256 + d]);
                return dot / 16.0;
            };
            const int target = p - (p % 2 == 0 ? 512 : 511);
            double best      = -1e300;
            for (int key = p - 512; key <= p; ++key)
                if (key != target) best = std::max(best, logit(key));
            lead = std::min(lead, logit(target) - best);
        }
    return lead;
}

void window_case() {
    DeviceContext device(0);
    device.bind_to_current_thread();
    const auto loaded = load(device, fixture_path("bf16.ninfer"));
    const auto data   = make_window_data();
    constexpr int L = WindowData::kLength, F = WindowData::kFirst;
    const double lead = smallest_lead(data);
    std::cout << "  the aimed key (distance 511 or 512) leads every other key by >= " << lead
              << " logits\n";
    require(lead >= 16.0, "the constructed keys do not lead by 16 logits");

    Rig rig(device, *loaded.parameters, 640, F);
    std::vector<std::uint16_t> got(std::size_t(L) * 8 * 256);
    std::optional<spark::TextInputs> last;
    for (const auto [first, count] : {std::pair{0, F}, std::pair{F, L - F}}) {
        const auto inputs = rig.upload(std::vector<int>(count, 1));
        const auto upload = [&](const std::vector<std::uint16_t>& host, int heads) {
            DeviceBuffer buffer(device.device(), std::size_t(count) * heads * 256 * 2);
            CUDA_CHECK(cudaMemcpy(buffer.data(), host.data() + std::size_t(first) * heads * 256,
                                  buffer.size(), cudaMemcpyHostToDevice));
            return buffer;
        };
        auto qb = upload(data.q, 8), kb = upload(data.k, 2), vb = upload(data.v, 2);
        DeviceBuffer ob(device.device(), qb.size());
        const auto tensor = [&](DeviceBuffer& b, int heads) {
            return Tensor(b.span(), DType::BF16, {256, heads, count, 1});
        };
        auto out = tensor(ob, 8);
        spark::sliding_attention(rig.view(), 0, tensor(qb, 8), tensor(kb, 2), tensor(vb, 2),
                                 inputs, out);
        CUDA_CHECK(cudaStreamSynchronize(device.execution_view().stream()));
        const auto part = download<std::uint16_t>(out);
        std::copy(part.begin(), part.end(), got.begin() + std::size_t(first) * 8 * 256);
        rig.advance(count);
        last = inputs;
    }

    const auto convert = [](const std::vector<std::uint16_t>& words) {
        std::vector<double> out(words.size());
        for (std::size_t i = 0; i < words.size(); ++i) out[i] = bf16_to_float(words[i]);
        return out;
    };
    const auto actual  = convert(got);
    const History none;
    const auto correct = attention_reference(data.q, data.k, data.v, data.positions, none, 511);
    const auto mutant  = attention_reference(data.q, data.k, data.v, data.positions, none, 512);
    const auto narrow  = attention_reference(data.q, data.k, data.v, data.positions, none, 510);
    const auto slice   = [](const std::vector<double>& x, int first, int count) {
        return std::vector<double>(x.begin() + std::size_t(first) * 8 * 256,
                                   x.begin() + std::size_t(first + count) * 8 * 256);
    };
    compare("window chunk 1 [0,400)", slice(actual, 0, F), slice(correct, 0, F), kAttentionCriterion);
    compare("window chunk 2 [400,600)", slice(actual, F, L - F), slice(correct, F, L - F),
            kAttentionCriterion);
    compare("mutation: q-k <= 512", slice(actual, F, L - F), slice(mutant, F, L - F),
            kAttentionCriterion, false);
    compare("mutation: q-k <= 510 ([i-510,i])", slice(actual, F, L - F), slice(narrow, F, L - F),
            kAttentionCriterion, false);

    // At the aimed channel the wrong window differs pointwise by the spike height: the 512 window
    // shows the distance-512 key to even positions, the 510 window hides the distance-511 key
    // from odd positions.
    double smallest = 1e300;
    for (int p = 0; p < L; ++p)
        for (int h = 0; h < 8; ++h) {
            const std::size_t base = (std::size_t(p) * 8 + h) * 256;
            if (p < 512) {
                for (int d = 0; d < 256; ++d)
                    require(correct[base + d] == mutant[base + d], "a window mutant moved a query without a key at 512");
                continue;
            }
            const int target  = p - (p % 2 == 0 ? 512 : 511);
            const int channel = (target + 64 * (h / 4)) % 256;
            const auto& wrong = p % 2 == 0 ? mutant : narrow;
            smallest = std::min(smallest, std::abs(wrong[base + channel] - correct[base + channel]));
        }
    std::cout << "  smallest pointwise mutant error at the aimed channel: " << smallest << "\n";
    require(smallest >= 0.5, "the window mutant is not separated pointwise by 0.5");

    // The ring now holds the 512 newest positions, bit for bit.
    const auto history = read_ring(rig.ring(0, *last), rig.window_offset(), L);
    require(history.first == L - 512 && history.count == 512, "ring history range");
    for (std::int64_t p = history.first; p < L; ++p)
        for (int h = 0; h < 2; ++h)
            for (int d = 0; d < 256; ++d) {
                const std::size_t at = std::size_t(p) * 2 + h;
                const std::int64_t in = ((p - history.first) * 2 + h) * 256 + d;
                require(history.k[in] == bf16_to_float(data.k[at * 256 + d]) &&
                            history.v[in] == fp16_of_bf16(data.v[at * 256 + d]),
                        "the ring does not hold the appended keys and values");
            }
}


// Four prompts of a converted checkpoint: every layer through the attention oracle and every
// position's logits against the full-model reference.
const std::vector<std::vector<int>> kPrompts = {
#include "real_prompts.inc"
};

void real_case() {
    const char* path = std::getenv("NINFER_SPARK2_5_ARTIFACT");
    if (!path || !*path) throw Skip{"NINFER_SPARK2_5_ARTIFACT is not set"};
    if (!std::filesystem::is_regular_file(path)) throw Skip{"NINFER_SPARK2_5_ARTIFACT names no file"};
    DeviceContext device(0);
    device.bind_to_current_thread();
    const auto loaded    = load(device, path);
    const auto& config   = loaded.parameters->config;
    const auto reference = load_reference(*loaded.reader, *loaded.definition);
    const auto criterion = fixture_logits_criterion(loaded.policy);
    // Top-1 must agree wherever the reference separates its two best tokens by kMargin logits;
    // closer calls flip on rounding alone and say nothing about the model.
    constexpr double kMargin = 0.1;
    const double required    = loaded.policy == ops::LinearPolicy::A16Only ? 0.98 : 0.95;
    std::size_t positions = 0, confident = 0, argmax_agree = 0;
    for (const auto& ids : kPrompts) positions += ids.size();
    require(kPrompts.size() == 4 && positions >= 256, "the real prompts must cover 4 x >= 256 positions");
    Rig rig(device, *loaded.parameters, 128, 128);
    std::vector<std::string> failures;
    for (std::size_t i = 0; i < kPrompts.size(); ++i) {
        const auto& ids = kPrompts[i];
        rig.reset();
        const int first = static_cast<int>(ids.size()) / 2, rest = int(ids.size()) - first;
        const auto logits = run_chunks(rig, loaded, device, ids, {first, rest}, config.layers.size());
        std::vector<int> emit(ids.size());
        for (std::size_t t = 0; t < emit.size(); ++t) emit[t] = static_cast<int>(t);
        const auto got  = select(logits, {0, first}, emit, config.vocabulary);
        const auto want = reference_logits(reference, ids, emit);
        // Every prompt is reported before the case fails.
        try {
            compare("prompt " + std::to_string(i) + " logits (" + std::to_string(ids.size()) +
                        " positions)",
                    got, want, criterion);
        } catch (const std::runtime_error& error) {
            failures.push_back(error.what());
        }
        for (std::size_t t = 0; t < ids.size(); ++t) {
            const auto begin = std::size_t(t) * config.vocabulary;
            const auto best  = std::max_element(want.begin() + begin, want.begin() + begin + config.vocabulary);
            double second    = -1e300;
            for (auto it = want.begin() + begin; it != want.begin() + begin + config.vocabulary; ++it)
                if (it != best) second = std::max(second, *it);
            if (*best - second < kMargin) continue;
            ++confident;
            const auto chosen = std::max_element(got.begin() + begin, got.begin() + begin + config.vocabulary);
            argmax_agree += (chosen - got.begin()) == (best - want.begin());
        }
    }
    const double rate = confident ? double(argmax_agree) / double(confident) : 0.0;
    std::cout << "  argmax agrees at " << argmax_agree << " of " << confident << " of " << positions
              << " positions with a reference margin >= " << kMargin << " (rate " << rate
              << ", required " << required << ")\n";
    require(failures.empty(), std::to_string(failures.size()) + " of " +
                                  std::to_string(kPrompts.size()) + " prompts outside budget: " +
                                  failures.front());
    require(confident > 0 && rate >= required, "top-1 agreement on confident positions is below " +
                                                   std::to_string(required));
}

// ---- hf: a full converted checkpoint against the HF fp32 logits ---------------------------------

std::filesystem::path environment_directory(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) throw Skip{std::string(name) + " is not set"};
    return value;
}

struct HfSequence {
    std::vector<int> ids, chunks, positions;
};

// hf-<variant>.json and hf-<variant>.logits.f32 of make_hf_reference.py.
struct HfReference {
    std::string reference;
    std::vector<HfSequence> sequences;
    std::vector<float> logits; // [sequence][position][vocabulary]
};

HfReference read_hf_reference(const std::filesystem::path& directory, const std::string& variant,
                              std::uint32_t vocabulary) {
    const auto manifest_path = directory / ("hf-" + variant + ".json");
    if (!std::filesystem::is_regular_file(manifest_path))
        throw Skip{manifest_path.filename().string() + " was not generated"};
    std::ifstream stream(manifest_path);
    const auto json = artifact::Json::parse(stream);
    const auto text = [&](const char* key) { return json.at(key).get<std::string>(); };
    require(text("schema") == "spark2_5-hf-reference/1", "unknown reference schema " + text("schema"));
    require(text("variant") == variant, "the reference is for variant " + text("variant"));
    require(text("activation_quantization") == "none", "the reference quantizes activations");
    require(text("dtype") == "float32" &&
                text("layout") == "[sequence][position][vocabulary] row-major",
            "unknown logits layout");
    require(text("weights") == (variant == "bf16" ? "source_bf16_upcast_fp32"
                                                  : "fp8_dequantized_per_channel_fp32"),
            "the reference weights do not match the variant: " + text("weights"));
    require(json.at("vocabulary").get<std::uint32_t>() == vocabulary,
            "the reference vocabulary differs from the model");
    HfReference out;
    out.reference = text("reference");
    std::size_t rows = 0;
    for (const auto& item : json.at("sequences")) {
        HfSequence sequence;
        sequence.ids       = item.at("ids").get<std::vector<int>>();
        sequence.chunks    = item.at("chunks").get<std::vector<int>>();
        sequence.positions = item.at("positions").get<std::vector<int>>();
        int covered        = 0;
        for (const int size : sequence.chunks) covered += size;
        require(covered == int(sequence.ids.size()), "chunks do not cover the reference ids");
        for (const int position : sequence.positions)
            require(position >= 0 && position < int(sequence.ids.size()), "position out of range");
        rows += sequence.positions.size();
        out.sequences.push_back(std::move(sequence));
    }
    const auto logits_path = directory / text("logits_file");
    require(std::filesystem::is_regular_file(logits_path) &&
                std::filesystem::file_size(logits_path) == rows * vocabulary * sizeof(float),
            "the logits file does not hold sequences x positions x vocabulary floats");
    out.logits.resize(rows * vocabulary);
    std::ifstream in(logits_path, std::ios::binary);
    in.read(reinterpret_cast<char*>(out.logits.data()), std::streamsize(out.logits.size() * 4));
    require(bool(in), "the logits file is truncated");
    return out;
}

std::size_t a8_uses(const Loaded& loaded) {
    std::size_t count = 0;
    for (const auto& weight : loaded.definition->weight_data())
        for (const auto& use : weight.uses) count += ops::allows_a8(use.policy);
    return count;
}

struct ParityStats {
    std::size_t positions = 0, top1 = 0, confident = 0, confident_top1 = 0;
    double cosine_min = 2.0, cosine_sum = 0.0;
    std::vector<std::string> misses;
};

// One position: bf16 production logits `got` against the fp32 reference row `want`.
void compare_position(ParityStats& stats, const std::string& label, const std::uint16_t* got,
                      const float* want, std::size_t vocabulary, double cosine_min) {
    double dot = 0, gg = 0, ww = 0, first = -1e300, second = -1e300;
    std::size_t got_best = 0, want_best = 0;
    double got_peak = -1e300;
    for (std::size_t v = 0; v < vocabulary; ++v) {
        const double g = bf16_to_float(got[v]), w = want[v];
        dot += g * w;
        gg += g * g;
        ww += w * w;
        if (g > got_peak) {
            got_peak = g;
            got_best = v;
        }
        if (w > first) {
            second    = first;
            first     = w;
            want_best = v;
        } else if (w > second) {
            second = w;
        }
    }
    const double cosine = dot / std::sqrt(gg * ww);
    const bool agree    = got_best == want_best;
    ++stats.positions;
    stats.top1 += agree;
    stats.cosine_sum += cosine;
    stats.cosine_min = std::min(stats.cosine_min, cosine);
    if (first - second >= 0.1) {
        ++stats.confident;
        stats.confident_top1 += agree;
    }
    if (!(cosine >= cosine_min) && stats.misses.size() < 8)
        stats.misses.push_back(label + ": cosine " + std::to_string(cosine));
}

void hf_case(const std::string& variant) {
    require(variant == "bf16" || variant == "fp8", "unknown variant " + variant);
    const auto reference_directory = environment_directory("NINFER_SPARK2_5_HF_REFERENCE");
    const auto artifact_directory  = environment_directory("NINFER_SPARK2_5_HF_ARTIFACTS");
    const auto artifact_path       = artifact_directory / (variant + ".ninfer");
    if (!std::filesystem::is_regular_file(artifact_path))
        throw Skip{artifact_path.filename().string() + " was not converted"};
    DeviceContext device(0);
    device.bind_to_current_thread();
    const auto loaded    = load(device, artifact_path);
    const auto& config   = loaded.parameters->config;
    const auto reference = read_hf_reference(reference_directory, variant, config.vocabulary);

    // The variant decides the path under test: BF16 never reaches an fp8 a8 kernel.
    const std::size_t a8 = a8_uses(loaded);
    if (variant == "bf16")
        require(a8 == 0 && loaded.policy == ops::LinearPolicy::A16Only &&
                    linear_weights_are_bf16(loaded),
                "the bf16 artifact must hold BF16 weights with every Use A16Only");
    else
        require(a8 > 0 && loaded.policy == ops::LinearPolicy::AllowA8,
                "the fp8 artifact must allow w8a8 on its quantized projections");
    const auto criterion = hf_parity_criterion(loaded);

    int capacity = 0, max_chunk = 0;
    for (const auto& sequence : reference.sequences) {
        capacity = std::max(capacity, int(sequence.ids.size()));
        for (const int size : sequence.chunks) max_chunk = std::max(max_chunk, size);
    }
    Rig rig(device, *loaded.parameters, capacity, max_chunk);
    const std::size_t vocabulary = config.vocabulary;
    ParityStats stats;
    std::size_t row = 0, tokens = 0;
    for (const auto& sequence : reference.sequences) {
        rig.reset();
        const std::size_t first_row = row;
        row += sequence.positions.size();
        int start = 0;
        for (const int size : sequence.chunks) {
            const std::vector<int> part(sequence.ids.begin() + start, sequence.ids.begin() + start + size);
            const auto logits = rig.chunk(part, nullptr);
            for (std::size_t i = 0; i < sequence.positions.size(); ++i) {
                const int position = sequence.positions[i];
                if (position < start || position >= start + size) continue;
                compare_position(stats, "position " + std::to_string(position),
                                 logits.data() + std::size_t(position - start) * vocabulary,
                                 reference.logits.data() + (first_row + i) * vocabulary, vocabulary,
                                 criterion.cosine_min);
            }
            start += size;
        }
        tokens += sequence.ids.size();
    }
    require(stats.positions == row && stats.positions >= 256,
            "the reference must compare at least 256 positions");
    const double ratio = double(stats.top1) / double(stats.positions);
    std::ostringstream line;
    line << std::setprecision(9) << "hf parity " << variant << ": positions=" << stats.positions
         << " cosine_min=" << stats.cosine_min
         << " cosine_mean=" << stats.cosine_sum / double(stats.positions) << " top1=" << stats.top1
         << "/" << stats.positions << " ratio=" << ratio << " criterion cosine>=" << criterion.cosine_min
         << " top1>=" << criterion.top1_min << " basis=" << criterion.basis
         << " reference=" << reference.reference;
    std::cout << line.str() << "\n";
    std::cout << "  tokens=" << tokens << " a8_uses=" << a8 << " top1 on reference margin >= 0.1: "
              << stats.confident_top1 << "/" << stats.confident << "\n";
    for (const auto& miss : stats.misses) std::cout << "  below the cosine bound at " << miss << "\n";
    require(stats.misses.empty() && stats.cosine_min >= criterion.cosine_min,
            "cosine below " + std::to_string(criterion.cosine_min) + " at some position");
    require(ratio >= criterion.top1_min, "top-1 agreement below " + std::to_string(criterion.top1_min));
}

// ---- lengths: the 2-layer fixture around the window and chunk boundaries -----------------------

std::vector<int> split_chunks(int length) {
    std::vector<int> sizes;
    for (int left = length; left > 0; left -= 512) sizes.push_back(std::min(left, 512));
    return sizes;
}

void lengths_case(const std::string& variant) {
    require(variant == "bf16" || variant == "fp8", "unknown variant " + variant);
    DeviceContext device(0);
    device.bind_to_current_thread();
    const auto loaded  = load(device, fixture_path(variant + ".ninfer"));
    const auto& config = loaded.parameters->config;
    require(config.layers.size() == 2 && config.layers[0].kind == spark::LayerKind::Sliding &&
                config.layers[1].kind == spark::LayerKind::Full,
            "the fixture must be sliding then full");
    if (variant == "bf16")
        require(a8_uses(loaded) == 0 && loaded.policy == ops::LinearPolicy::A16Only,
                "the bf16 artifact must hold every Use A16Only");
    const auto reference = load_reference(*loaded.reader, *loaded.definition);
    const auto criterion = fixture_logits_criterion(loaded.policy);
    constexpr int kLongest = 2048;
    // Logits of the last positions. A w8a8 run rounds every token's activations on its own, so the
    // error of a few rows scatters by +-40% around its mean; 64 rows pin the mean down.
    constexpr int kCompared = 64;
    Rig rig(device, *loaded.parameters, kLongest, 512);
    for (const int length : {1, 510, 511, 512, 513, kLongest}) {
        const auto sizes = split_chunks(length);
        std::vector<int> starts{0};
        for (std::size_t i = 0; i + 1 < sizes.size(); ++i) starts.push_back(starts.back() + sizes[i]);
        std::cout << variant << " seq=" << length << " in " << sizes.size() << " chunk(s)\n";
        rig.reset();
        const auto ids = random_ids(length, 0x1e57 + length, config.vocabulary);
        // The oracle checks every layer of every chunk on the inputs the layer received: the
        // sliding layer sees distances 0..511 (HF: kv_idx > q_idx - 512), the full layer all.
        const auto logits = run_chunks(rig, loaded, device, ids, sizes, config.layers.size());
        require(rig.peak_workspace() <= rig.planned_workspace(),
                "the workspace peak exceeded the planned bytes");
        std::vector<int> emit;
        for (int p = std::max(0, length - kCompared); p < length; ++p) emit.push_back(p);
        const auto got  = select(logits, starts, emit, config.vocabulary);
        compare("seq " + std::to_string(length) + " logits", got,
                reference_logits(reference, ids, emit), criterion);
        // Partial rope and the per-head gate decide the logits once positions are non-trivial.
        if (length < 512) continue;
        Mutation rope;
        rope.full_rotation = true;
        compare("mutation: full-channel rope", got, reference_logits(reference, ids, emit, rope),
                criterion, false);
        Mutation elementwise;
        elementwise.gate = Mutation::Gate::Elementwise;
        compare("mutation: elementwise gate", got, reference_logits(reference, ids, emit, elementwise),
                criterion, false);
        Mutation raw;
        raw.gate = Mutation::Gate::Raw;
        compare("mutation: gate without sigmoid", got, reference_logits(reference, ids, emit, raw),
                criterion, false);
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (mode == "layers" && argc > 3 && std::string_view(argv[2]) == "--variant") {
            layers_case(argv[3]);
        } else if (mode == "lengths" && argc > 3 && std::string_view(argv[2]) == "--variant") {
            lengths_case(argv[3]);
        } else if (mode == "hf" && argc > 3 && std::string_view(argv[2]) == "--variant") {
            hf_case(argv[3]);
        } else if (mode == "window") {
            window_case();
        } else if (mode == "real") {
            real_case();
        } else {
            std::cerr << "usage: test_execution layers|lengths|hf --variant bf16|fp8 | window | real\n";
            return 2;
        }
    } catch (const Skip& skip) {
        std::cout << "SKIP: " << skip.reason << "\n";
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
    std::cout << "PASS\n";
    return 0;
}
