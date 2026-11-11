// Spark2_5 text execution against independent FP64 references.
//
//   layers --variant bf16|fp8   2-layer fixture (layer 0 sliding, layer 3 full of the checkpoint),
//                               600 positions in three chunks, per-layer attention oracle through
//                               AttentionProbe, final logits against the full-model reference,
//                               and the rope / gate / head mutation self-checks
//   window                      constructed q/k/v through sliding_attention across a ring wrap
//   real                        a converted checkpoint named by NINFER_SPARK2_5_ARTIFACT
//
// The fixture directory comes from NINFER_SPARK2_5_FIXTURE (make_fixture.py). A missing fixture
// or artifact skips the case with exit code 77.
#include "harness.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
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

// Sliding-window boundary. Queries at positions >= 512 see the key 512 positions back only if the
// window is wrong: that key's logit leads every in-window key by >= 16 and its value is its own
// spike channel, so a q-k <= 512 window moves the output by about the spike height.
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
    // The query of position p aims at the key 512 positions back (its own for p < 512).
    constexpr double kGain = 28.0;
    for (int p = 0; p < L; ++p)
        for (int h = 0; h < 8; ++h) {
            const int target = p >= 512 ? p - 512 : p;
            for (int d = 0; d < 256; ++d)
                data.q[(std::size_t(p) * 8 + h) * 256 + d] = float_to_bf16(float(
                    kGain / 16.0 * bf16_to_float(data.k[(std::size_t(target) * 2 + h / 4) * 256 + d])));
        }
    return data;
}

// Smallest lead, in logits, of the key 512 positions back over the keys of the window.
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
            double best = -1e300;
            for (int key = p - 511; key <= p; ++key) best = std::max(best, logit(key));
            lead = std::min(lead, logit(p - 512) - best);
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
    std::cout << "  key at distance 512 leads the window by >= " << lead << " logits\n";
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
    const auto slice   = [](const std::vector<double>& x, int first, int count) {
        return std::vector<double>(x.begin() + std::size_t(first) * 8 * 256,
                                   x.begin() + std::size_t(first + count) * 8 * 256);
    };
    compare("window chunk 1 [0,400)", slice(actual, 0, F), slice(correct, 0, F), kAttentionCriterion);
    compare("window chunk 2 [400,600)", slice(actual, F, L - F), slice(correct, F, L - F),
            kAttentionCriterion);
    compare("mutation: q-k <= 512", slice(actual, F, L - F), slice(mutant, F, L - F),
            kAttentionCriterion, false);

    // Where the key 512 back exists, the mutated reference differs pointwise by the spike height.
    double smallest = 1e300;
    for (int p = 0; p < L; ++p)
        for (int h = 0; h < 8; ++h) {
            const std::size_t base = (std::size_t(p) * 8 + h) * 256;
            if (p < 512) {
                for (int d = 0; d < 256; ++d)
                    require(correct[base + d] == mutant[base + d], "a window mutant moved a query without a key at 512");
                continue;
            }
            const int channel = (p - 512 + 64 * (h / 4)) % 256;
            smallest = std::min(smallest, std::abs(mutant[base + channel] - correct[base + channel]));
        }
    std::cout << "  smallest pointwise mutant error at the distance-512 channel: " << smallest << "\n";
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
    std::size_t positions = 0, argmax_agree = 0;
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
            const auto row = [&](const std::vector<double>& x) {
                return std::max_element(x.begin() + t * config.vocabulary,
                                        x.begin() + (t + 1) * config.vocabulary) -
                       (x.begin() + t * config.vocabulary);
            };
            argmax_agree += row(got) == row(want);
        }
    }
    std::cout << "  argmax agrees at " << argmax_agree << " of " << positions << " positions\n";
    require(failures.empty(), std::to_string(failures.size()) + " of " +
                                  std::to_string(kPrompts.size()) + " prompts outside budget: " +
                                  failures.front());
}

// CASES_PLACEHOLDER
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        if (mode == "layers" && argc > 3 && std::string_view(argv[2]) == "--variant") {
            layers_case(argv[3]);
        } else if (mode == "window") {
            window_case();
        } else if (mode == "real") {
            real_case();
        } else {
            std::cerr << "usage: test_execution layers --variant bf16|fp8 | window | real\n";
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
