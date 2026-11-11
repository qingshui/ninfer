// Independent FP64 reference of the Spark2_5 text model (modeling_spark.py semantics).
//
// Weights are read from the artifact objects and decoded here: BF16 words directly, and
// fp8_e4m3fn_row_fp32 as code * row scale (the scale applied in FP64 after the dot product).
// Activations are never rounded: no BF16 stages, no activation quantization, no FP8 compute.
#pragma once

#include "artifact/reader.h"
#include "models/spark2_5/definition.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace spark_test {
using namespace ninfer;
namespace spark = ninfer::models::spark2_5;

inline void require(bool value, const std::string& message) {
    if (!value) throw std::runtime_error(message);
}

inline float bf16_to_float(std::uint16_t word) {
    return std::bit_cast<float>(std::uint32_t{word} << 16);
}

inline std::uint16_t float_to_bf16(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return static_cast<std::uint16_t>((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}

template <class Function>
void parallel_for(std::int64_t count, Function&& function) {
    const auto workers = std::max<std::int64_t>(
        1, std::min<std::int64_t>(count, std::min<unsigned>(64, std::thread::hardware_concurrency())));
    std::atomic<std::int64_t> next{0};
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(workers);
    for (std::int64_t w = 0; w < workers; ++w)
        threads.emplace_back([&, w] {
            try {
                for (std::int64_t i; (i = next.fetch_add(1)) < count;) function(i);
            } catch (...) { errors[w] = std::current_exception(); }
        });
    for (auto& thread : threads) thread.join();
    for (auto& error : errors)
        if (error) std::rethrow_exception(error);
}

inline double decode_e4m3(std::uint8_t code) {
    const int sign = code >> 7, exponent = (code >> 3) & 15, mantissa = code & 7;
    const double magnitude = exponent == 0 ? std::ldexp(mantissa / 8.0, -6)
                                           : std::ldexp(1.0 + mantissa / 8.0, exponent - 7);
    return sign ? -magnitude : magnitude;
}

// A weight matrix [rows, cols] as the exact code of every element and an optional row scale.
struct Matrix {
    std::int64_t rows = 0, cols = 0;
    std::vector<float> code;
    std::vector<double> scale;

    [[nodiscard]] double row_scale(std::int64_t row) const { return scale.empty() ? 1.0 : scale[row]; }
};

// The represented format of a decoded parameter, as the artifact names it.
struct Decoded {
    Matrix matrix;
    QType format = QType::BF16;
};

class ArtifactWeights {
public:
    ArtifactWeights(const artifact::Reader& reader, const spark::ModelDefinition& definition)
        : reader_(reader), definition_(definition) {}

    // 2-D parameters are decoded as matrices; 1-D parameters are returned as rows == 1.
    Decoded matrix(models::WeightId id) {
        const auto& reference = definition_.parameter(id);
        require(!reference.shape.empty() && reference.shape.size() <= 2, reference.name + ": rank");
        Decoded out;
        out.matrix.cols = static_cast<std::int64_t>(reference.shape.back());
        out.matrix.rows = reference.shape.size() == 2 ? static_cast<std::int64_t>(reference.shape[0]) : 1;
        out.matrix.code.resize(static_cast<std::size_t>(out.matrix.rows * out.matrix.cols));
        std::size_t filled = 0;
        for (const auto& part : reference.binding.parts) {
            const auto& g     = reader_.geometry(part.object);
            const auto& bytes = object(part.object);
            const auto first = part.begin, count = part.end - part.begin;
            require(filled + count <= out.matrix.code.size(), reference.name + ": part overflow");
            if (g.format == QType::BF16) {
                for (std::uint64_t e = 0; e < count; ++e) {
                    std::uint16_t word;
                    std::memcpy(&word, bytes.data() + 2 * (first + e), 2);
                    out.matrix.code[filled + e] = bf16_to_float(word);
                }
            } else if (g.format == QType::FP8_E4M3FN_ROW_FP32) {
                const auto columns = static_cast<std::uint64_t>(out.matrix.cols);
                require(first % columns == 0 && count % columns == 0 && g.shape.back() == columns,
                        reference.name + ": part is not whole rows");
                out.matrix.scale.reserve(out.matrix.scale.size() + count / columns);
                for (std::uint64_t r = 0; r < count / columns; ++r) {
                    const auto object_row = first / columns + r;
                    for (std::uint64_t c = 0; c < columns; ++c)
                        out.matrix.code[filled + r * columns + c] = static_cast<float>(decode_e4m3(
                            static_cast<std::uint8_t>(bytes[object_row * g.code_bytes_per_row + c])));
                    float scale;
                    std::memcpy(&scale, bytes.data() + g.scale_offset + 4 * object_row, 4);
                    out.matrix.scale.push_back(scale);
                }
            } else {
                throw std::runtime_error(reference.name + ": the reference does not decode this format");
            }
            out.format = g.format;
            filled += count;
        }
        require(filled == out.matrix.code.size(), reference.name + ": binding does not cover shape");
        return out;
    }

    std::vector<double> vector(models::WeightId id) {
        auto decoded = matrix(id);
        return {decoded.matrix.code.begin(), decoded.matrix.code.end()};
    }

private:
    const std::vector<std::byte>& object(artifact::ObjectHandle handle) {
        auto found = cache_.find(handle.index);
        if (found == cache_.end())
            found = cache_.emplace(handle.index, reader_.read_object(handle)).first;
        return found->second;
    }

    const artifact::Reader& reader_;
    const spark::ModelDefinition& definition_;
    std::unordered_map<std::size_t, std::vector<std::byte>> cache_;
};

struct RefLayer {
    std::vector<double> input_norm, post_norm;
    Matrix q, k, v, gate, output, mlp_gate, mlp_up, down;
};

struct RefModel {
    spark::TextConfig config;
    std::shared_ptr<Matrix> embedding, head;
    std::vector<double> final_norm;
    std::vector<RefLayer> layers;
    // Format of the attention/MLP projections, from which the LinearPolicy criterion follows.
    QType projection_format = QType::BF16;
};

inline RefModel load_reference(const artifact::Reader& reader,
                               const spark::ModelDefinition& definition) {
    ArtifactWeights weights(reader, definition);
    const auto& w = definition.weights().text;
    RefModel out;
    out.config    = definition.config();
    out.embedding = std::make_shared<Matrix>(weights.matrix(w.token_embedding).matrix);
    const auto& stored = definition.parameter(w.token_embedding).binding.parts;
    const auto& head   = definition.parameter(w.output_head).binding.parts;
    const bool shared  = stored.size() == head.size() &&
                        std::equal(stored.begin(), stored.end(), head.begin(), [](auto& a, auto& b) {
                            return a.object == b.object && a.begin == b.begin && a.end == b.end;
                        });
    out.head       = shared ? out.embedding : std::make_shared<Matrix>(weights.matrix(w.output_head).matrix);
    out.final_norm = weights.vector(w.final_norm);
    for (const auto& l : w.layers) {
        RefLayer layer;
        layer.input_norm = weights.vector(l.input_norm);
        layer.post_norm  = weights.vector(l.post_attention_norm);
        auto q = weights.matrix(l.attention.query);
        layer.q = std::move(q.matrix);
        out.projection_format = q.format;
        layer.k        = weights.matrix(l.attention.key).matrix;
        layer.v        = weights.matrix(l.attention.value).matrix;
        layer.gate     = weights.matrix(l.attention.gate).matrix;
        layer.output   = weights.matrix(l.attention.output).matrix;
        layer.mlp_gate = weights.matrix(l.mlp.gate).matrix;
        layer.mlp_up   = weights.matrix(l.mlp.up).matrix;
        layer.down     = weights.matrix(l.mlp.down).matrix;
        out.layers.push_back(std::move(layer));
    }
    return out;
}

// Deliberate errors a correct reference must differ from by more than the criterion.
struct Mutation {
    // A sliding key at distance d is visible when d <= max_distance (the model: 511).
    int max_distance = 511;
    // Full layers rotate every channel instead of the leading partial_rotary_factor share.
    bool full_rotation = false;
    enum class Gate { Headwise, Elementwise, Raw } gate = Gate::Headwise;
    // Replaces the output head (the source lm_head instead of the tied embedding).
    std::shared_ptr<Matrix> head;
};

// y[t][r] = scale[r] * sum_c w[r][c] * x[t][c], x is [tokens][cols], y is [tokens][rows].
inline std::vector<double> matmul(const Matrix& w, const std::vector<double>& x, std::int64_t tokens) {
    require(static_cast<std::int64_t>(x.size()) == tokens * w.cols, "matmul: input shape");
    std::vector<double> y(static_cast<std::size_t>(tokens * w.rows));
    parallel_for(w.rows, [&](std::int64_t r) {
        const float* row = w.code.data() + r * w.cols;
        for (std::int64_t t = 0; t < tokens; ++t) {
            const double* xt = x.data() + t * w.cols;
            double a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            std::int64_t c = 0;
            for (; c + 4 <= w.cols; c += 4) {
                a0 += double(row[c]) * xt[c];
                a1 += double(row[c + 1]) * xt[c + 1];
                a2 += double(row[c + 2]) * xt[c + 2];
                a3 += double(row[c + 3]) * xt[c + 3];
            }
            for (; c < w.cols; ++c) a0 += double(row[c]) * xt[c];
            y[t * w.rows + r] = w.row_scale(r) * ((a0 + a1) + (a2 + a3));
        }
    });
    return y;
}

inline std::vector<double> rmsnorm(const std::vector<double>& x, std::int64_t width,
                                   const std::vector<double>& gain, double epsilon) {
    std::vector<double> y(x.size());
    for (std::size_t t = 0; t < x.size() / width; ++t) {
        double sum = 0;
        for (std::int64_t d = 0; d < width; ++d) sum += x[t * width + d] * x[t * width + d];
        const double inverse = 1.0 / std::sqrt(sum / width + epsilon);
        for (std::int64_t d = 0; d < width; ++d) y[t * width + d] = x[t * width + d] * inverse * gain[d];
    }
    return y;
}

// Split-half rotation of the leading `rotary` channels of every head, in place.
inline void rotate(std::vector<double>& x, std::int64_t heads, std::int64_t head_dim,
                   std::int64_t rotary, double theta, const std::vector<int>& positions) {
    const std::int64_t tokens = positions.size(), half = rotary / 2;
    for (std::int64_t t = 0; t < tokens; ++t)
        for (std::int64_t h = 0; h < heads; ++h) {
            double* v = x.data() + (t * heads + h) * head_dim;
            for (std::int64_t i = 0; i < half; ++i) {
                const double angle = positions[t] * std::pow(theta, -2.0 * double(i) / double(rotary));
                const double c = std::cos(angle), s = std::sin(angle);
                const double a = v[i], b = v[i + half];
                v[i]        = a * c - b * s;
                v[i + half] = b * c + a * s;
            }
        }
}

inline double sigmoid(double x) { return 1.0 / (1.0 + std::exp(-x)); }

inline double gelu_erf(double x) { return 0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))); }

// Logits at the requested positions of one sequence, [positions.size()][vocabulary].
inline std::vector<double> reference_logits(const RefModel& model, const std::vector<int>& ids,
                                            const std::vector<int>& emit,
                                            const Mutation& mutation = {}) {
    const auto& c             = model.config;
    const std::int64_t tokens = ids.size(), hidden = c.hidden, heads = c.heads, kv_heads = c.kv_heads;
    const std::int64_t d = c.head_dim, group = heads / kv_heads;
    std::vector<int> positions(tokens);
    for (std::int64_t t = 0; t < tokens; ++t) positions[t] = static_cast<int>(t);
    std::vector<double> x(static_cast<std::size_t>(tokens * hidden));
    for (std::int64_t t = 0; t < tokens; ++t)
        for (std::int64_t i = 0; i < hidden; ++i)
            x[t * hidden + i] = model.embedding->code[ids[t] * hidden + i];
    for (std::size_t l = 0; l < model.layers.size(); ++l) {
        const auto& w    = model.layers[l];
        const auto& kind = c.layers[l].kind;
        const auto n     = rmsnorm(x, hidden, w.input_norm, c.norm_epsilon);
        auto q           = matmul(w.q, n, tokens);
        auto k           = matmul(w.k, n, tokens);
        const auto v     = matmul(w.v, n, tokens);
        const auto g     = matmul(w.gate, n, tokens);
        const auto& rope = c.rope(kind);
        const std::int64_t rotary =
            (kind == spark::LayerKind::Full && mutation.full_rotation) ? d : rope.rotary_dim;
        rotate(q, heads, d, rotary, rope.theta, positions);
        rotate(k, kv_heads, d, rotary, rope.theta, positions);
        std::vector<double> attended(static_cast<std::size_t>(tokens * heads * d));
        const double scale = 1.0 / std::sqrt(double(d));
        parallel_for(tokens * heads, [&](std::int64_t index) {
            const std::int64_t t = index / heads, h = index % heads, kvh = h / group;
            const std::int64_t first =
                kind == spark::LayerKind::Sliding ? std::max<std::int64_t>(0, t - mutation.max_distance) : 0;
            std::vector<double> logit(t - first + 1);
            double peak = -1e300;
            for (std::int64_t j = first; j <= t; ++j) {
                double dot = 0;
                for (std::int64_t i = 0; i < d; ++i)
                    dot += q[(t * heads + h) * d + i] * k[(j * kv_heads + kvh) * d + i];
                logit[j - first] = dot * scale;
                peak             = std::max(peak, logit[j - first]);
            }
            double total = 0;
            for (auto& value : logit) total += (value = std::exp(value - peak));
            double* out = &attended[(t * heads + h) * d];
            for (std::int64_t j = first; j <= t; ++j)
                for (std::int64_t i = 0; i < d; ++i)
                    out[i] += logit[j - first] / total * v[(j * kv_heads + kvh) * d + i];
        });
        for (std::int64_t t = 0; t < tokens; ++t)
            for (std::int64_t h = 0; h < heads; ++h)
                for (std::int64_t i = 0; i < d; ++i) {
                    const std::int64_t flat = h * d + i;
                    double gate;
                    switch (mutation.gate) {
                    case Mutation::Gate::Headwise:
                        gate = sigmoid(g[t * heads + h]);
                        break;
                    case Mutation::Gate::Elementwise:
                        gate = sigmoid(g[t * heads + flat % heads]);
                        break;
                    default:
                        gate = g[t * heads + h];
                        break;
                    }
                    attended[(t * heads + h) * d + i] *= gate;
                }
        const auto branch = matmul(w.output, attended, tokens);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += branch[i];
        const auto m  = rmsnorm(x, hidden, w.post_norm, c.norm_epsilon);
        auto gate     = matmul(w.mlp_gate, m, tokens);
        const auto up = matmul(w.mlp_up, m, tokens);
        for (std::size_t i = 0; i < gate.size(); ++i) gate[i] = gelu_erf(gate[i]) * up[i];
        const auto down = matmul(w.down, gate, tokens);
        for (std::size_t i = 0; i < x.size(); ++i) x[i] += down[i];
    }
    const auto final_hidden = rmsnorm(x, hidden, model.final_norm, c.norm_epsilon);
    std::vector<double> selected;
    for (const int position : emit)
        selected.insert(selected.end(), final_hidden.begin() + position * hidden,
                        final_hidden.begin() + (position + 1) * hidden);
    return matmul(mutation.head ? *mutation.head : *model.head, selected, emit.size());
}

} // namespace spark_test
