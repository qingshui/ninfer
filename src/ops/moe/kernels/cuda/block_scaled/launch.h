#pragma once

#include "core/platform.h"
#include "ninfer/ops/moe.h"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/schedule.h"

namespace ninfer::ops::detail::moe_cuda::block_scaled {

inline constexpr int kHidden = 2560, kIntermediate = 640, kExperts = 512, kTopK = 10;
inline constexpr int kScoreRows = (kExperts + 4) / 4 * 4;

enum class SharedAlgorithm { Vector, Mma };

struct Fp8LoadTuning {
    int maximum_small_load = 0;
    int large_gate_columns = 0, large_down_columns = 0;
    int large_queued_ctas_per_sm = 0;
    int maximum_large_load = 0, dense_queued_ctas_per_sm = 0;
};

struct Plan {
    // Policies select these widths from slice M, but each tile covers assignments
    // belonging to one expert. The entry policy owns the maximum slice size.
    QType format     = QType::NVFP4;
    int intermediate = kIntermediate;
    // Geometry selects the compiled shape instance; defaults keep the registered
    // 2560/640/512/10 profile. Shape policies set all four fields together.
    int hidden = kHidden, experts = kExperts, top_k = kTopK;
    int gate_columns = 0, down_columns = 0;
    bool vector                = false;
    bool atomic_output         = false;
    bool single_block_grouping = false;
    // GU consumes the original expert-major jobs; only down may reorder them.
    bool column_major_down      = false;
    bool preserve_balanced_down = false;

    SharedAlgorithm shared_algorithm = SharedAlgorithm::Vector;
    bool shared_split_activation     = true;
    // Complete may widen shared/control tiles independently of routed-shared.
    bool network_topk      = false;
    int shared_columns     = 32;
    // K splits of the mma.sync control and shared gate/up product. Above one, each split writes
    // FP32 partial sums that a second kernel adds in split order before the epilogue; it divides
    // the 48 hidden-dimension tiles of the 3072-wide profile.
    int shared_splits      = 1;
    int queued_ctas_per_sm = 0;

    int pool_min_experts = 0, pool_expert_limit = 0, pool_columns = 0;
    int pool_row_window = 0, pool_queued_ctas_per_sm = 0;
    int pool_token_threshold = 0;

    Nvfp4Tuning nvfp4;
    std::optional<Nvfp4Tuning> nvfp4_down;
    Fp8LoadTuning fp8;
    int group_columns = 0, small_expert_limit = 0;
    Nvfp4LoadSchedule medium, large, dense;

    int job_columns() const {
        int columns = gate_columns > down_columns ? gate_columns : down_columns;
        if (fp8.maximum_small_load) {
            if (fp8.large_gate_columns > columns) columns = fp8.large_gate_columns;
            if (fp8.large_down_columns > columns) columns = fp8.large_down_columns;
        }
        return columns > group_columns ? columns : group_columns;
    }
};

struct Groups {
    int* counts;
    int* offsets;
    int* cursors;
    int* job_experts;
    int* job_columns;
    int* jobs;
    int* assignments;
    int* inverse;
    int* load_range;
};

struct Storage {
    std::uint8_t* input_codes   = nullptr;
    void* input_scales          = nullptr;
    std::uint8_t* product_codes = nullptr;
    void* product_scales        = nullptr;
    // Hidden-state codes and scales in packed-slot order (tcgen05 gate/up reads them directly).
    std::uint8_t* gathered_codes = nullptr;
    void* gathered_scales        = nullptr;
    float* contributions        = nullptr;
    float* vector_activation    = nullptr;
    float* routed_sum           = nullptr;
    void* shared_activation     = nullptr;
    // Split partial sums of the control and shared gate/up rows, [splits][tokens][rows].
    float* shared_partials      = nullptr;
    Groups groups{};
};

struct Preparation {
    bool shared_activation = false;
    bool route_counts      = false;
};

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& storage,
            CudaExecutionView execution, Preparation prepared = {});
void launch_shared(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertRoutes& routes,
                   MoeEpilogue mode, Tensor& destination, const Storage& storage, const Plan& plan,
                   CudaExecutionView execution, bool prepared = false);

#if NINFER_TARGET_B200
// B200 runs the 122B shared expert (hidden 3072) through the sm100 dense tcgen05 GEMMs
// (nvfp4/b200/shared.cu). Returns false when the call must run the vector or mma.sync
// kernels instead (small-token vector plans, FP32 contribution outputs).
bool launch_shared_dense(const Tensor& x, const ScaledSwiGLUTerm& shared, const ExpertRoutes& routes,
                         MoeEpilogue mode, Tensor& destination, const Storage& storage,
                         const Plan& plan, CudaExecutionView execution, bool prepared);

// The complete Op's control launch normally produces the shared activation beside the router
// logits. For calls large enough to fill a dense GEMM, B200 writes it with the sm100 BF16 SwiGLU
// GEMM instead and the control kernel keeps only the router rows. Returns false when the control
// kernel must produce the activation as well.
bool launch_control_shared_gate_up_dense(const Tensor& x, const MoeWeights& weights,
                                         const Storage& storage, const Plan& plan,
                                         CudaExecutionView execution);

// With the shared activation written densely, the router rows of the same call run as one sm100
// BF16 GEMM that stores the FP32 scores and zeroes the per-expert counts. Returns false when the
// router weights do not match the 122B geometry; the control kernel then computes the scores.
bool launch_control_router_dense(const Tensor& x, const MoeWeights& weights, float* scores,
                                 const Storage& storage, const Plan& plan,
                                 CudaExecutionView execution);
#endif

void launch_control_shared(const Tensor& x, const MoeWeights& weights, const ExpertRoutes& routes,
                           ExpertSubset subset, const Tensor& shared, float* scores,
                           const Storage& storage, const Plan& plan, CudaExecutionView execution);

// Overwrites job_experts/job_columns after GU has finished on the same stream.
// Counts, offsets, packed assignments and all quantized payloads remain unchanged.
void launch_column_jobs(Groups groups, int tokens, int columns, bool preserve_balanced,
                        CudaExecutionView execution);

} // namespace ninfer::ops::detail::moe_cuda::block_scaled
