#include "core/platform.h"
#include "ops/moe/kernels/cuda/block_scaled/grouping.cuh"
#include "ops/moe/kernels/cuda/block_scaled/nvfp4/launch.h"
#include "ops/moe/kernels/cuda/block_scaled/output.cuh"

#if !NINFER_TARGET_B200
// The fp8 block-scaled experts run mma.sync kernels that only exist off B200; the fp8 family
// is not part of the Qwen3.5-122B b200 build (the shape dispatch reports it unavailable).
#include "ops/moe/kernels/cuda/block_scaled/fp8/launch.h"
#endif

#include <algorithm>
#include <climits>
#include <stdexcept>

namespace ninfer::ops::detail::moe_cuda::block_scaled {
namespace {

__device__ __forceinline__ bool selected(ExpertSubsetKind kind, const int* mask, int e) {
    return kind == ExpertSubsetKind::All || mask[e] == (kind == ExpertSubsetKind::MaskOnes);
}

template <class Shape>
__global__ void count_assignments(const int* ids, ExpertSubsetKind kind, const int* mask,
                                  int* counts, int assignments) {
    __shared__ int histogram[Shape::experts];
    for (int e = threadIdx.x; e < Shape::experts; e += blockDim.x) { histogram[e] = 0; }
    __syncthreads();
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a < assignments) {
        const int e = ids[a];
        if (selected(kind, mask, e)) { atomicAdd(histogram + e, 1); }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < Shape::experts; e += blockDim.x) {
        if (histogram[e]) { atomicAdd(counts + e, histogram[e]); }
    }
}

template <class Shape>
__global__ void scan_assignments(Groups g, int columns) {
    __shared__ int scan[Shape::experts];
    const int e     = threadIdx.x;
    const int count = g.counts[e];
    publish_load_range<Shape::experts>(g.load_range, count);
    scan[e] = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < Shape::experts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    g.offsets[e] = e ? scan[e - 1] : 0;
    g.cursors[e] = g.offsets[e];
    if (e == Shape::experts - 1) { g.offsets[Shape::experts] = scan[e]; }
    __syncthreads();
    const int jobs = (count + columns - 1) / columns;
    scan[e]        = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < Shape::experts; delta *= 2) {
        const int preceding = e >= delta ? scan[e - delta] : 0;
        __syncthreads();
        scan[e] += preceding;
        __syncthreads();
    }
    const int first = e ? scan[e - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = e;
        g.job_columns[first + j] = j * columns;
    }
    if (e == Shape::experts - 1) { *g.jobs = scan[e]; }
}

template <class Shape>
__device__ __forceinline__ bool fully_covered_groups(Groups groups, int assignments) {
    const int tokens = assignments / Shape::top_k;
    return groups.load_range && groups.load_range[0] == tokens &&
           groups.load_range[1] == tokens;
}

template <class Shape>
__global__ void index_assignments(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                                  int assignments) {
    const int a = blockIdx.x * blockDim.x + threadIdx.x;
    if (a >= assignments) { return; }
    const int expert = ids[a];
    if (!selected(kind, mask, expert)) {
        if (g.inverse) { g.inverse[a] = -1; }
        return;
    }
    const int packed = fully_covered_groups<Shape>(g, assignments)
                           ? g.offsets[expert] + a / Shape::top_k
                           : atomicAdd(g.cursors + expert, 1);
    g.assignments[packed] = a;
    if (g.inverse) { g.inverse[a] = packed; }
}

template <class Shape>
__global__ void group_small(const int* ids, ExpertSubsetKind kind, const int* mask, Groups g,
                            int assignments, int columns) {
    __shared__ int counts[Shape::experts], ranks[Shape::top_k * 64], scan[Shape::experts];
    const int tid = threadIdx.x;
    counts[tid]   = 0;
    __syncthreads();
    for (int a = tid; a < assignments; a += Shape::experts) {
        const int e = ids[a];
        ranks[a]    = selected(kind, mask, e) ? atomicAdd(counts + e, 1) : -1;
    }
    __syncthreads();
    const int count = counts[tid];
    publish_load_range<Shape::experts>(g.load_range, count);
    scan[tid] = count;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < Shape::experts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int begin = tid ? scan[tid - 1] : 0;
    g.offsets[tid]  = begin;
    if (tid == Shape::experts - 1) { g.offsets[Shape::experts] = scan[tid]; }
    const int jobs = (count + columns - 1) / columns;
    __syncthreads();
    scan[tid] = jobs;
    __syncthreads();
#pragma unroll
    for (int delta = 1; delta < Shape::experts; delta *= 2) {
        const int preceding = tid >= delta ? scan[tid - delta] : 0;
        __syncthreads();
        scan[tid] += preceding;
        __syncthreads();
    }
    const int first = tid ? scan[tid - 1] : 0;
    for (int j = 0; j < jobs; ++j) {
        g.job_experts[first + j] = tid;
        g.job_columns[first + j] = j * columns;
    }
    if (tid == Shape::experts - 1) { *g.jobs = scan[tid]; }
    const bool ordered = fully_covered_groups<Shape>(g, assignments);
    for (int a = tid; a < assignments; a += Shape::experts) {
        const int packed = ranks[a] >= 0 ? g.offsets[ids[a]] + (ordered ? a / Shape::top_k : ranks[a])
                                         : -1;
        if (g.inverse) { g.inverse[a] = packed; }
        if (packed >= 0) { g.assignments[packed] = a; }
    }
}

template <class Shape, class Output>
__global__ void reduce(Storage storage, const float* coefficients, Output output, int elements) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i < elements) {
        const float4 routed =
            routed_value4<Shape>(i / Shape::hidden, i % Shape::hidden, storage, coefficients);
        output.store(i, output.initial(i) + routed.x);
        output.store(i + 1, output.initial(i + 1) + routed.y);
        output.store(i + 2, output.initial(i + 2) + routed.z);
        output.store(i + 3, output.initial(i + 3) + routed.w);
    }
}

// One 512-thread CTA emits column layer j across all experts. Its first output
// index is sum_e min(j, jobs_e), followed by the prefix of experts with jobs_e > j.
// Only the job metadata changes; packed assignments and GU products keep their offsets.
template <bool PreserveBalanced>
__global__ void order_jobs(Groups g, int columns) {
    __shared__ int warp_offsets[16], warp_before[16], base;
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const int jobs = (g.counts[e] + columns - 1) / columns;
    const int j    = blockIdx.x;
    if constexpr (PreserveBalanced) {
        int maximum = jobs;
#pragma unroll
        for (int d = 16; d; d /= 2) {
            maximum = max(maximum, __shfl_down_sync(0xffffffff, maximum, d));
        }
        if (!lane) warp_before[warp] = maximum;
        __syncthreads();
        if (!warp) {
            int total = lane < 16 ? warp_before[lane] : 0;
#pragma unroll
            for (int d = 16; d; d /= 2) {
                total = max(total, __shfl_down_sync(0xffffffff, total, d));
            }
            if (!lane) base = total;
        }
        __syncthreads();
        // GU consumed expert-major jobs. Keeping adjacent pairs reuses their
        // weights; larger hotspot groups retain column ordering for scheduling.
        // The predicate is uniform and derived solely from Device group counts.
        if (base <= 2) return;
    }
    const int value = int(jobs > j);
    int prefix = value, before = min(j, jobs);
#pragma unroll
    for (int d = 1; d < 32; d *= 2) {
        const int preceding = __shfl_up_sync(0xffffffff, prefix, d);
        if (lane >= d) { prefix += preceding; }
    }
#pragma unroll
    for (int d = 16; d; d /= 2) { before += __shfl_down_sync(0xffffffff, before, d); }
    if (lane == 31) { warp_offsets[warp] = prefix; }
    if (!lane) { warp_before[warp] = before; }
    __syncthreads();
    if (!warp) {
        int total     = lane < 16 ? warp_offsets[lane] : 0;
        const int own = total;
#pragma unroll
        for (int d = 1; d < 32; d *= 2) {
            const int preceding = __shfl_up_sync(0xffffffff, total, d);
            if (lane >= d) { total += preceding; }
        }
        if (lane < 16) { warp_offsets[lane] = total - own; }
        int sum = lane < 16 ? warp_before[lane] : 0;
#pragma unroll
        for (int d = 16; d; d /= 2) { sum += __shfl_down_sync(0xffffffff, sum, d); }
        if (!lane) { base = sum; }
    }
    __syncthreads();
    if (value) {
        const int index      = base + warp_offsets[warp] + prefix - 1;
        g.job_experts[index] = e;
        g.job_columns[index] = j * columns;
    }
}

} // namespace

void launch_column_jobs(Groups groups, int tokens, int columns, bool preserve_balanced,
                        CudaExecutionView execution) {
    // Stream ordering is the lifetime boundary: GU must finish reading the
    // expert-major metadata before this launch overwrites it for down.
    if (tokens <= columns) { return; }
    if (!groups.counts) { throw std::logic_error("job ordering requires counted groups"); }
    if (preserve_balanced)
        order_jobs<true><<<(tokens + columns - 1) / columns, kExperts, 0, execution.stream()>>>(
            groups, columns);
    else
        order_jobs<false><<<(tokens + columns - 1) / columns, kExperts, 0, execution.stream()>>>(
            groups, columns);
    CUDA_CHECK(cudaGetLastError());
}

template <class Shape>
void launch_shape(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
                  ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
                  Tensor& destination, const Plan& plan, const Storage& s,
                  CudaExecutionView execution, Preparation prepared) {
    const auto stream = execution.stream();
    if (subset.kind != ExpertSubsetKind::None && plan.vector) {
        if (plan.format == QType::NVFP4) {
            nvfp4::launch_vector(x, experts, routes, subset, shared != nullptr, mode, destination,
                                 s, plan, execution);
        } else {
#if NINFER_TARGET_B200
            throw std::invalid_argument("moe block scaled: fp8 is not available on b200");
#else
            fp8::launch_vector(x, experts, routes, subset, shared != nullptr, mode, destination, s,
                               plan, execution);
#endif
        }
        if (!shared) { return; }
    } else if (subset.kind != ExpertSubsetKind::None) {
        const int assignments = x.ne[1] * Shape::top_k;
        const auto* ids       = static_cast<const int*>(routes.ids.data);
        const auto* mask      = static_cast<const int*>(subset.mask.data);
        if (plan.format == QType::NVFP4) {
            nvfp4::launch_input(x, experts, s, execution, !prepared.route_counts);
        } else {
#if NINFER_TARGET_B200
            throw std::invalid_argument("moe block scaled: fp8 is not available on b200");
#else
            fp8::launch_input(x, experts, s, execution, !prepared.route_counts);
#endif
        }
        if (plan.single_block_grouping) {
            group_small<Shape><<<1, Shape::experts, 0, stream>>>(ids, subset.kind, mask, s.groups,
                                                                assignments, plan.job_columns());
            CUDA_CHECK(cudaGetLastError());
        } else {
            if (!prepared.route_counts) {
                count_assignments<Shape><<<(assignments + 255) / 256, 256, 0, stream>>>(
                    ids, subset.kind, mask, s.groups.counts, assignments);
                CUDA_CHECK(cudaGetLastError());
            }
            scan_assignments<Shape><<<1, Shape::experts, 0, stream>>>(s.groups,
                                                                     plan.job_columns());
            CUDA_CHECK(cudaGetLastError());
            index_assignments<Shape><<<(assignments + 255) / 256, 256, 0, stream>>>(
                ids, subset.kind, mask, s.groups, assignments);
            CUDA_CHECK(cudaGetLastError());
        }
        if (plan.format == QType::NVFP4) {
            nvfp4::launch_experts(x, experts, routes, plan, s, execution);
        } else {
#if NINFER_TARGET_B200
            throw std::invalid_argument("moe block scaled: fp8 is not available on b200");
#else
            fp8::launch_experts(x, experts, routes, plan, s, execution);
#endif
        }
    }
    if (shared) {
        launch_shared(x, *shared, routes, mode, destination, s, plan, execution,
                      prepared.shared_activation);
    } else {
        with_output(destination, mode, [&](auto output) {
            const int elements = x.ne[1] * Shape::hidden;
            reduce<Shape><<<(elements / 4 + 255) / 256, 256, 0, stream>>>(
                s, static_cast<const float*>(routes.coefficients.data), output, elements);
        });
        CUDA_CHECK(cudaGetLastError());
    }
}

void launch(const Tensor& x, const CudaSwiGLUExperts& experts, const ExpertRoutes& routes,
            ExpertSubset subset, const ScaledSwiGLUTerm* shared, MoeEpilogue mode,
            Tensor& destination, const Plan& plan, const Storage& s, CudaExecutionView execution,
            Preparation prepared) {
    if (plan.hidden == 3072) {
        launch_shape<SwiGLUH3072I1024E256K8>(x, experts, routes, subset, shared, mode,
                                              destination, plan, s, execution, prepared);
        return;
    }
    launch_shape<SwiGLUH2560I640E512K10>(x, experts, routes, subset, shared, mode, destination,
                                         plan, s, execution, prepared);
}
} // namespace ninfer::ops::detail::moe_cuda::block_scaled
