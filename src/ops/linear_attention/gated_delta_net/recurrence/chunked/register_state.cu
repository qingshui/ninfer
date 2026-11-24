#include "ops/common/host/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/launch.h"
#include "ops/linear_attention/gated_delta_net/recurrence/chunked/register_state.cuh"

namespace ninfer::ops::detail::gated_delta_net::chunked {
namespace {

namespace regs = registers;

__device__ __forceinline__ void load_stage(regs::Stage& stage, const RecurrenceOperands& args,
                                           const QkChunk* qk_workspace,
                                           const ControlChunk* control_workspace, int head,
                                           int value_start, int chunk, int tid) {
    const int qh       = head / (args.value_heads / args.qk_heads);
    const int chunks   = chunk_count(args.tokens);
    const auto* qk_src = reinterpret_cast<const uint4*>(
        qk_workspace + static_cast<std::int64_t>(qh) * chunks + chunk);
    auto* qk_dst = reinterpret_cast<uint4*>(&stage.qk);
#pragma unroll
    for (int i = tid; i < sizeof(QkChunk) / 16; i += regs::kThreads) {
        common::cp_async<16, Cache::cg>(qk_dst + i, qk_src + i);
    }
    const auto* ctrl_src = reinterpret_cast<const uint4*>(
        control_workspace + static_cast<std::int64_t>(head) * chunks + chunk);
    auto* ctrl_dst = reinterpret_cast<uint4*>(&stage.control);
#pragma unroll
    for (int i = tid; i < sizeof(ControlChunk) / 16; i += regs::kThreads) {
        common::cp_async<16, Cache::cg>(ctrl_dst + i, ctrl_src + i);
    }
    constexpr int vectors = kChunkSize * regs::kValues / 8;
#pragma unroll
    for (int i = tid; i < vectors; i += regs::kThreads) {
        const int row = i / (regs::kValues / 8), col = i % (regs::kValues / 8) * 8;
        const int token       = chunk * kChunkSize + row;
        const int valid_token = token < args.tokens ? token : 0;
        const auto off =
            (static_cast<std::int64_t>(valid_token) * args.value_heads + head) * kStateDim +
            value_start + col;
        common::cp_async_zfill<16, Cache::cg>(stage.value + regs::value_index(row, col),
                                              args.v + off, token < args.tokens ? 16 : 0);
    }
    common::cp_commit();
}

template <class Word>
__global__ __launch_bounds__(regs::kThreads, 3) void chunk_register_kernel(
    RecurrenceOperands args, const QkChunk* __restrict__ qk_workspace,
    const ControlChunk* __restrict__ control_workspace) {
    extern __shared__ __align__(256) unsigned char backing[];
    auto& sm              = *reinterpret_cast<regs::Shared*>(backing);
    const int tid         = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head        = blockIdx.x / (kStateDim / regs::kValues);
    const int value_start = blockIdx.x % (kStateDim / regs::kValues) * regs::kValues;
    const int value_row   = value_start + 16 * warp;
    const int chunks      = chunk_count(args.tokens);
    const auto state_base = static_cast<std::int64_t>(head) * kStateDim * kStateDim;
    const auto out_pitch  = static_cast<std::int64_t>(args.value_heads) * kStateDim;

    regs::State state;
    regs::load_state(state, static_cast<const Word*>(args.state_in) + state_base, value_row, lane);
    load_stage(sm.stage[0], args, qk_workspace, control_workspace, head, value_start, 0, tid);
    if (chunks > 1)
        load_stage(sm.stage[1], args, qk_workspace, control_workspace, head, value_start, 1, tid);

    int stage = 0;
    for (int chunk = 0; chunk < chunks; ++chunk) {
        if (chunk + 1 < chunks)
            common::cp_wait<1>();
        else
            common::cp_wait<0>();
        __syncthreads();
        if (chunk + 2 < chunks) {
            load_stage(sm.stage[(stage + 2) % regs::kStages], args, qk_workspace,
                       control_workspace, head, value_start, chunk + 2, tid);
        }
        auto* out = args.out + (static_cast<std::int64_t>(chunk) * kChunkSize * args.value_heads +
                                head) * kStateDim + value_row;
        regs::step(state, sm.stage[stage], sm.output[warp], out, out_pitch,
                   min(kChunkSize, args.tokens - chunk * kChunkSize), args.scale, warp, lane);
        stage = stage + 1 == regs::kStages ? 0 : stage + 1;
    }
    regs::store_state(state, static_cast<Word*>(args.state_out) + state_base, value_row, lane);
}

template <class Word>
void launch_register_state(const RecurrenceOperands& args, const QkChunk* qk,
                           const ControlChunk* control, cudaStream_t stream) {
    constexpr int shared_bytes = sizeof(regs::Shared);
    (void)common::prepare_shared<shared_bytes, chunk_register_kernel<Word>, true>();
    chunk_register_kernel<Word><<<args.value_heads*(kStateDim / regs::kValues), regs::kThreads,
                                  shared_bytes, stream>>>(args, qk, control);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void launch_register_state(const RecurrenceOperands& args, const QkChunk* qk,
                           const ControlChunk* control, cudaStream_t stream) {
    if (args.state_dtype == DType::BF16)
        launch_register_state<__nv_bfloat16>(args, qk, control, stream);
    else
        launch_register_state<float>(args, qk, control, stream);
}

} // namespace ninfer::ops::detail::gated_delta_net::chunked
