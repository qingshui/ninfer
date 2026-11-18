### 5.3 A3　MoE grouped GEMM 与 shared expert（与 A4 一起，36.1 到 48.07 req/s）

**证据。** 4096 token 时 routed MoE 为 1,463 µs（约 FP4 峰值的 5%）。ncu 与调试开关显示：gate/up 544 µs（SM 83%，DRAM 30%），down 859 µs（SM 73%，DRAM 12%，L1TEX scoreboard 停顿 40%）。只跳过 epilogue 时 gate/up 降到 330 µs，down 降到 129 µs，说明瓶颈在 epilogue。RED 微基准（4096 token）：标量 `atomicAdd` 750 µs，`red.global.add.v4.f32` 188 µs。

**机制。** down 的 epilogue 改用 `red.global.add.v4.f32`，每个 warp 经 smem 转置（每个 lane 对同一 token 的 4 个连续输出行做加法），并把 `tcgen05.ld` 流水化。gate/up 用 TileN=64（两个累加器 stage；TMEM 共 512 列，scale 因子拷贝占 96 或 64 列，仅当 `2*kAccCols + SF <= 512` 才启用 `kAccStages=2`），down 在超过 1536 token 时用 TileN=128（64 与 128 持平）。`all_slice_tokens` 由 4096 提到 8192（`ShapePolicy` 新字段）。shared expert 的 SwiGLU gate/up 走 dense tcgen05（`launch_control_shared_gate_up_dense`），`shared_mma` 保留 router logits 与 shared gate 行。之后又加入单线程 gather 与 8 个 epilogue warp（kernel 变为 12 个 warp，4 到 11 号做 epilogue）。

**效果。** 4096 token 的 routed MoE 从 1,463 µs 降到约 660 µs（down 从 859 降到约 250 µs）。均匀路由下 4096 token 的 tile 扫描（µs，TileN 为 16 / 32 / 64 / 128）：gate/up 673 / 461 / 373 / 563，down 455 / 279 / 224 / 247。这一步里的 atomic 合并在 B2 被有序合并取代（6.1 节）。

**文件。** `src/ops/moe/kernels/cuda/block_scaled/nvfp4/b200/grouped_tcgen05.cuh`、`src/ops/moe/routed_swiglu/cuda/nvfp4/b200/shapes/h3072_i1024_e256_k8/dispatch.cpp`、`src/ops/moe/topk_softmax_swiglu_moe/`、`src/ops/moe/kernels/cuda/block_scaled/{launch.h, shared.cu}`、`tests/ops/moe/test_block_scaled_122b.cpp`（28 个用例）。

### 5.4 A4　Sampler 重写

`sample_row_kernel` 每个 flight 耗时 4.8 ms，重写后（用 warp shuffle 维护 top-k 候选列表）微基准约 0.28 ms，serving 中约 0.5 到 0.7 ms，每个 flight 调用一次。在 512 个测试行上结果与旧 kernel 完全相同；列数不超过 16 的路径仍是多 block 的 partial top-k。文件：`src/ops/sampling/kernels/distribution.cuh`、`tests/ops/sampling/sampling.cpp`。

### 5.5 A5　Attention：planner、K/V loader 与按段 envelope（49.95 与 55.99 req/s）

**证据。** 引擎对每个 prefill 段、每一层发一次 `causal_softmax_attention`（每个 flight 约 13 到 15 个 B=1 的段加一次 decode batch）。当时的 profile：fmha 占 12.0%（约 13.5 ms/flight），merge 占 3.3%（约 3.8 ms/flight）。每个 CTA 的固定开销约 25 µs（Q 准备），每个 key tile 约 5.6 µs。默认计划在 `--envelope-max 2042`、W=64 时选 9 路 split 加 merge，52 µs，而单路只要 38 µs。

**机制。**

1. planner（`b200/plan.cpp`）：`splits = target < kMinSplitTarget(3) ? 1 : max(2, min(target, key_tiles/2))`，因为 merge 要多一次 launch 和一遍 FP32 partial。单路还是 split 的拓扑只取决于调用形状，不取决于 envelope，因为已捕获的 prefill 计划图要在不同 envelope profile 之间用 `DecodeGraphExecutable::update_child` 更新；拓扑随 envelope 变化曾触发 `cudaErrorGraphExecUpdateFailure`，已修复并加了回归用例。
2. fmha 的 K/V loader（`fmha.cuh`、`kv_policy.cuh`、`common/quant/fp8.cuh`）：先批量 `load_kv` 再 `store_kv`（`TileRegs`、`VRaw`、`FusedV`），fp8 的 `load_k_scale_pair`、`decode_k_scales`、`decode_v8`，使依赖的反量化不再夹在两次 V 加载之间；320 线程 CTA 的寄存器上限 168。
3. 混合（eager）遍历按段使用 envelope `{segment.keys, segment.keys}`（`SequenceSegment::keys = main_frontier + columns`）；图捕获的 packed 遍历仍用共享 envelope。

**效果（op bench）。** d256-h32-kv2，fp8，`--envelope-max 2042`，eager，GPU 为共享、有噪声的环境。单位 µs，旧的单路 split 对新实现：

| 每段 token / 上下文 | 旧 | 新 |
| --- | --- | --- |
| 223 / 0 | 54.8 | 40.2 |
| 536 / 0 | 89.8 | 54.8 |
| 1024 / 0 | 147.7 | 94.6 |
| 223 / 300 | 81.4 | 51.7 |
| 536 / 300 | 116.4 | 64.1 |
| 1024 / 300 | 197.5 | 121.6 |

当时的 nsys 汇总显示每个 flight 的 attention 从约 13.5 + 3.8 ms 降到约 3.75 ms。端到端：kMinSplitTarget=3 使 48.07 升到 49.95，fmha loader、planner 修复与按段 envelope 使 52.01 升到 55.99。测试：fp8 52 个用例，五种 KV 格式共 828 个 envelope 更新用例。

**文件。** `src/ops/softmax_attention/dense/causal_cache/b200/{plan.cpp, fmha.cuh, kv_policy.cuh}`、`src/models/qwen3_5/execution/{text.h, text.cpp}`、`program/{inputs.cpp, graphs.cpp}`、`tests/ops/softmax_attention/causal/cases.cpp`、`tests/models/qwen3_5/test_packed_forward_contract.cpp`；bench 新增几何 `d256-h32-kv2`。
