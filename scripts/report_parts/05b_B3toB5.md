### 6.2 B3 与 B4　Dense GEMM 的 256 token tile 与小 kernel 调优

**B3 证据。** dense sm100 GEMM 占 kernel 时间的 17% 到 22%，BF16 只达到 BF16 峰值（孤立测得 1.55 PFLOP/s）的 65% 到 69%，而且只用了 TileN=128（8,964 次 launch，平均 149.8 µs，占 dense 时间的 82%）。

**B3 机制。** `sm100_gemm_kernel` 是 persistent、`cta_group::1`、swap-AB（权重做 MMA 的 M，token 做 N）的实现，由 TMA producer、MMA、TMEM allocator 与 4 个 epilogue warp 组成。新增 TileN=256 路径（累加器双缓冲，占 2 x 256 列 TMEM），在 rows 不少于 1024 时使用。Add（残差）GEMM 的 `PlainEpilogue` 向量化：残差与输出别名，经每个 warp 的 smem 暂存；实测用例是 shared expert 的 down（3072 x 1024）。

**B3 效果。** 孤立的 BF16 in_proj（4600 token）从 430 降到 373 µs；Add epilogue GEMM（3072 x 1024）从 129 降到约 30 µs；端到端两次运行为 70.79 与 70.40 req/s。评估过 2-CTA（`cta_group::2`）的 BF16 kernel，最好情况只比 256 tile 高约 16%，没有做。测试修复：最初的 256 tile 用例没有走到新路径（阈值是 rows 不少于 2048，而用例 rows 为 1536），阈值降到 1024。

**B3 文件。** `src/ops/common/cuda/sm100/gemm_mainloop.cuh`、`src/ops/linear/kernels/sm100/{bf16_gemm.cu, gemm.h, gemm_launch.cuh}`、`src/ops/linear/add/`。

**B4 小 kernel 与 launch bound。**

- causal conv varlen 重构：`__launch_bounds__` 的 minBlocks 为 5，编译期 stride，ftz silu。T=3442 时 89.8 降到 73.1 再到 48.7 µs，ncu 指令数从 79M 降到 20M。第一次重构后 occupancy 仍是 25%、106 个寄存器，没有收益，加上寄存器上限之后才见效。
- GDN `prepare_kernel` 的 launch bounds：varlen op 在 T=3442 时 435.7 µs；minBlocks 为 5、6、7、8 时分别为 418.6、407.8、394.9、394.9 µs，选 8。
- rmsnorm 的近似与 router 列处理：3700 token 的完整 MoE op 从 648.6 降到 577.0 µs。
- 测试：gated_delta_net 的 chunked、varlen、replay_record、replay_fold、control 共 5 组，`causal_conv1d_varlen` 8 个用例，`causal_conv1d_silu` 205 个，normalization 569 个，均通过。
- 端到端三次运行 72.54、72.46、71.94 req/s。

### 6.3 B5　Grouped（ragged）causal attention 与 grouped FP8 KV 写

**证据。** nsys：fmha 占 9.57%，外加 merge；每层约 15 次按段、B=1 的 launch，每次约 26 到 31 µs，占用低，还有 wave quantization。

**机制。**

- 一次 1-D grid 的 `fmha_group_kernel<Kv>` 覆盖混合（eager）遍历里的所有组（一个组是共享 token 轴上连续 `batch` 个条目、每条 `width` 列）。组表（最多 72 组，`cta_base` 升序）放在 `__grid_constant__` 参数里；每个 CTA 映射到（组，条目，tile，kv head），组用二分查找，后面的 tile 先做。每个 key 范围都是一个 split，直接写 BF16，没有 partial，没有 merge，没有 workspace。数值上与 split 加 merge 只差 FP32 求和顺序，各组互相隔离。
- packed prefill（可能被图捕获）保留按段调用；非 B200 或非 tcgen05 的目标回退到按组调用 `causal_softmax_attention`。
- grouped FP8 KV 写：每个（column, kv head）一个 warp，在组表上二分，用 `valid_columns` 夹紧；取代了按段写。只适用于 FP8 存储且有 2 个 KV head，其他 dtype 逐组写。

**效果。** `fmha_group` 占 kernel 时间 5.65%（`segments_and_decode` 用例 58 µs）。端到端：build #6 三次运行均值 75.75，build #7（加 grouped KV 写）76.09 req/s，TPOT 从 42.1 到 42.5 ms 降到 41.5 到 41.9 ms。测试：`causal_groups/*` 对 bf16、int8、nvfp4、k8v4、fp8 通过。

**一个缺陷。** `write_fp8_groups_kernel` 里，后一个 chunk 的二分查找可能落到 `token_base` 超过该列的组上，使 `local` 为负；已加 `local < 0` 检查。

**文件。** `src/ops/softmax_attention/`（fmha group 路径与组表）、`src/ops/kv_cache/kernels/write.{cu,h}` 与 `write/fp8.cuh`、`include/ninfer/ops/softmax_attention.h`、`bench/ops/causal_softmax_attention_bench.cu`、`src/models/qwen3_5/execution/text.cpp`。
