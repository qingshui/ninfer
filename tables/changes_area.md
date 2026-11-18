| 区域 | 文件数 | 新增行 | 删除行 | 内容 |
| --- | --- | --- | --- | --- |
| `src/runtime/engine` | 8 | 719 | 491 | 引擎：采样组与 `fork_batch`、flight 内检查点的发布、留存控制、host 路径、遥测 |
| `src/models/qwen3_5` | 17 | 922 | 865 | Session 与执行：`fork_batch`、flight 内检查点、binding、request/sequence、grouped attention 的接入 |
| `src/models/common` | 2 | 12 | 4 | prefix identity 的几何增长 `reserve_for_growth` |
| `src/ops/moe` | 13 | 534 | 268 | MoE：grouped GEMM、`gather_input`、有序合并、shared expert、量化器 |
| `src/ops/linear_attention` | 13 | 554 | 290 | GDN：varlen recurrence、寄存器 state、grouped prepare、conv varlen、input-conv 的 FP8 接入 |
| `src/ops/linear` | 17 | 120 | 34 | dense 与 FP8 GEMM：256 token tile、FP8 W8A8、BF16 输出头、add 与 split 的分发 |
| `src/ops/softmax_attention` | 7 | 520 | 88 | attention：planner、K/V loader、grouped 路径 |
| `src/ops/kv_cache` | 3 | 101 | 0 | grouped FP8 KV 写 |
| `src/ops/sampling` | 3 | 137 | 34 | sampler 重写 |
| `src/ops/common` | 4 | 69 | 44 | FP8、NVFP4 量化辅助、`silu_approx`、GEMM mainloop |
| `src/ops/normalization` | 1 | 1 | 1 | rmsnorm 的近似 |
| `src/serve` | 2 | 17 | 0 | 指标与日志：`fork_batches`、`forked_branches`、`inflight_checkpoints_*` |
| `src/runtime/contract` | 1 | 3 | 0 | 请求合同的小改动 |
| `include` | 2 | 46 | 0 | 公开头文件：observability 快照字段、grouped attention 接口 |
| `tools` | 1 | 15 | 2 | 转换 recipe：GDN 投影 FP8，输出头 BF16 |
| `tests` | 27 | 907 | 1,143 | 测试：新增 FP8、grouped attention、留存、conv、GDN 用例；删去旧的 capture 行为测试 |
| `bench` | 1 | 7 | 3 | attention bench 增加几何 `d256-h32-kv2` |
| `docs` | 2 | 5 | 3 | recipe 描述的两处小改动 |
| 合计（已跟踪文件） | 124 | 4,689 | 3,270 | |
