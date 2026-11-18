## 2. 测试方法与口径

### 2.1 模型与负载

- 模型：Qwen3.5-122B，NVFP4 权重。48 层，其中 36 层 GDN（Gated DeltaNet）和 12 层全注意力。hidden 3072，256 个专家，top-8，moe_intermediate 1024，另有一个 1024 的 shared expert。注意力为 32 个 query head、2 个 KV head、head_dim 256；GDN 为 16 个 key head x 128 加 64 个 value head x 128；词表 248,320。
- 负载：2000 条多轮对话记录，平均 prompt 536 token（前缀复用后实际计算约 413 到 462 token）。每个请求 n=4 个采样，一次 parent prefill 后 fork；`max_tokens=15`；客户端并发 130，所以同时有 520 个序列，而显存只够 353 到 363 条 lane（context pool 101 到 104 GiB，每个序列的 GDN State 镜像 144 到 151 MiB）。
- 硬件：1 张 B200（只用 GPU 1），服务时 SM 时钟约 1958 MHz，功耗 850 到 910 W（上限 1000 W）。CUDA 13.4，构建 preset 为 `b200-release`。

### 2.2 闭环吞吐关系

客户端是闭环的：每个请求完成后才发下一个，并发固定为 130。因此吞吐近似为

```
req/s ≈ 130 / 平均 E2EL
```

最终版本 130 / 1.348 s = 96.4（实测 95.26），HEAD 130 / 58.3 s = 2.23（实测 2.21）。差距来自启动与收尾阶段。这个关系说明，优化只有缩短单个请求的端到端延迟才会提高吞吐。

### 2.3 三种测量口径

| 口径 | 工具与窗口 | 能证明什么 | 不能证明什么 |
| --- | --- | --- | --- |
| 端到端 harness | 锁定的客户端、数据集与服务参数，完整 2000 请求 | 吞吐与延迟 | 不能分解到单个 kernel |
| nsys 窗口 | nsys 2026.3.2，`--cuda-graph-trace=node`；最终版本 5 s，HEAD 10 s | flight 内各类 kernel 的时间、GPU 忙闲 | profile 会让运行变慢；HEAD 的窗口只完成 4 个请求，需要归一化（第 4 节） |
| op 微基准与 ncu | 单个 op 的 bench 可执行文件、ncu | 单个 op 的耗时与资源占用 | 只证明 op 性能，不能直接外推到端到端 |

### 2.4 本报告新增的对照测量

为了让"前"和"后"处于同一条件，本报告在同一台机器、同一块 GPU、同一个锁定客户端上重新测了三组，依次运行，任一时刻 GPU 上只有一个服务进程：

| 组 | 服务程序 | artifact | 次数 | 用途 |
| --- | --- | --- | --- | --- |
| HEAD | `3e31d5a4` 的 detached worktree，同样的 `b200-release` preset | 原 artifact（BF16 GDN 投影，Q8 输出头） | 1 | 基线。HEAD 不含 FP8 投影的 shape，只能加载原 artifact |
| 最终，canonical | 当前工作树 | canonical artifact（FP8 GDN 投影，BF16 输出头） | 3 | 交付配置 |
| 最终，原 artifact | 当前工作树 | 原 artifact | 2 | 把 artifact 的贡献单独分出来 |

HEAD 只测了 1 次，因为一次需要 905 s。阶段 A 之后的 3 次数据来自当时记录的 harness 结果，官方 `ninfer_e2e.sh` 的 1 次来自最终验收。flight 形状来自引擎每 5 s 打印的 `throughput |` 日志行，取至少 100 个请求在跑时的中位数。

### 2.5 数据标注

- 无标记：来自 harness 输出的 JSON 或日志，已重新核对。
- `~`：取自开发笔记（压缩前的会话摘要），没有保留对应的 benchmark 文件。
- 开发期的 ncu、nsys 与微基准数字沿用笔记中的值，本报告没有重新测量；各处会注明。
