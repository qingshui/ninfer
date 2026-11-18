# Qwen3.5-122B（NVFP4）在 B200 上的优化技术与修改总结

- 日期：2026-11-18
- 基线：`3e31d5a4`（流水线产出的 HEAD）；本报告描述的全部改动位于工作树，**尚未提交**。
- 硬件与范围：单卡 B200（GPU 1），锁定 harness（2000 请求，并发 130，n=4，max_tokens=15）。
{{LOCAL}}
- 配套材料：`figures/`（SVG 与 PNG）、`tables/`（图的数据表）、`data/`（原始测量 JSON）、`scripts/`（生成图表的脚本）。
{{/LOCAL}}
{{KU}}
- 配套材料：本文是本地报告的知识库版本，图为 PNG 渲染。SVG 原图、数据表、原始测量 JSON 与生成脚本在本地目录 `.agent-tmp/b200-qwen35/optimization-report/`。
{{/KU}}

## 1. 摘要

目标是把 Qwen3.5-122B 适配到 B200，在保证功能正确的前提下让锁定 harness 的吞吐达到 91.5 req/s。最终官方 `ninfer_e2e.sh` 为 **95.26 req/s**（2000/2000 成功）。同一 harness 的对照重测显示，HEAD 为 2.21 req/s，最终版本三次均值为 94.45 req/s。

| 构建 | req/s | 平均 E2EL | 平均 TTFT | 说明 |
| --- | --- | --- | --- | --- |
| HEAD `3e31d5a4`（本报告重测，1 次） | 2.21 | 58.3 s | 1,765 ms | 流水线自己记录的 HEAD 为 2.14 |
| 阶段 A 之后（3 次均值） | 59.45 | 2.17 s | 361 ms | 控制面、留存控制、首批 kernel 改动 |
| 最终，canonical artifact（3 次均值） | 94.45 | 1.36 s | 203 ms | 三次分别为 95.27、94.22、93.85 |
| 最终，官方 `ninfer_e2e.sh`（1 次） | 95.26 | 1.35 s | 199 ms | PPL 比参考高 0.016% |
| sglang 参照（记录值） | 61.00 | 2.10 s | 1,779 ms | 同一 harness |
| 目标 | 91.5 | | | |

{{FIGNT:fig1|图 1　锁定 harness 的请求吞吐：HEAD、阶段 A 之后、sglang 参照与最终版本}}

### 1.1 主要结论

- **总提升约 43 倍**（2.21 到 94.45）。其中阶段 A（控制面与资源管理）把 2.21 提到 59.45，约 27 倍；阶段 B（kernel 与权重表示）再提 1.59 倍，到 94.45。
- **HEAD 慢的原因不是 GPU 空闲**。nsys 窗口里 HEAD 的 GPU 有 83% 的时间在跑 kernel，最终版本约 90%。差别在每个请求消耗的 GPU kernel 时间：HEAD 为 374 ms，最终版本为 9.6 ms，相差 39 倍。HEAD 的 flight 平均只含 62 个 prefill token 和 1.6 个 decode 行，每次 flight 都要重新读取权重，GEMM 也只能落在 16、32、64 的小 token tile 上（后两点是由 flight 形状推断的，没有单独测量）。
- **阶段 A** 去掉了让 flight 变小的串行点：每个 n 采样请求的隐藏 helper lane、占用 Session 唯一 context operation 槽位的 capture 与 fork、必须等 flight 排空的静默迭代。检查点改为在 prefill flight 内写入，n 个采样通过一次 `fork_batch` 共享冻结的 `StateAnchor` 分叉。随后用按 pool 余量触发的留存控制替换静态留存上限。
- **阶段 B** 做了两类事。kernel 方面：MoE gate/up 与有序合并、dense GEMM 的 256 token tile、grouped attention、GDN 的寄存器状态 chunk step、recurrent 路径与 grouped prepare。权重方面：GDN 的 in/out 投影改为 FP8 W8A8，输出头改为 BF16，artifact 的 payload 减少 2.45 GB。
- **受控 A/B**（最终引擎，canonical artifact，每组 3 次交替运行）：把 GDN recurrence 的四个文件换回 HEAD 的实现，吞吐从 94.11 降到 85.57 req/s，这组 GDN kernel 值约 +10.0%；把有序 MoE 合并换回 atomic，吞吐为 93.29 req/s（-0.87%），n=3 无法确定这个差距，结论是有序合并没有可测的吞吐代价。
- **正确性**：官方 PPL 为 3.867882，参考值 3.867259（+0.016%）；生成 sanity 20/20（文本合理性检查）；GDN、MoE、attention、linear 各 op 的测试对照 FP64 或独立参考实现。MoE 合并改为固定顺序后，同一输入的评分在多次运行间可重复（HEAD 的 fp32 atomic 合并做不到）。

### 1.2 阅读前需要知道的限定

1. 所有改动在工作树中，没有提交；暂存区里有 16 个流水线 T3 阶段留下的 qwen3_5 Session 与测试文件，工作树上叠加了 Engine 与 kernel 的改动。两者是同一项变更，逐行归属无法严格区分，本报告按功能区域描述。
2. 阶段中间值（图 3）是开发期的 harness 测量，不少阶段打包了多项改动，不是单项改动的孤立收益。标 `~` 的值取自开发笔记，没有保留对应的 benchmark 文件。
3. 数值路径有多处变化（第 7 节），只有端到端 PPL（+0.016%）与 op 级对照作为证据。
4. 锁定的 `ppl_check.sh` 门禁无法判定数值：它从 artifact 的名字里解析出 "35" 当作结果，通过不构成数值证据。真实 PPL 来自 `report.json` 与参考文件。
5. 存在与本次改动无关的既有失败项（第 7.4 节），未处理。
6. 性能数字分三种口径：端到端 harness、nsys 窗口、op 微基准。口径不同的数字不能互相换算，各处已注明。

### 1.3 章节

2 测试方法与口径，3 端到端结果，4 性能剖析，5 阶段 A，6 阶段 B，7 正确性与数值，8 修改清单，9 被否决的尝试，10 局限与未探索方向，11 复现，附录。
