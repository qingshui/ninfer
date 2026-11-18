## 4. 性能剖析：HEAD 为什么只有 2.2 req/s

### 4.1 flight 的形状

flight 是引擎一次提交给 GPU 的执行批次，可以同时包含若干个请求的 prefill 窗口和若干个 decode 行。图 4 给出三个构建的 flight 形状（引擎遥测，稳态中位数）。

{{FIG:fig4|图 4　一个 flight 装载的内容：HEAD、阶段 A 之后与最终版本}}

- HEAD：中位数为 62 个 prefill token 加 1.6 个 decode 行，每 5 s 约 192 个 flight，平均每个 26 ms。GPU 一次只处理很少的 token。
- 阶段 A 之后：3,341 个 prefill token 加 286 个 decode 行，每个 flight 135 ms。每个 flight 的工作量放大了数十倍，prefill 吞吐从 3.5k 提到 24.4k token/s。
- 最终：3,837 个 prefill token 加 312 个 decode 行，每个 flight 99 ms，prefill 吞吐 39.5k token/s。阶段 B 的 kernel 工作让 flight 更短（135 到 99 ms），同时装载更多 prefill。

图 4 里的 prefill 吞吐与每个 flight 的 token 数来自两个不同的引擎计数器：吞吐来自 `computed_prefill_tokens`（各请求 prefill 进度之和），每个 flight 的 token 数来自 `flight_prefill_tokens`（提交进 flight 的 prefill token 之和）。后者占前者的比例，阶段 A 之后为 95.8%，最终为 96.2%，HEAD 只有 66.6%。所以 HEAD 的两组数字不能互相换算：62 个 token 乘每秒 38 个 flight 是 2.4k token/s，日志里的吞吐是 3.5k token/s。HEAD 缺的那三分之一没有追查来源；HEAD 的 capture 作为独立的 context operation 执行，可能是没有计入 flight 的部分，这一点没有验证。改用 `computed_prefill_tokens` 除以 flight 数折算，HEAD 为 94 个 token/flight，仍比最终版本的 3,682 小约 39 倍，结论不变。比例与折算值保存在 `data/data.json` 各构建的 `telemetry` 里（`flight_counter_share`、`counter_prefill_per_flight`）。

### 4.2 每个请求消耗的 GPU kernel 时间

对三种配置各取一个 nsys 窗口（HEAD 10 s，最终 5 s），把窗口内所有 kernel 的耗时按 kernel 名分到七个类别，再除以窗口内完成的请求数。

{{FIG:fig5|图 5　每个请求的 GPU kernel 时间（ms），按类别}}

读图前先说明归一化方式。最终两组用窗口内完成的请求数（canonical 480 个，原 artifact 411 个）。HEAD 的 10 s 窗口只完成了 4 个请求，样本太少，改用该次运行测得的 2.21 req/s 与窗口的 GPU 忙时占比（0.827）：每个请求的忙时为 0.827 x 1000 / 2.21 = 374 ms，再按各类别在窗口内的占比分摊。三组窗口里 kernel 耗时之和都等于并集忙时（没有 kernel 重叠），所以 HEAD 一侧只是把分母从窗口内完成的 4 个请求换成了测得的吞吐，其余口径与最终版本相同；HEAD 的数值仍属推算值。

观察到的现象：

1. **HEAD 并不空闲。** 窗口内 GPU 并集忙时占比 HEAD 为 82.7%，最终 canonical 为 89.8%，原 artifact 为 91.9%。吞吐差 43 倍，但忙时占比只差 7 个百分点。
2. **差别在每个请求花多少 GPU 时间**，374 ms 对 9.64 ms，相差 39 倍。
3. **按类别看，随 flight 摊薄的部分降幅最大。** MoE 路由专家从 156.4 降到 2.6 ms（60 倍），dense GEMM 从 97.2 降到 2.3 ms（43 倍），MoE 路由与 shared expert 从 51.0 降到 0.5 ms（106 倍），全注意力从 32.4 降到 0.74 ms（44 倍）。随 token 或 row 数增长的部分降幅小得多：GDN decode 的 state 更新只从 2.87 降到 1.70 ms（1.7 倍），因为每个 decode row 每步要读写的 state 量是固定的。

由此可以推断（本报告没有对 HEAD 做权重流量的专门测量）：HEAD 的每个 flight 只有约 64 个 token，却要读取全部层的 dense 权重和大部分专家的权重，GEMM 也只能落在 16、32、64 的小 token tile 上，权重读取成本被很少的 token 分摊。阶段 A 的目标就是让每个 flight 装满。

本报告没有阶段 A 之后的 nsys profile，因此无法把 39 倍在阶段 A 与阶段 B 之间划分。吞吐在两个阶段之间的划分见图 3。

### 4.3 最终版本的 flight 构成与剩余空间

最终 canonical 的 5.15 s nsys 窗口内共有 52 个 flight（按 kernel 之间的间隙划分），平均周期 99 ms。flight 内部平均 92.5 ms（kernel 忙时的中位数 84.5 ms），flight 之间有 51 个超过 1 ms 的间隙，合计 342 ms，约 6.7 ms 一个，是 host 侧的 flight 边界（admission、reserve、提交输出、settle、fork 等）。窗口内求和最大的 kernel 是 MoE grouped GEMM（964 ms）、dense GEMM（939 ms）、GDN decode 的 state 更新（817 ms）、GDN 寄存器 recurrence（451 ms）、grouped attention（313 ms）。

剩余空间主要有两处：GDN decode 的 state 更新已经接近 HBM 带宽（笔记记录约 79% 的峰值，每个请求的流量下限约 1.4 到 1.5 ms，当前为 1.70 ms），继续压缩需要降低 state 精度（第 10 节）；flight 之间约 6.7 ms 的 host 边界约占周期的 7%，目前没有处理。
