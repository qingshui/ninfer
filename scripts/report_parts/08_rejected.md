## 9. 被否决的尝试与无效的尝试

下表中的数字多数是单次运行或开发期的 op 基准，用来说明当时为什么放弃，不是严格的 A/B。

| 想法 | 决定它的数字 | 结论 |
| --- | --- | --- |
| GDN value tile 32 或 16（varlen recurrence） | 50.59 与 29.61 req/s，对照 50.98 | 保留 tile 64 |
| MoE gate 用 TileN=128 | 380k 到 433k 周期，对 TileN=64 的约 350k；4096 token 时 563 对 373 µs | 还原 `plan.cpp` |
| MoE down 的 tile 64 对 128 | 48.24 对 48.07 req/s | 持平，保留默认 |
| MoE scale loader 预取、L2 policy 切换、L2 prefetch | 无收益、无效果、gate/up 变慢（513 对 374 µs，后来证实是外部 GPU 负载造成的假象） | 全部还原 |
| 把 quantizer epilogue 的除法当作 gate/up 的瓶颈 | op 只快了约 1 µs | 真正原因是 gather4 的发射（B1） |
| 静态留存上限 290 到 700，及 lineage 变体 | 57.78、58.99、46.15、38.82、33.65、32.86 req/s | 被 pool 余量控制器取代（图 7） |
| host 缓存层（`host_capacity > 0`） | 83.35 对 85.7 到 86.1 req/s；代价模型与降级停顿 | 默认保持 0 |
| lane 数 | v2 阶段：520 条 75.66 req/s（host 占 92%），300 条 85.57，对照 85.7 到 86.1；59 阶段：256、288、320 条为 58.55、59.17、59.15，对照 58.49 到 59.27 | 300 与默认持平，520 明显变差；保留默认的 361 到 363 条 |
| 关掉 host exposure 记账（NOEXPOSE） | 59.06 对 58.49 | 记账代价不超过约 1%，改为廉价记账 |
| 前缀缓存的容量或策略 | 模拟：256 项时复用率 22.2% 到 22.9%，512 项 28% 到 29.7%，不限 30% 到 32%；受 pool 容量限制（约 290 到 301 个 144 MiB 的共享检查点） | 不改 |
| prefill 与 decode 的 GDN kernel 放两个 stream | 无收益（两者都占满 CTA 槽位） | 不做 |
| GDN chunk 大小改动 | 会同时改变 chunked 与 varlen，受逐位一致合同约束 | 降低优先级 |
| fp16 或 bf16 的 GDN State、bf16 的 MoE contributions、bf16 atomic | 合同变更：FP32 主 State 是公开合同，牵涉约 10 个 op；仅 decode 流量一项估计有 7% 到 8% | 未探索 |
| MoE 的 atomic 合并 | 480.5 对 437.7 µs，且不确定 | 被有序合并取代（B2） |
| GDN 投影改 NVFP4 | PPL 门限 ±1% 下精度风险；转换器没有 BF16 到 NVFP4 的 RTN | 否决 |
| FP8 shared expert | 净收益约 0.4% | 否决 |
| 2-CTA（`cta_group::2`）BF16 GEMM | 最好情况只比 TileN=256 高约 16% | 不做 |
| varlen recurrence 里的 smem swizzle | op 394.8 对 388.5 µs（-1.6%），指令多 31%，flight 内无收益；还会改动与 dgx_spark 共用的 kernel | 还原 |
| `reclaim_order` 的 O(owners^2) 修复 | 87.21、87.27、86.97 对 87.1 到 87.3 | 没有可测的变化，保留 |
| 异步、流水化的 launch-ahead 调度 | 能藏住约 7 ms/flight（约 8%），但需要设备侧 token 回馈、ghost row 与 lane 隔离的重新设计 | 未做 |
| multi-step decode、半 flight A/B 交替、PDL、交换 begin/submit_iteration | 持平到 +1%，或造成 decode-only 与 prefill-heavy flight 交替 | 不做 |
| 多流 fmha | 需要 stream 池与按段 workspace | 未做 |
| 在 serving 下用 ncu | 只抓到 warm-up 的 kernel，或以 LaunchFailed 退出 | 改用独立 bench |
