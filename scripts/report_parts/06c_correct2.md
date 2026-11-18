### 7.4 与本次改动无关的既有失败项

下面这些失败或限制经分诊认为与本次改动无关，没有处理。其中第 5 项在 HEAD 上是否也失败没有确认。

1. `ninfer_platform_targets_test` 与 `ninfer_apps_help_test`：CMake 引用了提交 `934ce1a4` 中已删除的脚本。
2. `ninfer_spark2_5_hf_fixture_build` 及依赖它的测试。
3. NVFP4 的 `mma.sync`、KDA、QSA 与非 122B MoE 的用例抛出 "... is not available on b200"。
4. `ninfer_glm5next_session_test` 的 worker 自旋，用 HEAD 的引擎源码也能复现。
5. native-transactions 的两个阶段：`rewrite_segments_share_prefill_unit`（fixture 的几何问题）与 `capped_prefill_matches_uncapped`（canonical artifact 上 K 为 0.136、V 为 0.130，预算 0.12；原 artifact 上为 0.130 与 0.124；在 HEAD 上未确认）。
6. `ninfer_kimi_delta_attention_bench` 链接失败（未定义的 `kda_input::mxfp8_execution_workspace`），构建需要 `ninja -k 0`。
7. `ninfer_context_store_test` 偶发失败（约 9 次一次，测试自身的竞争）。
8. 锁定的 `ppl_check.sh` 无法判定数值（7.1 节），已记录在 `docs/b200-delivery.md`。

### 7.5 数值路径的变化清单

下面每一项都改变了某个 op 的舍入或求和顺序，或者引入了近似。它们各自的误差没有被单独测量，证据是上面的 op 级测试与端到端 PPL（+0.0161%）。

| 变化 | 位置 | 性质 |
| --- | --- | --- |
| MoE 合并改为固定顺序的 fp32 求和 | B2 | 与 atomic 的求和顺序不同；结果确定 |
| `silu_approx`，共用于 6 处（conv 的 silu 4 处、gated RMSNorm、MoE gate/up） | B1、B4 | 近似的指数与除法；笔记记录最坏相对误差 3.2e-7，此界没有仓库内测试，未注明输入范围 |
| shared expert 在不少于 128 token 时走 BF16 GEMM | B1、B2 | 舍入点与旧路径不同（单次舍入） |
| grouped attention 的 FP32 求和顺序 | B5 | 与 split 加 merge 不同；各组互相隔离 |
| split 启发式 `kMinSplitTarget = 3` | A5 | 单路与 split 的选择变化，求和顺序随之变化 |
| 新 GDN chunk step 的累加 | B9 | 两条 TF32 累加链，与 HEAD 不逐位相同；仍在 FP64 参考的预算内 |
| GDN 投影的 W8A8 | B6 | 权重按行的 E4M3，激活按 token 的 E4M3；FP32 累加，epilogue 里一次 BF16 舍入 |
| 输出头由 Q8 改为 BF16 | B7 | 精度升高，输出头字节增加 |

另有一项推理得出但没有测试的风险：为 NVFP4 量化器引入的倒数乘法形式 `fdiv_rn_by_reciprocal`，在输入为 -0.0 或除法溢出时与 `__fdiv_rn` 的结果应当不同。笔记记录它在 E4M3 scale 的 5.37e9 个样本上与 `__fdiv_rn` 逐位一致，但笔记没有说明样本是否包含 -0.0 与溢出的情形。
