## 7. 正确性与数值

### 7.1 验证证据

- **端到端 PPL。** canonical artifact 为 3.867882，参考值 3.867259，高 0.0161%（来自 `report.json` 与 `ppl_ref.json`）。阶段 A 终点为 3.864874；中间的 v2 artifact（FP8 GDN 加 BF16 输出头）为 3.869522，高 0.0585%。v2 与 canonical 的 recipe 相同而 PPL 不同，原因没有查清，所以 0.0161% 只是最终 artifact 的实测值，不能当作 FP8 投影本身的误差。
- **生成 sanity。** 20 条 prompt、每条 4 个采样，检查输出非空、没有替换字符、没有重复塌缩，以及文字类型与 sglang 的输出一致，20/20 通过（`sanity.json`）。这是文本合理性检查，不能证明数值正确。
- **锁定的 `ppl_check.sh` 不能作为数值证据。** 它从 artifact 的名字里解析出 "35" 当作结果，无论数值如何都会通过。上面的 PPL 来自 `report.json` 与参考文件，不来自这个门禁。
- **op 级测试，对独立参考实现。**
  - GDN：所有套件对 FP64 参考通过（chunked 342 个用例，varlen 7 个，replay_record 279 个，replay_fold 155 个，另有 replay 记录与 control）。
  - MoE：`ninfer_moe_block_scaled_122b_test` 28 个用例。
  - attention：fp8 52 个用例，五种 KV 格式共 828 个 envelope 更新用例，`causal_groups/*` 覆盖 bf16、int8、nvfp4、k8v4、fp8。
  - conv 与 norm：`causal_conv1d_varlen` 8 个，`causal_conv1d_silu` 205 个，normalization 569 个。
  - linear：FP8 的 B200 测试（`tests/ops/linear/fp8_b200`）、split、add、projection；转换测试 `test_gdn_projections_are_row_scaled_fp8_with_a8_permission`。
  - sampling：重写的 sampler 在 512 个测试行上与旧 kernel 结果完全相同。
- **引擎级真实模型测试。** `preemption_real` 的 4 个场景通过（例如 replay 场景 5 次抢占、5 次恢复、重放 16,296 个 token）；新增 `ninfer_qwen3_5_zero_suffix_real_test`；`grammar_real`、`tools_real`、`score_real` 里过期的断言与挂起问题已修正（7.3）。
- 测试在 `ctest` 里的整体结果有 22 个失败，经逐一分诊，全部归入 7.4 的既有问题。`prefix_real` 里仍有编码旧检查点行为的测试没有运行（5.1 节）。

### 7.2 评分的非确定性：根因与修复

**现象。** 对同一段 1,537 token 的文本重复 6 次 `score_tokens`，约 1,530 个位置的结果不同，最大绝对差达 7.76 nat；两个进程里整段的平均 log-prob 分别为 -0.173364 与 -0.177435。

**根因。** HEAD 的 MoE down epilogue 在 `tokens >= 128` 时使用 fp32 `red.global.add` 合并（`.atomic_output`，`h3072_i1024_e256_k8` 的 dispatch），加法顺序随 L2 atomic 的到达顺序变化。把 `atomic_output` 关掉后，重复评分逐位相同。NVFP4 模型会放大这种舍入噪声（约 0.08% 的位置出现多个 nat 的尖峰），而且批内的 NVFP4 激活 scale 会把各行耦合在一起。

**修复。** B2 的有序合并（6.1 节）。122B 形状从此只用有序模式，评分在多次运行间可重复。其他几何仍保留 atomic 形式，不在本次范围内。
