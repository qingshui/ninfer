# Spark2_5

Spark2_5 是 `Spark2_5ForCausalLM`（`model_type = spark2_5_text`）的唯一架构入口。
BF16 与 FP8 两个 checkpoint 共用该入口，源码不按 checkpoint 名称分支。
具体类型、接口与支持范围以 `src/models/spark2_5/` 为准，本文只记录取舍和验证入口。

## 架构事实

| 项 | 取值 |
|---|---|
| 层数 | 28 层，每 4 层为 SWA、SWA、SWA、Full（21 SWA + 7 Full） |
| 注意力 | GQA，heads 8 / kv_heads 2，head_dim 256，按 head 乘 `sigmoid(g_proj)` |
| Full 层 rope | rotary 维度 64（前 64 维旋转，其余原样），theta 5e6 |
| SWA 层 rope | rotary 维度 256，theta 1e4 |
| SWA 窗口 | `sliding_window = 512`，可见距离 0..511，即位置 i 看到 `[i-511, i]` |
| MLP | gelu，intermediate 6656 |
| Norm | RMSNorm，eps 1e-6 |
| 输出头 | 与 embedding 共享（tied），产物中没有独立 `lm_head` |
| EOS | token 1 |

SWA 窗口以 `modeling_spark` 为准。Spec 字面写的 `[i-510, i]` 与 HF 实现不一致，测试按 HF 语义校验。

## 配置与绑定

- `parse_text_config` 对 `hidden_act`、`attention_gate`、`sliding_window`、`layer_types`、
  `rope.<kind>.rotary_dim|theta`、`tie_word_embeddings`、`rms_norm_eps` 的非法取值抛出
  `std::invalid_argument`，`what()` 含字段名。
- 权重缺失或 dtype/shape 不匹配时，异常信息含张量名。
- tied 产物的 `output_head` 借用 embedding payload。
- 物化后的设备权重字节通过 observations 的 `device_weights_bytes` 报告（`/metrics` 的 weights/device）。
- BF16 产物所有 Use 为 `A16Only`，不进入 FP8 a8 kernel。FP8 产物的 qkv/out/gate_up/down 为
  `AllowA8` w8a8，`g_proj` 与 `lm_head` 按产物实际表示绑定。

## 产物表示

表示由 `tools/convert/official_recipes.py` 中的 recipe 决定，运行时按产物实际表示绑定。

| recipe | 投影权重 | embedding / 输出头 |
|---|---|---|
| `spark_x2_6_17b_bf16` | BF16 原字，`A16Only` | Q8（tied，共享） |
| `spark_x2_6_17b_fp8` | 发布的 row-scaled FP8 字与 FP32 scale，`AllowA8`（`g_proj` 除外） | Q8（tied，共享） |

`fp8_row_fp32_maxabs` 方法与注册保留在 converter 中，可供 recipe 把 BF16 投影量化为
FP8 row + FP32 scale。改用该表示属于 F8 授权的调优面。改动 recipe 时必须同时更新
`tests/convert/test_spark2_5_real.py` 的投影断言（改为独立宿主参考量化对拍）和
`test_execution.cpp` 的 `hf_case` 契约，使 HF 对拍判据切换到量化档。
`tests/ops/linear/projection/cases_q8.cpp` 覆盖 N131072×K2048 的 Q8 输出头形状。

## KV 与容量

- SWA 层每流每层使用 512 槽 ring（不超过 511 + 一个 chunk）。
- Full 层使用 paged KV，默认 `Fp8E4m3fnGroup128Ue8m0`。
- 容量不足的请求排队，不返回失败。请求结束后 KV 页和 StateImage 归还，
  `ninfer_context_kv_page_groups` 回到 0，pool free 字节回到初值。

## 转换

```bash
python3 -m tools.convert --model <ckpt> --recipe spark_x2_6_17b_bf16 --out <path>.ninfer
python3 -m tools.convert --model <ckpt> --recipe spark_x2_6_17b_fp8  --out <path>.ninfer
```

同样输入两次转换得到相同 sha256。converter 只修 bug，recipe 可改用 q8 等表示。

## 验证入口

数值验证遵循 [Op 开发](op-development.md#数值验证)，测试组织见 [tests/README.md](../../tests/README.md)。

| 目标 | 测试 | 说明 |
|---|---|---|
| 层级与长度 | `ninfer_spark2_5_execution_{layers,lengths}_{bf16,fp8}_test`、`..._window_test` | 2 层 fixture，对 FP64 参考。lengths 覆盖 seq ∈ {1,510,511,512,513,2048}，校验 SWA 窗口、full 层仅旋转前 64 维、逐 head gate |
| 模型级对拍 | `ninfer_spark2_5_execution_hf_{bf16,fp8}_test` | 对 HF `modeling_spark` fp32 前向的逐位置 logits |
| 真实产物 | `ninfer_spark2_5_execution_real_test` | 需要 `NINFER_SPARK2_5_ARTIFACT`。margin ≥ 0.1 位置的 top-1 一致率：A16Only ≥ 0.98，AllowA8 ≥ 0.95 |
| 注册与容量 | `tests/models/spark2_5/test_spark2_5_*.py` | gate 判定、numerics 注册、serving 容量 |

HF 对拍判据由 `hf_parity_criterion` 给出，判定依据写入输出行的 `basis`：

| 表示 | cosine 下限 | top-1 一致率下限 |
|---|---|---|
| BF16 且 A16Only | 0.999 | 0.98 |
| 量化表示或 AllowA8 | 0.99 | 0.95 |

阈值是假设值，量化表示下按 Op 规范放宽，依据见 `basis`。

生成 HF 参考：

```bash
python3 -I tests/models/spark2_5/numerics/make_hf_reference.py \
  --model <ckpt_dir> --variant bf16|fp8 --out <dir>
```

退出码 0 表示写出，77 表示环境不可用。脚本只读 checkpoint 目录。参考文件包含
`hf-<variant>.json` 与 `hf-<variant>.logits.f32`。

运行（GPU 5，不碰 3/4）：

```bash
export LD_LIBRARY_PATH=/home/opt/nvidia_lib:/root/local-deps/usr/lib
CUDA_VISIBLE_DEVICES=5 ctest --test-dir build/rtx5090-release -R 'ninfer_spark2_5_execution_(lengths|hf)_' --output-on-failure
```

HF 对拍和 real 用例标签为 `gpu;real`，耗时长，不进入 `test_command`。
