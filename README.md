# NInfer

> 在 RTX 5090、B200 与 DGX Spark 上对精选模型做原生 C++/CUDA 推理。

NInfer 通过本地 CLI 或兼容 OpenAI 与 Anthropic 的 HTTP API 运行文本与图像/视频生成。它支持单卡执行，也支持跨两台 DGX Spark 的模型并行，带有 CUDA Graph decode、投机解码与共享前缀缓存。一个 Engine 服务一个模型，使用可配置数量的执行 lane：服务默认按模型的上下文容量自动拟合 lane 数（最多 1024 条，各模型上限不同），lane governor 按 decode 节拍约束常驻 lane 以优先保证延迟；显式 `--max-concurrency N` 固定 N 条 lane（见[服务指南](docs/serving.md#execution-behavior)）。

## 支持的模型与部署

| 模型 | 架构 | 部署 | 投机解码后端 |
|---|---|---|---|
| Qwen3.6-27B / Qwen3.8-27B | Qwen3.5 Dense | RTX 5090 | MTP；Qwen3.8-27B 配套权重可用 DFlash2 |
| Qwen3.6-35B-A3B | Qwen3.5 MoE | RTX 5090，专家常驻或 CPU offload | 配套权重可用 MTP 或 DFlash |
| Qwen3.5-MoE-122B | Qwen3.5 MoE | B200 | — |
| Qwen3.8-Flash-Next | Qwen4Exp | RTX 5090 + CPU 专家 offload；单或双 Spark | MTP |
| GLM-5.3-Flash | GLM5Next | 双 Spark | MTP |
| Spark2_5 17B | Spark2_5 | RTX 5090 | — |

除 Qwen3.5-MoE-122B 只支持文本外，其余均支持文本、视觉、工具调用、受约束输出与离线困惑度评分。Spark2_5 仅文本，且没有已发布的 artifact；用 [Spark 配方](docs/weight-conversion.md#spark2_5)转换本地 `Spark2_5ForCausalLM` checkpoint。Qwen3.5-MoE-122B 同样没有已发布 artifact；用 [`qwen3_5_122b_nvfp4` 配方](docs/weight-conversion.md)转换本地 ModelOpt NVFP4 checkpoint，其在 B200 上的编译、部署与验证流程见 [Qwen3.5-MoE-122B B200 指南](docs/qwen35-122b-b200.md)。视觉与投机解码要求 artifact 内含相应组件并在启动时开启；评分只加载 Text。CPU 专家 offload 适用于 RTX 5090 目标；Spark 执行把部分权重留在 GPU。Flash Next 的 offload 还需要较大系统内存存放专家权重，见 [CLI 指南](docs/cli.md#cpu-expert-offload)。

双 Spark 的 Flash Next 使用 TP2，可选 EP2。GLM 使用 TP2 加 attention DP2，可选 EP2。[部署指南](docs/startup-configuration.md#distributed-deployment)解释这些设置与 worker 启动；[示例配置](examples/configs/)覆盖常见部署，包括视觉、FP8 KV、MTP3 与优化后的 draft head。

## 构建

在 Linux 上原生构建：RTX 5090 用 x86-64 / `sm_120a`，B200 用 x86-64 / `sm_100a`，DGX Spark 用 AArch64 / `sm_121a`。依赖：支持目标架构的 CUDA toolkit、CMake 3.28+、C++20 宿主编译器、Ninja、`pkg-config`、libjpeg 开发头文件、FFmpeg 开发库（`libavformat`、`libavcodec`、`libavutil`、`libswscale`）、`hwloc >= 2.5`、`liburing >= 2.5`、`libcurl >= 7.85`。CUDA 13.4 是开发工具包；CMake 会检查所请求的架构。

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer
cmake --preset rtx5090-release
cmake --build build/rtx5090-release -j
```

B200 用 `b200-release` 与 `build/b200-release`；单台 Spark 用 `dgx-spark-release` 与 `build/dgx-spark-release`。两台 Spark 需要安装 NCCL 开发头文件与库，并在**两个节点**上都构建：

```bash
cmake --preset dgx-spark-release -DNINFER_ENABLE_DISTRIBUTED=ON
cmake --build build/dgx-spark-release -j
```

产物是 `apps/ninfer`、`apps/ninfer-serve`、`apps/ninfer-perplexity`，开启分布式后还有 `apps/ninfer-worker`。请从构建树里运行；移动输出时保持 `lib/` 目录与 `apps/` 同级。没有可安装的 C++ SDK。

产品 preset 不含测试与基准。`rtx5090-dev` / `b200-dev` / `dgx-spark-dev` preset 同时开启两者；对应的 `-ops` preset 只构建底层库与其检查。开发选项见[构建组织](docs/maintainer/build-system.md)。

## 获取权重

Engine 读取 v3 `.ninfer` artifact，其中包含权重、配置、tokenizer、chat template 与组件资源。官方下载如下：

| 模型 | 权重 | Artifact | 下载与模型卡 |
|---|---|---|---|
| Qwen3.6-27B | `groupwise-int` | `qwen3_6_27b.ninfer` | [Qwen3.6-27B](https://huggingface.co/neroued/Qwen3.6-27B-NInfer) |
| Qwen3.6-27B | `nvfp4` | `qwen3_6_27b_nvfp4.ninfer` | [Qwen3.6-27B NVFP4](https://huggingface.co/neroued/Qwen3.6-27B-nvfp4-NInfer) |
| Qwen3.8-27B | `groupwise-int` | `qwen3_8_27b.ninfer` | [Qwen3.8-27B](https://huggingface.co/neroued/Qwen3.8-27B-NInfer) |
| Qwen3.8-27B | `nvfp4` | `qwen3_8_27b_nvfp4.ninfer` | [Qwen3.8-27B NVFP4](https://huggingface.co/neroued/Qwen3.8-27B-nvfp4-NInfer) |
| Qwen3.6-35B-A3B | `groupwise-int` | `qwen3_6_35b_a3b.ninfer` | [Qwen3.6-35B-A3B](https://huggingface.co/neroued/Qwen3.6-35B-A3B-NInfer) |
| Qwen3.8-Flash-Next | 混合 NVFP4/Q8/NF4 | `qwen3_8_flash_next.ninfer` + 3 个分卷 | [Qwen3.8-Flash-Next](https://huggingface.co/neroued/Qwen3.8-Flash-Next-NInfer) |
| GLM-5.3-Flash | 混合 NVFP4/Q8 | `glm5_3_flash.ninfer` + 5 个分卷 | [GLM-5.3-Flash](https://huggingface.co/neroued/GLM-5.3-Flash-NInfer) |

```bash
hf download neroued/Qwen3.8-27B-nvfp4-NInfer \
  qwen3_8_27b_nvfp4.ninfer --local-dir models
```

下载 Flash Next 或 GLM 时要连同全部存储分卷：

```bash
hf download neroued/Qwen3.8-Flash-Next-NInfer \
  --include 'qwen3_8_flash_next.ninfer*' \
  --local-dir models/Qwen3.8-Flash-Next-NInfer

hf download neroued/GLM-5.3-Flash-NInfer \
  --include 'glm5_3_flash.ninfer*' \
  --local-dir models/GLM-5.3-Flash-NInfer
```

Flash Next 共 101.00 GiB、四个文件；GLM 共 178.33 GiB、六个文件。所有分卷都要与入口文件放在一起，调用 NInfer 时只给入口路径。两台 Spark 部署时每个节点都要放全套；按 rank 切分发生在加载阶段。模型卡描述了各自的混合精度、组件与部署选项。

自定义权重见[权重转换](docs/weight-conversion.md)。旧的 v2 文件可以[在本地升级](docs/weight-conversion.md#upgrade-an-existing-v2-artifact)。

## 运行

在 RTX 5090 上跑一个请求：

```bash
./build/rtx5090-release/apps/ninfer models/qwen3_8_27b_nvfp4.ninfer \
  --prompt "Explain prefill and decode briefly." \
  --max-context 8192 --max-tokens 1024 --kv-dtype fp8
```

回答内容输出到 stdout；推理过程、启动信息与结果报告输出到 stderr。图像/视频输入用 `--messages FILE` 加 `--vision`。详见 [CLI 指南](docs/cli.md)。

常驻服务请编辑一个部署示例后启动：

```bash
cp examples/configs/rtx5090-qwen3_8-27b-nvfp4.toml ninfer.local.toml
# 把 model.path 改成你的 artifact。
./build/rtx5090-release/apps/ninfer-serve --config ninfer.local.toml
```

该示例开启了视觉、FP8 KV、MTP3 与优化后的 draft head。它把每请求上下文上限设为 240,000 token；实际共享的 KV/State 容量取决于可用显存。启动日志会报告 Context 池大小与估算的 token 容量。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"Hello!"}],"max_tokens":1024}'
```

服务默认只监听回环地址。需要局域网访问时设置 `server.host = "0.0.0.0"`，并按需配置 `server.api_key`。[服务指南](docs/serving.md)覆盖端点、工具调用、结构化输出、流式、指标与请求日志。

两个程序都接受 `--config FILE`；显式命令行值优先于配置文件。根目录的 [ninfer.example.toml](ninfer.example.toml) 列出全部字段；[部署示例](examples/configs/)提供更有针对性的起点。[启动配置](docs/startup-configuration.md)解释默认值、资源控制与分布式启动。个人 TOML 文件被 Git 忽略。

## 上下文与前缀复用

`max_context` 限制一个请求的输入加生成 token 总量。`context_capacity` 控制一个独立的、启动时定容的字节池（每个 rank 一份），供持续增长的 KV 历史与续写状态共享。设为 `auto` 时，池的大小用权重与执行资源规划之后的剩余预算；它不乘以并发数，逻辑上下文上限也不保证每个活跃请求都能同时达到该长度。

服务会精确保留文本与多模态前缀，包括续写它们所需的状态。内存吃紧时可以回收缓存前缀，或暂停活跃请求等待恢复/重放。RTX 5090 还能把上下文保留在锁页 Host 内存；Spark 使用 Device 上下文存储。调优见[资源控制](docs/startup-configuration.md#context-and-memory)。

## 性能

以下记录的压测使用 RTX 5090 与各结果页标注的引擎版本。[性能索引](docs/performance.md)链接到每个模型的运行记录与[测量规则](docs/performance/methodology.md)；下面的表格摘自这些详细结果。压测中 Qwen3.8 使用 FP8 E4M3 row-256 KV，Qwen3.6 使用 INT8 group-64 KV。这些测量保留原始缓存格式与资源配置，不是对当前 Spark 或分布式路径的测量。

### 并发 MTP3 decode

饱和 decode 使用 CUDA Graphs、MTP3，且每个活跃请求一次生成 8,192 token。吞吐取完整时间区间内实际 decode 批等于配置并发的累计已提交 decode token。验收覆盖完整的请求波次；下表速率是稳定 decode（tok/s）。

| 模型配置 | C=1 tok/s / 接受率 | C=2 tok/s / 接受率 | C=4 tok/s / 接受率 | C=8 tok/s / 接受率 |
|---|---:|---:|---:|---:|
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `groupwise-int` | 185.8 / 68.2% | 247.0 / 69.0% | 309.5 / 68.4% | 535.0 / 68.3% |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#decode-saturation) `nvfp4` | 202.4 / 69.3% | 399.7 / 71.4% | 699.7 / 69.3% | 1,146.9 / 68.6% |
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#decode-saturation) `groupwise-int` | 642.5 / 68.6% | 907.0 / 66.3% | 1,213.5 / 69.6% | 1,380.7 / 68.0% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `groupwise-int` | 136.5 / 44.4% | 253.0 / 45.2% | 398.1 / 46.1% | 582.4 / 46.4% |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#decode-saturation) `nvfp4` | 147.7 / 46.2% | 291.0 / 48.7% | 522.3 / 45.8% | 922.4 / 46.1% |

### 单请求服务

串行服务语料使用 CUDA Graphs、1,024 token 的 prefill chunk 与预热后的五个固定种子。每个已发布配置保留一个短 prefill、一个极端 prefill 与一个结构化输出 MTP3 测点；完整上下文与场景矩阵从下面各模型链接。

| 模型配置 | 7,680 token prefill | 260,096 token prefill | 结构化 MTP3 decode |
|---|---:|---:|---:|
| [Qwen3.6-35B-A3B](docs/performance/qwen3.6-35b-a3b.md#single-request-speculative-decode) `groupwise-int` | 17,705.4 tok/s | 5,247.0 tok/s | 779.6 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `groupwise-int` | 3,218.1 tok/s | 1,614.8 tok/s | 193.0 tok/s |
| [Qwen3.6-27B](docs/performance/qwen3.6-27b.md#single-request-speculative-decode) `nvfp4` | 11,191.5 tok/s | 2,510.6 tok/s | 252.2 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `groupwise-int` | 3,331.9 tok/s | 2,139.2 tok/s | 214.7 tok/s |
| [Qwen3.8-27B](docs/performance/qwen3.8-27b.md#single-request-speculative-decode) `nvfp4` | 12,819.1 tok/s | 4,016.3 tok/s | 231.7 tok/s |

## 评测

下面的 27B/35B 能力分数通过 NInfer 的 OpenAI 兼容服务路由测得，开启思考模式、MTP3 与 EvalScope 1.9.0（0-shot、规则打分、每题一个样本）：

| 模型配置 | AIME 2025 | AIME 2026 | GPQA-Diamond | ERQA | RealWorldQA |
|---|---:|---:|---:|---:|---:|
| [Qwen3.6-27B groupwise-int](model-cards/Qwen3.6-27B-NInfer/README.md) | 86.67% | 93.33% | 86.87% | — | — |
| [Qwen3.6-27B NVFP4](model-cards/Qwen3.6-27B-nvfp4-NInfer/README.md) | 93.33% | 93.33% | 84.34% | — | — |
| [Qwen3.6-35B-A3B groupwise-int](model-cards/Qwen3.6-35B-A3B-NInfer/README.md) | 90.00% | 90.00% | 85.35% | — | — |
| [Qwen3.8-27B groupwise-int](model-cards/Qwen3.8-27B-NInfer/README.md) | 96.67% | 96.67% | 87.37% | 66.25% | 82.22% |
| [Qwen3.8-27B NVFP4](model-cards/Qwen3.8-27B-nvfp4-NInfer/README.md) | 96.67% | 96.67% | 90.40% | 66.25% | 83.53% |

Qwen3.6 各行用 temperature 0.6、presence penalty 1.0；Qwen3.8 各行用 temperature 1.0、presence penalty 0.0。多模态评测使用 `--vision` 与 81,920 token 上下文上限；文本评测使用 262,144 token，Qwen3.8-27B NVFP4 为放下 RTX 5090 的权重改用 252,928 token。每个分数是每题一个样本；模型卡里有对/总题数与评测备注。

[Flash Next](model-cards/Qwen3.8-Flash-Next-NInfer/README.md#evaluation) 在两台 Spark 上单独评测，使用 TP2/EP2、BF16 KV、视觉、MTP3、优化后的 draft head 与 EvalScope 1.10.0：

| 基准 | 准确率 | 对 / 计分样本数 | 每题重复 |
|---|---:|---:|---:|
| GPQA-Diamond | 91.79% | 727 / 792 | 4 |
| AIME 2026 | 97.78% | 88 / 90 | 3 |
| ERQA | 68.00% | 272 / 400 | 1 |
| RealWorldQA | 85.75% | 656 / 765 | 1 |
| MMMU-Pro（标准 10 选项） | 80.81% | 1,398 / 1,730 | 1 |

这些是样本平均准确率，不是 pass@k。模型卡记录了生成设置，并把截断/未解析样本计入总数。视觉分数保留原始压测的解码与打分行为。

## Docker

仓库自带的 Dockerfile 构建 RTX 5090 的 CLI 与服务。它要求宿主机装有 NVIDIA Container Toolkit：

```bash
docker build --tag ninfer:local .
docker run --rm --gpus '"device=0"' --publish 8080:8080 \
  --volume "$PWD/models:/models:ro" ninfer:local \
  ninfer-serve /models/qwen3_8_27b_nvfp4.ninfer \
  --host 0.0.0.0 --max-context 8192 --kv-dtype fp8
```

## 文档

- [部署配置](examples/configs/)与[启动参数](docs/startup-configuration.md)
- [CLI](docs/cli.md)、[HTTP 服务](docs/serving.md)与[困惑度](docs/perplexity.md)
- [权重转换](docs/weight-conversion.md)
- [性能记录](docs/performance.md)、[基准测试](bench/README.md)与[评测](eval/README.md)
- [文档索引](docs/README.md)与[贡献指南](CONTRIBUTING.md)

## 许可证

NInfer 基于 [Apache License 2.0](LICENSE) 许可。

27B 与 35B-A3B 的 artifact 来源采用 Apache-2.0（见其模型卡）。Flash Next 把 [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) 的源与 [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) 的权重组合。GLM 的源权重使用 [MIT](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/LICENSE)。模型许可与 NInfer 的软件许可相互独立；第三方依赖在 `third_party/` 下保留各自许可文件。
