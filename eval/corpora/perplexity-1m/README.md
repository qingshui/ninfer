# NInfer 困惑度语料

`ninfer-ppl-1m-v1` 是 NInfer 因果困惑度评测使用的固定文本语料，包含来自四个领域的 16 条相互独立的 UTF-8 流。评分时各流不做拼接；每条流的第一个 token 是唯一没有因果评分的 token。

| 领域 | 来源 | 流数 |
|---|---|---:|
| `english_reference` | WikiText-2 raw test | 4 |
| `english_long_form` | PG-19 test books | 4 |
| `chinese_reference` | 中文维基百科 | 4 |
| `ninfer_code` | NInfer 生产 C++/CUDA 源码 | 4 |

`full` 模式包含全部 16 条流，工作量约一百万 token；`quick` 模式从每个领域选取 `00` 号流，约为前者的四分之一。`manifest.json` 记录流顺序与路径；`provenance/*.jsonl` 把每条输出流映射回源记录或文件字节区间。再分发语料前请先阅读 `THIRD_PARTY_NOTICES.md`。

各流的规模最初用 Qwen3.6 tokenizer（不含特殊 token）粗调，只为了让工作量落在预期的大致区间。tokenizer 的身份与精确 token 数不属于语料合同；每次评测都用其 artifact 的 tokenizer 对提交的文本重新分词并报告所得计数。变更后的语料使用新的语料身份。
