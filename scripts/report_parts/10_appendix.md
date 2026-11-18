## 附录

### A. 每个开发阶段的吞吐

{{TABLE:fig3}}

### B. 受控对照与官方运行的逐次结果

所有运行均为 2000/2000 成功，0 失败。

{{TABLE:runs}}

### C. 术语

| 术语 | 含义 |
| --- | --- |
| flight | 引擎一次提交给 GPU 的执行批次，可同时含若干请求的 prefill 窗口与若干 decode 行 |
| lane | Session 里承载一个并发序列的槽位，持有该序列的 GDN State（镜像 144 到 151 MiB） |
| State、`StateAnchor` | 序列的 GDN 递推状态；`StateAnchor` 是 fork 时被 n 个采样共享的冻结状态 |
| 检查点（checkpoint） | 在前缀末尾保存的 State 与 KV 引用，用于前缀复用 |
| context operation | Session 里需要独占槽位的上下文操作，HEAD 的 capture、fork、bind 都属于它 |
| 静默迭代 | 没有 flight 在运行的引擎迭代，HEAD 的 context operation 必须等到它 |
| GDN | Gated DeltaNet，Qwen3.5 里的线性注意力层（36 层） |
| NVFP4 | 4 位浮点权重加分块 scale；本模型的专家与全注意力权重使用 |
| E4M3、W8A8 | FP8 的一种格式（4 位指数、3 位尾数）；权重与激活都用 8 位 |
| tcgen05、TMEM | Blackwell 的 tensor core 指令族与张量内存 |
| TTFT、TPOT、E2EL | 首 token 延迟、每个输出 token 的时间、请求端到端延迟（harness 的定义） |
| req/s | 锁定 harness 报告的 `request_throughput` |

### D. 配套文件

{{LOCAL}}
- `figures/`：图 1 到图 10 的 SVG 与 PNG。SVG 随系统的浅色或深色模式切换配色，每个色块带悬停提示；PNG 是浅色渲染。
- `tables/`：每张图对应的数据表，以及修改清单与逐次运行表。
- `data/`：`data.json`（harness 运行汇总与引擎遥测）、`prof/*.json`（nsys 分析结果）、`variants/`（受控 A/B 的结果与补丁）与 `history_notes.md`（图 3 里标 `~` 的值的来源笔记）。
- `scripts/`：生成图表与拼装本文的脚本、本文的 Markdown 片段（`report_parts/`），以及 `collect/`（从 harness 结果、serve 日志与 nsys 数据库得到 `data/` 的采集脚本；它们读取的原始日志与数据库没有保留，只作方法记录，不能直接重跑）。
{{/LOCAL}}
{{KU}}
本文是本地报告的知识库版本，图 1 到图 10 为 PNG 渲染。完整材料在本地目录 `.agent-tmp/b200-qwen35/optimization-report/`：`figures/`（SVG 与 PNG；SVG 随系统的浅色或深色模式切换配色，每个色块带悬停提示）、`tables/`（每张图的数据表）、`data/`（harness 汇总、nsys 分析结果、受控 A/B 的结果与补丁、图 3 里标 `~` 的值的来源笔记）、`scripts/`（生成图表与拼装本文的脚本；`collect/` 是从 harness 结果、serve 日志与 nsys 数据库得到 `data/` 的采集脚本，它们读取的原始日志与数据库没有保留，只作方法记录）。
{{/KU}}
