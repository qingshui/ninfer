## 8. 代码修改清单

### 8.1 规模与归属

相对 HEAD `3e31d5a4`，工作树有 **124 个已跟踪文件被修改（+4,689 行，-3,270 行）**，另有 **7 个未跟踪的新文件（694 行）**。其中一个测试文件被删除（`tests/models/qwen3_5/test_capture_nonblocking.cpp`，260 行，旧 capture 行为的测试）。测试一侧共删除 1,143 行：799 行来自 `test_native_transactions.cpp`（+200 / -799）按新合同的重写，260 行来自被删除的文件。

暂存区里有 16 个 qwen3_5 的 Session 与测试文件，它们是流水线 T3 阶段（在 `dev/task-0002` 上完成并验证）的 Session 侧合同，包括 `captures_in_flight`、`fork_batch` 与 flight 内检查点行；Engine 侧（T4）缺失，由本次工作补上，并在暂存文件之上又做了未暂存的修改。暂存的 Session 与未暂存的 Engine 是同一项变更，两者的逐行归属无法严格区分，下表按功能区域统计。

### 8.2 按区域

{{TABLE:changes_area}}

### 8.3 新增文件

{{TABLE:changes_new}}

六个 kernel 与分发文件加一个测试文件。主要的 GDN kernel 改动（recurrent 路径与 grouped prepare）在已有文件 `varlen/recurrence.cu` 里，不在新文件里。

### 8.4 变动最大的文件

{{TABLE:changes_top}}

### 8.5 模块边界与合同

- **Engine 与 Session 的分工。** `EngineCore` 负责决定何时分叉、何时发布检查点、留存多少；Session 负责物理上如何分叉与写检查点，通过 `captures_in_flight()`、`fork_batch`、`take_flight_checkpoints` 暴露。qwen4_exp、glm5next、spark2_5 不声明 `captures_in_flight()`，继续走旧路径。这是编译期分支（`if constexpr`），其他模型的 Session 没有改动，但共享的 Engine 代码被改动了，没有在这些模型上验证（第 10 节）。
- **Op 的支持范围由 op 自己判断。** FP8 投影的 shape 支持写在 `linear_backend()` 里（`backend.h`），上层只传递 recipe 给出的格式与激活策略；`A16Only` 与 FP8 权重的组合由 op 以 `invalid_argument` 拒绝。
- **合同保持。** varlen GDN 与单段 chunked 共用同一种算术，输出与检查点逐位相同；FP32 主 state 是公开合同，没有改动；grouped attention 保持各组互相隔离。
- **文档。** 只改了两处 recipe 描述：`docs/b200-qwen35-122b.md` 与 `docs/weight-conversion.md`，说明 GDN 投影改为 FP8、输出头改为 BF16。这两处是实现过程中顺手改的；按仓库 AGENTS.md 的文档维护规则，稳定文档应由用户安排集中梳理，不随实现更新，所以可以直接丢弃。`docs/b200-qwen35-122b.md` 里的几何表已经过期，没有更新。
