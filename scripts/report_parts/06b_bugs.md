### 7.3 本次工作中发现并修复的缺陷

| # | 现象 | 根因 | 修复 |
| --- | --- | --- | --- |
| 1 | `submit_samples` 竞态 | `sample_group_id` 与 `members` 在请求已对 `pending_` 可见之后才赋值 | 在 `create_request` 里加锁链接，并显式封口采样组 |
| 2 | 旧 engage 代码里 `free_lane()` 反复返回同一条 lane | 没有排除已领取的 lane | 新的 claim 代码扫描互不相同的空闲 lane |
| 3 | `logic_error: fresh admission exceeded the FIFO bypass allowance` | `claim_fork_children` 对每个子 lane 调一次 `scheduler_.admitted()` | `Scheduler::admitted_with` 把一个组算作一次 admission |
| 4 | 新引擎没有发生任何 fork | fork 资格要求无 penalty 且无 rewrite 检查点，过严 | 改为 sample_count 规则（5.1 节） |
| 5 | `cudaErrorGraphExecUpdateFailure`（出现在 `causal_append/.../envelope_updates`） | attention planner 让单路或 split 加 merge 取决于 envelope 的 key tile 数，捕获的图在 envelope profile 之间更新时拓扑改变 | 拓扑只取决于调用形状；回归用例 `chunk_limits{160,2048,160,1024,160}`，828 个用例通过 |
| 6 | replay lane 拿不到 unit permit（流水线 T06 阶段留下的潜在缺陷，`preemption_real` 暴露） | `reserve_planned_units` 因 replay lane 不在 prefill 列表里而跳过它 | 单独排序，并给 `unit.tokens = prefill_chunk`，不占 prefill 预算 |
| 7 | `grammar_real` 与 `tools_real` 的 snapshot 挂起；混合 flight 报 "packed prefill window cannot share an execution" | 保留的 continuation 停在 `frontier == prompt_tokens`，留下空窗口，packed 与 mixed 路径拒绝空窗口 | `build_lane_table` 给不 fork 的 lane `prefill_limit = max(min(chunk, suffix-computed), 1)`；Session 把空窗口单独运行（`operation.packed = Prefill && rows.size() > 1`，`start_prefill` 处理 `end == begin`），第一个 token 从来源的 `continuation_hidden` 采样；新增零后缀真实测试 |
| 8 | 评分不可重复 | fp32 atomic 合并的加法顺序（7.2 节） | 有序合并 |
| 9 | grouped FP8 KV 写在后一个 chunk 上越界 | 二分查找可能落在 `token_base` 超过该列的组上，`local` 为负 | `write_fp8_groups_kernel` 加 `local < 0` 检查 |
| 10 | 分配器占边界采样的 17.7% | prefix identity 里精确大小的 `reserve` 使几何增长失效 | `reserve_for_growth`（6.5 节） |
| 11 | 若干过期的测试断言 | 合同已变 | `score_real` 的断言 `!=` 改为 `<`（`context_device_bytes` 对 pool 管理的 `context_capacity_bytes`）；`tools cancel` 改为解析器在 Cancelled 时保留 Incomplete 的调用；`preemption_real` 的 cancel-replay 断言改为按请求；`kRecutBudget` 保持 0.12，因为 packed prefill 重切的噪声下限约为 rel_l2 0.127 |
| 12 | 构建与测试侧的零散问题 | 多种 | host pass 初始化里改用 `__int_as_float`；conv kernel 里用 `INT_MAX`；`resource_manager.h:1367` 补 `typename`；FP8 shape 文件显式加进 `ninfer_ops`；256 tile 用例阈值；B200 的 fp8 attention bench 增加几何 `d256-h32-kv2` |

另有两项在阶段开始前已存在于 HEAD 之中，不属于未提交改动：`pending_timeout` 默认值由 300 s 改为 590 s（e2e 门禁曾出现 99 与 44 个 HTTP 503 `request_queue_timeout`，被拒绝的请求还会占住客户端槽位 300 s），以及 `ReclaimSelection::best()` 的无界遍历导致的引擎卡死，已加上抢占式的 deadline 与迭代预算（5 次运行无卡死）。
