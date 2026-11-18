import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import collections
import os
import subprocess

W0 = os.environ.get("REPORT_DATA", ROOT + "/data")
REPO = os.environ.get("REPO", os.getcwd())
OUT = os.environ.get("REPORT_OUT", ROOT)

AREAS = [
    ("src/runtime/engine", "引擎：采样组与 `fork_batch`、flight 内检查点的发布、留存控制、host 路径、遥测"),
    ("src/models/qwen3_5", "Session 与执行：`fork_batch`、flight 内检查点、binding、request/sequence、grouped attention 的接入"),
    ("src/models/common", "prefix identity 的几何增长 `reserve_for_growth`"),
    ("src/ops/moe", "MoE：grouped GEMM、`gather_input`、有序合并、shared expert、量化器"),
    ("src/ops/linear_attention", "GDN：varlen recurrence、寄存器 state、grouped prepare、conv varlen、input-conv 的 FP8 接入"),
    ("src/ops/linear", "dense 与 FP8 GEMM：256 token tile、FP8 W8A8、BF16 输出头、add 与 split 的分发"),
    ("src/ops/softmax_attention", "attention：planner、K/V loader、grouped 路径"),
    ("src/ops/kv_cache", "grouped FP8 KV 写"),
    ("src/ops/sampling", "sampler 重写"),
    ("src/ops/common", "FP8、NVFP4 量化辅助、`silu_approx`、GEMM mainloop"),
    ("src/ops/normalization", "rmsnorm 的近似"),
    ("src/serve", "指标与日志：`fork_batches`、`forked_branches`、`inflight_checkpoints_*`"),
    ("src/runtime/contract", "请求合同的小改动"),
    ("include", "公开头文件：observability 快照字段、grouped attention 接口"),
    ("tools", "转换 recipe：GDN 投影 FP8，输出头 BF16"),
    ("tests", "测试：新增 FP8、grouped attention、留存、conv、GDN 用例；删去旧的 capture 行为测试"),
    ("bench", "attention bench 增加几何 `d256-h32-kv2`"),
    ("docs", "recipe 描述的两处小改动"),
]


def area(path):
    for a, _ in sorted(AREAS, key=lambda x: -len(x[0])):
        if path == a or path.startswith(a + "/"):
            return a
    return "(其他)"


def main():
    rows = [l.split("\t") for l in open(W0 + "/numstat.txt").read().splitlines() if l.strip()]
    agg = collections.defaultdict(lambda: [0, 0, 0])
    for a, d, p in rows:
        v = agg[area(p)]
        v[0] += 1
        v[1] += int(a)
        v[2] += int(d)
    lines = ["| 区域 | 文件数 | 新增行 | 删除行 | 内容 |", "| --- | --- | --- | --- | --- |"]
    tf = ta = td = 0
    for a, desc in AREAS:
        if a in agg:
            n, ad, de = agg[a]
            lines.append(f"| `{a}` | {n} | {ad:,} | {de:,} | {desc} |")
            tf, ta, td = tf + n, ta + ad, td + de
    lines.append(f"| 合计（已跟踪文件） | {tf} | {ta:,} | {td:,} | |")
    open(OUT + "/tables/changes_area.md", "w", encoding="utf-8").write("\n".join(lines) + "\n")

    new = [l.strip() for l in open(W0 + "/untracked.txt") if l.strip()]
    nl = ["| 新增文件 | 行数 |", "| --- | --- |"]
    total = 0
    for p in new:
        n = int(subprocess.check_output(["wc", "-l", os.path.join(REPO, p)]).split()[0])
        total += n
        nl.append(f"| `{p}` | {n} |")
    nl.append(f"| 合计（{len(new)} 个未跟踪文件） | {total} |")
    open(OUT + "/tables/changes_new.md", "w", encoding="utf-8").write("\n".join(nl) + "\n")

    top = sorted(rows, key=lambda r: -(int(r[0]) + int(r[1])))[:14]
    tl = ["| 文件 | 新增 | 删除 |", "| --- | --- | --- |"]
    for a, d, p in top:
        tl.append(f"| `{p}` | {int(a):,} | {int(d):,} |")
    open(OUT + "/tables/changes_top.md", "w", encoding="utf-8").write("\n".join(tl) + "\n")
    runs_table()
    print(tf, ta, td, total)


def runs_table():
    import json
    d = json.load(open(W0 + "/data.json"))
    rows = ["| 构建 | 各次 req/s | 均值 | 平均 E2EL (s) | 平均 TTFT (ms) | p99 TTFT (ms) | 平均 TPOT (ms) | 时长 (s) |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, label in (("head", "HEAD `3e31d5a4`"), ("phase_a", "阶段 A 之后"), ("final_o", "最终，原 artifact"),
                       ("final_c", "最终，canonical artifact")):
        g = d[key]
        rr = g["runs"]
        rows.append("| %s | %s | %.2f | %.2f | %.0f | %.0f | %.1f | %s |" % (
            label, " / ".join("%.2f" % r["request_throughput"] for r in rr), g["mean_request_throughput"],
            g["mean_mean_e2el_ms"] / 1000 if "mean_mean_e2el_ms" in g else g["mean_e2el_ms"] / 1000,
            g["mean_ttft_ms"], g["mean_p99_ttft_ms"], g["mean_tpot_ms"],
            " / ".join("%.1f" % r["duration_s"] for r in rr)))
    o = d["final_official"]
    rows.append("| 官方 `ninfer_e2e.sh` | %.2f | %.2f | %.2f | %.0f | %.0f | %.1f | %.1f |" % (
        o["request_throughput"], o["request_throughput"], o["mean_e2el_ms"] / 1000, o["mean_ttft_ms"], o["p99_ttft_ms"],
        o["mean_tpot_ms"], o["duration_s"]))
    s = d["sglang"]
    rows.append("| sglang 参照 | %.2f | %.2f | %.2f | %.0f | %.0f | %.1f | %.1f |" % (
        s["request_throughput"], s["request_throughput"], s["mean_e2el_ms"] / 1000, s["mean_ttft_ms"], s["p99_ttft_ms"],
        s["mean_tpot_ms"], s["duration_s"]))
    open(OUT + "/tables/runs.md", "w", encoding="utf-8").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
