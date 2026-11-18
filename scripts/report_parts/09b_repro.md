## 11. 复现

以下命令中的路径是本机环境的值，仓库代码不依赖它们。

**构建与官方验收**（编译、缺 artifact 时转换、PPL 门禁、服务压测一次完成；要求 GPU 1 已用显存不超过 10000 MiB）：

```bash
bash /sgl-workspace/load_test/scripts/ninfer_e2e.sh
```

它依次执行：

```bash
export CUDACXX=/usr/local/cuda-13.4/bin/nvcc
cmake --preset b200-release
cmake --build --preset b200-release --parallel 160
python3 -m tools.convert --model /raid0/models/checkpoint_merged_nvfp4 \
  --recipe qwen3_5_122b_nvfp4 --components text \
  --resource "chat_template.jinja=/raid0/models/checkpoint_merged_nvfp4/chat_template.jinja" \
  --name qwen35-122b --out /raid0/models/ninfer/qwen35-122b.ninfer
bash /sgl-workspace/load_test/scripts/ppl_check.sh
bash /sgl-workspace/load_test/scripts/ninfer_bench.sh --require-complete 2000
```

**只跑吞吐压测。** 锁定脚本把 `metrics.json` 写到当前目录，请在临时目录里运行。可用环境变量 `GPU_ID`、`PORT`、`ARTIFACT`、`SERVE`、`OUT_DIR`、`NINFER_SERVE_EXTRA` 覆盖默认值。

```bash
cd "$(mktemp -d)" && GPU_ID=1 PORT=30100 \
  ARTIFACT=/raid0/models/ninfer/qwen35-122b.ninfer \
  SERVE=<repo>/build/b200-release/apps/ninfer-serve \
  bash /sgl-workspace/load_test/scripts/ninfer_bench.sh --require-complete 2000
```

压测使用的服务参数是 `--max-context 2042 --kv-dtype fp8`，客户端是未修改的 `benchmark_stream_server_ts.py`，参数为并发 130、n=4、`max_tokens=15`、temperature 0.9、top_p 0.9、超时 600 s，数据集为 `infer_data_formatted_online.jsonl`（2000 条）。

**重测 HEAD。** 用 detached worktree 构建 HEAD，并使用原 artifact（HEAD 不含 FP8 投影的 shape）：

```bash
git worktree add --detach <dir> 3e31d5a4
(cd <dir> && cmake --preset b200-release && \
  cmake --build --preset b200-release --target ninfer-serve --parallel 96)
cd "$(mktemp -d)" && GPU_ID=1 PORT=30100 \
  ARTIFACT=/raid0/models/ninfer/orig/qwen35-122b.ninfer \
  SERVE=<dir>/build/b200-release/apps/ninfer-serve \
  bash /sgl-workspace/load_test/scripts/ninfer_bench.sh --require-complete 2000
```

**nsys 窗口。** 自己启动服务并控制窗口，避免锁定脚本结束时杀掉服务导致报告损坏：

```bash
nsys launch --session-new=<名字> --trace=cuda,nvtx --cuda-graph-trace=node \
  <serve> <artifact> --host 127.0.0.1 --port 30100 --device 0 \
  --max-context 2042 --kv-dtype fp8 &
# 等服务就绪并预热，启动压测客户端，跳过启动阶段后：
nsys start --session=<名字> --output=<前缀> --sample=none --cpuctxsw=none --stats=false
sleep <窗口秒数>; nsys stop --session=<名字>
```

窗口取 5 s（最终版本）与 10 s（HEAD，启动 60 s 之后）。

**本报告的图与表。** `scripts/` 里的 `build_all.py` 读取 `data/` 里的测量 JSON，生成 `figures/` 与 `tables/`；`assemble.py` 把报告片段与图表拼成最终的 Markdown。
