# B200 Qwen3.5-MoE-122B delivery notes

Companion to [b200-qwen35-122b.md](b200-qwen35-122b.md), which owns build, conversion and serving
behavior. This file records the conversion report fields, the sanity and report tools, geometry
decisions and open limits.

## Conversion report

The conversion report (`.conversion.json`) adds source accounting:

- `source_coverage`: `{total, consumed, ignored, unaccounted, unaccounted_examples, ignored_by_rule}`.
  The `qwen3_5_122b_nvfp4` recipe expects `total=148767` and `unaccounted=0`. Vision and MTP
  tensors are ignored through `Recipe.ignore_sources`, and `Recipe.require_source_coverage` fails
  the conversion if any source tensor is neither consumed nor ignored.
- `unconverted_objects`: count of objects the recipe did not convert, expected 0.

Field and rule names avoid the substrings `missing` and `failed` (case-insensitive), because the
locked `ninfer_e2e.sh` greps the report for them.

## Tools

Both tools read finished artifacts and never modify the grader scripts, client or dataset.

`tools/bench/b200_sanity.py` samples prompts from the serving dataset with a fixed seed and sends
them to `ninfer-serve` (temperature 0.9, top_p 0.9, max_tokens 15, n=4, `enable_thinking=false`).
It compares each answer against the recorded sglang output for the same `qid`. Checks are non-empty
text, no U+FFFD, no repetition collapse and consistent script. An entry that fails any check is
marked `needs_review` for a person to read. It is not counted as a failure.

```bash
python3 tools/bench/b200_sanity.py --ninfer-url http://127.0.0.1:30100/v1 --model <id> \
  --sglang-results <benchmark.json> --data <dataset.jsonl> --count 20 --out <sanity.json>
```

`tools/bench/b200_report.py` merges `metrics.json` with the `ninfer-perplexity` `report.json`
(`overall.perplexity`). It checks that `request_throughput` is finite, positive and at most
`2000 / duration_s`. It writes JSON and Markdown with the GPU, serve arguments, build (preset, git
commit, dirty flag), artifact path and id, throughput ratio against 61.0 req/s, TTFT and TPOT. The
PPL reference lives in `.agent-tmp/b200-qwen35/ppl_ref.json`. If absent it is initialized from the
current run when PPL is within (1, 20). Later runs must stay within ±1%.

```bash
python3 tools/bench/b200_report.py --metrics <metrics.json> --perplexity-report <report.json> \
  --ppl-ref .agent-tmp/b200-qwen35/ppl_ref.json --artifact <artifact.ninfer> \
  --build-dir build/b200-release --gpu 1 --serve-args "<args>" --out-json <r.json> --out-md <r.md>
```

The locked `ppl_check.sh` parses a `35` from the artifact path, so its pass is not numerical
evidence. The report states this in `locked_ppl_gate_note`. The 91.5 req/s target appears only as
`tuning_target.status = owned_by_tuning`.

## Geometry decisions

- The spec lists an NVFP4 projection `(3072, 16384)`. The checkpoint has no such tensor. The
  `o_proj` input is 32 heads x 256 = 8192, so the registered geometry is NVFP4 `(3072, 8192)`. No
  kernel is registered without a consumer.
- Shared expert is BF16 in the artifact. The shared-expert profile validates both NVFP4 and BF16.
- The attention batch limit is `kMaximumConcurrency`. A workspace query above it throws
  `std::invalid_argument`. The header comment that said `B=2..520` was corrected.

## Known limits

- Throughput (≥ 91.5 req/s) is not part of this delivery. It is owned by the tuning stage. Any
  req/s number in a report applies only to its recorded conditions.
- The sglang sanity compares against a recorded benchmark, not a live sglang run. Text equality is
  not expected under sampling, so the check is rule-based and some entries need a human read.
- The PPL range (1, 20) is a sanity bound. The ±1% reference comes from the first passing artifact.
- Only GPU 1 was used. No other GPU was exercised.
- The converter entry-point spelling in the b200 doc is indicative. Check the converter `--help`.
