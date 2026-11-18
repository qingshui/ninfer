# Variant A/B on the final build: V1 gdn_head and V2 moe_atomic

Both variants are the final working tree (uncommitted, on HEAD 3e31d5a4) with one optimization switched back. GPU 1, canonical artifact `/raid0/models/ninfer/qwen35-122b.ninfer`, locked harness.

## Setup

- Variant trees: detached worktrees `.agent-tmp/qwen122b-report/var_gdn_head` and `var_moe_atomic`, overlaid with the 130 changed files of the final tree. `cmp` against the main tree showed 0 mismatches before the edit. After the edit only the files listed below differ.
- Build: `CUDACXX=/usr/local/cuda-13.4/bin/nvcc cmake --preset b200-release`, target `ninfer-serve`. Both EXIT=0 (206 s and 207 s).
- Control: the main tree's final binary (built 12:34), run read-only. The only source file newer than it is `tests/models/qwen3_5/test_native_transactions.cpp`, which is not part of `ninfer-serve`.
- Runs: 9 plain runs, interleaved v1, v2, control for 3 rounds (14:11:59 to 14:22:52), each through `ninfer_bench.sh --require-complete 2000` (GPU_ID=1, PORT=30100, own scratch dir). All 9 passed, 2000/2000 completed, 0 failed. Interleaving is there so that drift over the session hits all three arms.
- Final build earlier today (`ab/runs/final_c1..3`): 95.27, 94.22, 93.85 req/s, mean 94.45, range 1.42.

## Diffs applied

- V1: four files reverted to HEAD, byte-identical to `git show HEAD:<path>`. Changed lines versus the final tree: `recurrence/varlen/recurrence.cu` +227/-379, `recurrence/chunked/recurrence.cu` -4, `recurrence/chunked/launch.h` -3, `gated_delta_net/sources.cmake` +1/-3. `register_state.cu/.cuh` stay in the tree, unreferenced by `sources.cmake`. Effect: HEAD's varlen `recurrence_kernel` (shared-memory snapshot chunk step plus its recurrent route), HEAD's `prepare_kernel` (one block per value head, `__launch_bounds__(256, 2)`), and `chunk_recurrence_kernel` for the chunked route. Diffs: `variants/v1_varlen.diff`, `variants/v1_small.diff`.
- V2: one line added after `.vector = tokens == 1,` in both `src/ops/moe/routed_swiglu/cuda/nvfp4/b200/shapes/h3072_i1024_e256_k8/dispatch.cpp` and `src/ops/moe/topk_softmax_swiglu_moe/nvfp4/b200/shapes/h3072_i1024_e256_k8/dispatch.cpp`: `.atomic_output = tokens >= 128,` (HEAD's selection). Diff: `variants/v2.diff`.

## Throughput (req/s per run)

| round | V1 gdn_head | V2 moe_atomic | control |
| --- | --- | --- | --- |
| 1 | 85.87 | 93.70 | 94.00 |
| 2 | 85.29 | 93.48 | 93.92 |
| 3 | 85.55 | 92.70 | 94.41 |
| mean | 85.57 | 93.29 | 94.11 |
| sample stdev | 0.29 | 0.52 | 0.26 |

| arm | req/s | vs control | vs control % | vs earlier final mean 94.45 | mean E2EL (s) | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| control | 94.11 | | | -0.34 (-0.36%) | 1.365 | 205 | 1111 | 35.99 |
| V1 gdn_head | 85.57 | -8.54 | -9.07% | -8.88 (-9.40%) | 1.503 | 222 | 1258 | 39.69 |
| V2 moe_atomic | 93.29 | -0.82 | -0.87% | -1.15 (-1.22%) | 1.377 | 206 | 1139 | 36.27 |

Per-round difference to the control of the same round: V1 -8.13, -8.62, -8.86; V2 -0.30, -0.44, -1.70 req/s. Full per-run table with p99 and TPOT: `variants/results.md`.

## V1: what the profile shows

One nsys window per build (5 s after a 3 s skip, same client, `ab/prof_driver.sh`). V1: 44.2 flights, 416 requests, 168,757 computed prompt tokens (3,925 per flight). Final (`ab/prof/final_c`): 53.5 flights, 480 requests, 191,577 computed prompt tokens (3,684 per flight). Flights are `recurrent_batch_update_kernel` launches divided by the 36 GDN layers. Requests are `req# done` lines of the serve log inside the window. Profile runs carry nsys overhead and are not throughput claims.

| kernel | V1 ms/flight | V1 ms/request | final ms/flight | final ms/request |
| --- | --- | --- | --- | --- |
| varlen recurrence (`recurrence_kernel` in V1, `register_recurrence_kernel` in final) | 16.905 | 1.795 | 8.425 | 0.939 |
| `prepare_kernel` | 5.744 | 0.610 | 1.300 | 0.145 |
| chunk step (`chunk_recurrence_kernel` / `chunk_register_kernel`) | 0.080 | 0.008 | 0.015 | 0.002 |
| `plan_kernel` | 0.267 | 0.028 | 0.259 | 0.029 |
| **subtotal, kernels in the reverted files** | 22.996 | 2.442 | 9.999 | 1.114 |
| `recurrent_batch_update_kernel` (decode, not touched) | 15.374 | 1.632 | 15.269 | 1.702 |
| `gdn_projected_conv_kernel` (decode conv, not touched) | 1.104 | 0.117 | 1.109 | 0.124 |
| `varlen_output_kernel` + `varlen_state_kernel` (causal-conv varlen, not touched) | 2.109 | 0.224 | 1.955 | 0.218 |
| all kernels in the window | 104.89 | 11.136 | 86.53 | 9.645 |

Raw output with launch counts: `variants/gdn_table.md` (its "GDN kernels total" row also counts the untouched kernels, so use the subtotal above).

- The kernels in the reverted files fall from 22.996 to 9.999 ms/flight (-13.0, -56.5%), from 2.442 to 1.114 ms/request (-54.4%) and from 6.02 to 2.79 ms per 1k computed prompt tokens (-53.6%). Varlen recurrence (-8.48 ms/flight) and prepare (-4.44) make up 12.92 of the 13.0. The chunk step kernel is nearly absent in production flights (288 and 108 launches in the window).
- Kernels V1 does not touch agree between the windows: decode update plus decode conv 16.48 vs 16.38 ms/flight (0.6%), causal-conv varlen kernels 0.552 vs 0.546 ms per 1k computed prompt tokens (1.1%). The same causal-conv kernels differ by 7% per flight (2.109 vs 1.955) because the windows differ in token mix, so per request or per token is the fair normalization for prefill kernels.
- All kernels per request: -1.49 ms (-13.4%). The kernels in the reverted files account for 1.33 ms of it (89%). The other 0.17 ms/request is in untouched kernels and is not attributed (different flight mixes in the two windows).
- Request rate inside the profile windows: 81.9 req/s (V1) vs 93.2 req/s (final), -12.1%. That matches what per-request kernel time and utilization (0.912 vs 0.898) predict (-12.0%). The plain runs show -9.1%. The 3-point gap between profile and plain runs is not explained; nsys overhead and the 5 s window are candidates, neither was tested.

## V2: reading

- With the atomic combine restored for tokens >= 128 the server measured 93.29 req/s against 94.11 for the control: -0.82 req/s (-0.87%). Mean E2EL +12 ms, TTFT +1 ms, TPOT +0.28 ms.
- All three V2 runs (92.70 to 93.70) sit below all three control runs (93.92 to 94.41) and below the three earlier final runs (93.85 to 95.27). The gap is about 0.8 to 1.1 req/s against a within-arm stdev of 0.26 to 0.52, so n=3 per arm does not establish a difference of this size. The data do show that the atomic combine is not faster.
- Reading for the report: the ordered combine costs no measurable throughput. The point estimate favours it by 0.9%, which is not established at n=3.
- Caveat: V2 restores HEAD's `atomic_output` selection, not HEAD's combine code. The atomic path in the final tree is the 4-wide `red.global.add.v4.f32` form with `routed_sum`; HEAD used scalar `atomicAdd`. V2 therefore measures the final kernels with an atomic combine at >= 128 tokens. No accuracy or determinism check was run on V2.

## What these A/Bs support

- GDN kernel group (V1): on the final engine and artifact it is worth +10.0% request throughput (94.11 vs 85.57 req/s, interleaved, n=3 per arm). The three paired rounds all show -8.1 to -8.9 req/s, and the gap is six times the final build's earlier range of 1.42 req/s. The figure belongs to the group as a whole (varlen recurrence, prepare, chunk step, recurrent route). It is not an additive share of the 2.21 to 95.26 climb, because the other optimizations interact with it.
- Ordered MoE combine (V2): no measurable throughput cost.
- Not measured here: recipe (measured earlier on the final code: 83.38 vs 94.45 req/s, original vs canonical artifact), control plane, MoE GEMM, attention, sampling.

## Caveats

- n=3 per arm, one session. The harness checks completion (2000/2000), not output correctness, and no accuracy check was run on V1 or V2.
- Profile windows are 5 s and carry nsys overhead.
- The control is the main tree's binary; the variants are separate builds in other directories. Their sizes differ by 4,096 B (20,136,712 vs 20,140,808); the cause was not checked.
- Kernel names `varlen_output_kernel` and `varlen_state_kernel` belong to `src/ops/linear_attention/causal_conv/varlen.cu`, not to the GDN recurrence, although `gdn_table.py` lists them under GDN.

## Files (under `.agent-tmp/qwen122b-report/`)

- `variants/results.md` and `variants/agg.py`: per-run tables and deltas. `variants/gdn_table.md` and `gdn_table.py`: kernel table.
- `variants/runs/{v1,v2,fc}_{1,2,3}/`: bench JSON and serve logs. `variants/run_variants.sh`, `run_variants.log`, `build_*.log`.
- `variants/v1_varlen.diff`, `v1_small.diff`, `v2.diff`: the applied changes against the final tree.
- `ab/prof/gdn_head/` (nsys report, sqlite, serve log) and `ab/prof/gdn_head.json`.
- The worktrees `var_gdn_head` and `var_moe_atomic` were removed after the runs; the diffs above recreate them.
