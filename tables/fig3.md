| stage | req/s | runs | change |
| --- | --- | --- | --- |
| HEAD 3e31d5a4 | 2.21 | re-measured, 1 run (pipeline recorded 2.14) | Pipeline baseline |
| A1 Engine rework, no helper lane | ~17.3 | run3 on 2000 requests, from notes | fork_batch, in-flight capture, no quiescent iterations; pool-full admission stalls remain |
| A2 Retained-point cap 176 | ~36.1 | cap sweep, from notes | Cap on retained checkpoints removes the admission stalls |
| A3 MoE down epilogue, sampler | 48.07 | run13, 1 run | red.global.add.v4 epilogue, sampler rewrite, slice cap 8192, dense control gate/up |
| A5 Attention split heuristic | 49.95 | run17, 1 run | kMinSplitTarget=3 |
| A6 Conv varlen rewrite | 50.98 | run18, 1 run | causal conv varlen kernel |
| A3/A7 MoE gather, batched completions | 52.01 | run21, 1 run | single-thread gather, 8-warp epilogue, completion batching, batched reserve |
| A5 fmha K/V loader, envelope | 55.99 | run22, 1 run | batched K/V loads, planner contract fix, per-segment envelope |
| A2/A7 Headroom controller, host path | 59.45 | 3 runs: 59.41, 59.66, 59.28 | pool-headroom retention, lineage-first eviction, cheap exposure accounting |
| B1 MoE gate/up gather_input | 65.59 | 1 run | contiguous TMA loads, silu_approx, shared-reciprocal quantizer |
| B2 MoE ordered combine | 67.40 | 4 runs: 67.19 to 67.60 | fixed-order fp32 contributions, deterministic |
| B3 BF16 TileN=256, Add epilogue | 70.60 | 2 runs: 70.79, 70.40 | 256-token tile, vectorized residual epilogue |
| B4 Conv, prepare, rmsnorm tuning | 72.31 | 3 runs: 71.94 to 72.54 | launch bounds, ftz silu, router columns |
| B5 Grouped ragged attention | 75.75 | 3 runs: 75.49 to 76.17 | one launch for all prefill segments |
| B5 Grouped FP8 KV write | 76.09 | 3 runs: 76.03 to 76.12 | one launch writes all KV segments |
| B6 FP8 GDN projections | 83.95 | 3 runs: 83.66 to 84.24 | tcgen05 E4M3 W8A8 for GDN in/out projections |
| B7 BF16 output head | 85.91 | 2 runs: 85.69, 86.13 | lm_head Q8 to BF16 tcgen05 |
| B8 In-flight checkpoint capture | ~87.2 | runs 87.11 to 87.28 | checkpoints ride the final prefill window, one unit per request |
| B9 Register-state GDN step | 90.71 | 1 run | state held in MMA accumulators, 1 sync per chunk |
| B10 Recurrent route, grouped prepare | 94.21 | 3 runs: 93.77 to 94.49 | warp-shared loads in recurrent rows, 4 value heads per prepare block |
| Official run, final sources | 95.26 | ninfer_e2e.sh, 1 run | canonical artifact, diagnostics removed |
