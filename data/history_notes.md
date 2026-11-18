# Optimization history, Qwen3.5-122B NVFP4 on B200: HEAD 3e31d5a4 (2.14 req/s) to 95.26 req/s

Sources: the 27 compaction summaries (history/sum_01..27.txt) plus targeted greps of the raw transcript. Labels: **exact** = a harness JSON/log line printed in the transcript or in /raid0/models/ninfer/bench_results; **approx** = taken from my own summaries or notes. All uncommitted work is on top of HEAD 3e31d5a4. Throughput relation used throughout: closed loop, c=130, so req/s ~ 130 / mean E2EL.

## 0. Framing

- **Phase 0, autodevx pipeline (in HEAD, 12 commits a801fe65..3e31d5a4).** T1-T5 done; e2e gate passed 2000/2000 at 1.44 req/s; tune round 1 best 2.14 req/s; tune round 2 HALT (acceptance_defect: T3/T4 split contradicts the whole-repo rtx5090-release gate). The pipeline's T3 (Session-side in-flight capture and fork_batch) was complete and verified on dev/task-0002; it was squash-merged into the index (the 16 staged qwen3_5 Session/test files). The engine side (T4) was missing, so HEAD's engine is the old one (helper lane, context operations, quiescent iterations).
- **Phase A (this conversation, up to about 59.4 req/s):** T4 engine rewrite, retention control, MoE/sampler/attention/conv kernel work, host-path trimming. Measured on the original artifact (BF16 GDN, Q8 head).
- **Phase B (after the Stop hook said the 91.5 target was unmet, about 59.4 to 95.26):** MoE gate/up and combine, dense GEMM tiles, grouped attention, FP8 W8A8 GDN projections, BF16 head, in-flight capture refinement, register-state GDN, recurrent-route and grouped prepare.
- **Artifact lineage.** orig/qwen35-122b.ninfer (BF16 GDN projections, Q8 head, payload 74,757,843,968 B, 69.6 GiB; PPL reference 3.867259) was used for everything up to 76.1 req/s. qwen35-122b-fp8gdn.ninfer (FP8 GDN + Q8 head, experimental) gave 83.95. qwen35-122b-v2.ninfer (FP8 GDN + BF16 head, 67.3 GiB, PPL 3.869522 = +0.0585%) gave about 86. The canonical qwen35-122b.ninfer (id d1c45569..., payload 72,305,503,232 B, PPL 3.867882 = +0.0161%) was regenerated from the same final recipe and gave 94.0-94.8 and the official 95.26. The fp8gdn and v2 files were deleted.
- **Workload:** 2000 multi-turn chat records, mean prompt 536 tokens (about 413-462 computed after reuse), n=4 samples per request (fork after one parent prefill), max_tokens 15, about 35 output tokens per request over 4 branches, 130 x 4 = 520 sequences against 353-363 lanes (memory-fitted: context pool 101-104 GiB, GDN State image 144-151 MiB per sequence).
- **Model:** 48 layers = 36 GDN + 12 full attention, hidden 3072, 256 experts top-8, moe_intermediate 1024, shared expert 1024, 32 q heads / 2 KV heads / head_dim 256, GDN 16 key heads x128 + 64 value heads x128, vocab 248320.
- **Hardware note:** 1 B200 (GPU 1 only), SM clock about 1958 MHz in serving, 850-910 W against a 1000 W cap.

## 1. Throughput milestones

Columns: label | req/s | mean TTFT ms | mean E2EL ms | what changed | exactness. All runs 2000/2000 and 0 failed unless noted.

### 1a. Phase 0 and phase A (original artifact)

| label | req/s | TTFT | E2EL | change | exact? |
|---|---|---|---|---|---|
| e2e gate, benchmark_20261116_233426 | 1.44 | 2603 | 89787 | pipeline gate (1393.6 s) | exact |
| benchmark_20261116_230710 | 1.75 | 1395 | 73909 | earlier harness run (1144.8 s), not labeled in the summaries | exact |
| **HEAD / tune round 1, benchmark_20261117_000202** | **2.14** | 2489 | 60263 | pipeline best (935.2 s) | exact |
| run1 | 10.32 | | | new T4 engine, no forks (fork eligibility too strict), 300-request subset | approx |
| run3 | 32.95 (300 subset) / 17.32 (2000) | | | forks fixed; full run starved by pool-full admission stalls | approx |
| run7 | 34.23 | | | `--no-prefix-reuse`: GPU-bound, shows the stall was host/cache side | approx |
| run8 | 32.06 | 806 | 4031 | retention cap + branch-0-only continuation publishing | exact |
| cap sweep 44/88/176 | 34.87 / 35.34 / 36.13 | 707 / 676 / 689 | 3702 / 3654 / 3570 | unified retained-point cap lanes/2 | approx (176: exact 688.6/3570.1) |
| run13 | 48.07 | 506 | 2682 | MoE down epilogue (red.v4), sampler rewrite, slice cap 8192, dense control gate/up | exact |
| run14 | 48.24 | 507 | 2673 | down tile 64 (wash, default kept) | exact |
| run16 | 48.74 | 506 | 2643 | retain cap 240 | exact |
| run17 | 49.95 | 501 | 2580 | attention kMinSplitTarget=3 | exact |
| run18 | 50.98 | 480 | 2527 | conv varlen rewrite | exact |
| run19 / run20 | 50.59 / 29.61 | 480 / 3284 | 2547 / 4287 | GDN value tile 32 / 16 (rejected) | exact |
| run21 | 52.01 | 476 | 2477 | MoE gather + 8-warp epilogue, engine completion batching, batched reserve | exact |
| run22 | 55.99 | 446 | 2299 | fmha loader, planner contract fix, per-segment envelope (cap 240) | exact |
| run23 | 58.99 | 417 | 2184 | cap 330, lineage-first eviction (3% skipped checkpoints) | exact |
| run24 / run27 | 57.78 / 57.33 | 434 / 437 | 2228 / 2247 | cap 290 with lineage-first / cap 330 with lineage-first off | exact |
| run25 / 26 / 28 / 29 | 33.65 / 32.86 / 46.15 / 38.82 | 1274 / 1261 / 595 / 911 | 3813 / 3918 / 2793 / 3323 | cap 480 / 700 / 380 / 420: throughput cliff (pool full) | exact |
| run30-33 | 57.62 / 58.10 / 57.65 / 57.69 | ~430 | ~2230 | pool-headroom divisor 10 / 16 / 8 / 12 | exact |
| run34 | 58.49 | 427 | 2200 | pool-headroom divisor 20 (chosen default) | exact |
| run35 | 59.18 | 421 | 2175 | divisor 24 (persistent skips, not chosen) | exact |
| run36 / run37 | 59.06 / 58.49 | 420 / 421 | 2181 / 2200 | host exposure accounting off (upper bound) / control | exact |
| run38 | 59.27 | 424 | 2172 | cheap exposure accounting (cached snapshot, O(1) membership) | exact |
| run39 / 40 / 41 | 59.17 / 58.55 / 59.15 | 744 / 909 / 586 | ~2170-2188 | max-concurrency 288 / 256 / 320 (flat) | exact |
| run42 | 59.28 | 372 | 2173 | completion flush after submit, accounting fix, TEMP hooks removed | exact |
| release build, 3 runs | 59.41 / 59.66 / 59.28 | ~364 | ~2167 | final phase-A tree (p99 TTFT ~1887 ms, TPOT ~55.2 ms, 33.5-33.7 s); PPL 3.864874; sanity 20/20 | exact |

### 1b. Phase B (locked harness; run labels are the harness run numbers or my scratch labels)

| label | req/s | TTFT | E2EL | change | exact? |
|---|---|---|---|---|---|
| (phase-A end) | 59.45 mean | 364 | 2167 | start of phase B | exact |
| single run | 65.59 | 328.0 | 1962.4 | MoE gather_input + contiguous 2D TMA + silu_approx + smem offsets + shared-reciprocal NVFP4 quantizer (atomic combine); p99 1718 ms, TPOT 49.74, 30.49 s | exact |
| runs 1-4 | 67.58 / 67.60 / 67.19 / 67.23 | 328 / 319 / 321 / 327 | 1904 / 1903 / 1916 / 1914 | MoE ordered vectorized combine, ftz-silu in conv | exact |
| runs 5-6 | 70.79 / 70.40 | 304.8 / 301.0 | 1817.7 / 1827.2 | Add-epilogue vectorization + BF16 TileN=256 (release #3); TPOT 46.3/46.6, p99 1503/1488 | exact |
| runs 7-9 | 72.54 / 72.46 / 71.94 | 303 / 305 / 305 | 1773 / 1775 / 1786 | build #5: varlen conv, ftz silu, rmsnorm approx, router columns, GDN prepare launch bounds (minBlocks 8) | exact |
| runs 10-12 | 75.49 / 75.59 / 76.17 | 305 / 301 / 303 | 1702 / 1700 / 1687 | build #6: grouped (ragged) causal attention; TPOT 42.1-42.5 | exact |
| runs 13-15 | 76.03 / 76.12 / 76.12 | 315 / 315 / 310 | 1691 / 1688 / 1688 | build #7: grouped FP8 KV write; TPOT 41.5-41.9 | exact |
| runs 16-18 | 83.66 / 84.24 / 83.94 | 239 / 242 / 241 | 1536 / 1526 / 1530 | build #8 + FP8 GDN projections (Q8 head artifact); TPOT 39.9-40.3; profiled run 83.85 | exact |
| prof5 / 6 / 7 (nsys) | 85.00 / 84.82 / 85.36 | 255 (prof5) | 1512 | artifact v2 (+BF16 head) under profiling | exact (prof5), approx (others) |
| v2 unprofiled | 85.69 / 86.13 | 253 / 246 | 1499 / 1491 | artifact v2; also 83.35 with host cache tier (rejected), 85.57 with 300 lanes (wash) | exact |
| twc1 / twc2 | 87.28 / 87.11 | 215 / 221 | 1472 / 1476 | in-flight capture refinement ("Approach C"); +1.7% | exact |
| rcm1-3 | 87.21 / 87.27 / 86.97 | 224 / 212 / 225 | 1474 / 1474 / 1478 | reclaim_order O(owners^2) fix: no measurable change | exact |
| swz1-3 | 87.66 / 86.97 / 85.83 | | | smem swizzle experiment in varlen recurrence (reverted): within noise | exact |
| reg1 | 90.71 | 211.7 | 1417.3 | register-state chunked GDN step; p99 TTFT 1122 ms, TPOT 37.36 | exact |
| prof11 (nsys) | 90.80 | 217.0 | 1415.1 | same build under profiling | exact |
| trace1 / trace3 | 91.42 / 89.10 | 206.6 / 161.6 | 1406 / 1443 | runs with the diagnostic D2H trace patch; not representative | exact |
| prep1 / 2 / 3 | 94.36 / 93.77 / 94.49 | 210 / 206 / 205 | 1361 / 1371 / 1360 | recurrent-route rewrite + grouped prepare; TPOT 35.6-36.1, p99 1083-1168 | exact |
| final1 / 2 / 3 | 94.65 / 94.79 / 94.01 | 210 / 204 / 203 | 1358 / 1355 / 1367 | canonical regenerated artifact, same code; p99 ~1.11 s, TPOT 35.5-36.0, ~21.1 s | exact |
| **official ninfer_e2e.sh** | **95.2644** | 199.4 | 1347.65 | final sources (after 12 diagnostic NVTX ranges removed); p99 1156.2, TPOT 35.67, 20.99 s | exact |

Stage deltas (approx, from means): 59.45 -> 65.59 (+10.3%) -> ~67.4 (+2.7%) -> ~70.6 (+4.8%) -> ~72.3 (+2.4%) -> ~75.75 (+4.8%) -> 76.09 (+0.4%) -> 83.95 (+10.3%, estimated 6% beforehand) -> ~86 (+2.4%) -> ~87.3 (+1.5%) -> 90.7 (+3.9%) -> ~94.2 (+3.9%) -> 94.48 canonical mean -> 95.26 official. Start-to-end: 2.14 -> 95.26 is 44.5x on the same harness; 59.45 -> 95.26 is 1.60x.
Run-to-run spread: +-0.5 req/s in phase A, +-0.5-0.9 in late phase B (effects under about 1% were judged with nsys per-kernel ms/flight or microbenchmarks, not e2e).

## 2. Optimizations and fixes, in the order they were made

Format: evidence -> mechanism -> measured effect -> files. Identifiers are exact; "est." marks numbers that were estimates, not measurements.

### Already in HEAD (not part of the uncommitted diff)
- pending_timeout default 300 s -> 590 s (e2e gate had 99 then 44 HTTP 503 "request_queue_timeout"; rejected requests had also held client slots for 300 s).
- ReclaimSelection::best() unbounded per-candidate traversal (100 ms budget only checked between best() calls) caused an engine wedge; fixed with a preemptive deadline plus iteration budget (5 runs, 0 freezes).

### Phase A (original artifact)

**A1. T4 engine rewrite: remove control-plane serialization (2.14 -> 10.32 -> 17.32/32.95).**
- Evidence: REARCH analysis of 2.14 req/s: capture, fork and binding shared the Session's single context-operation slot and needed quiescent iterations (compute_flights()==0); one hidden parent-helper lane per n-sample group; tiny flights (~55 prefill tokens + 1.7 decode rows each), host 60-80% of wall.
- Mechanism: EngineCore keys on `kInFlightCapture` (Session declares `captures_in_flight()`; only qwen3_5 does; qwen4_exp, glm5next, spark2_5 stay on the legacy path through `if constexpr`). SampleGroup (submission level, sealed under the queue lock) and ForkGroup (worker only); `engage_fork_group` -> Ready/Waiting/Failed; `claim_fork_children` takes distinct free lanes in `Forking` state (`Scheduler::admitted_with` counts a group as one admission); `fork_ready_groups` issues one `session->fork_batch` when the parent rests at `fork_frontier = prompt_tokens-1`; `start_binding(..., beside_flights = !quiescent)` binds synchronously and chains inside `admit_pending` within `kAdmissionBurst` = 50 ms; reclaim no longer waits for quiescence; checkpoints are written inside flights (`kMaxFlightCheckpoints` = 2: system-prompt end and last shared message), published at retirement (`publish_flight_checkpoint` -> `DeferredSettlement{publication_only}` -> `resources_.publish`). Fork eligibility: sample_count>1 && !vision && prompt_tokens>=2 && !masked_draft && !causal_scoring (penalties no longer matter: per-lane device penalty table zeroed in `install_binding`). `fork_prompt`, `sample_helper`, `advance_sample_groups`, `complete_helper` removed.
- Effect: run1 10.32 (no forks), run3 32.95 on the 300-request subset but 17.32 on 2000 (next item).
- Files: src/runtime/engine/{engine_core.h, engine_schedule.inl, engine_execution.inl, engine_requests.inl, request_record.h, scheduler.h}; the staged T3 Session files supply `captures_in_flight`, `fork_batch`, in-flight checkpoint rows.

**A2. Retention control: stalled admissions -> retention cap -> pool-headroom controller (17.32 -> 36.13 -> ~58.5).**
- Evidence: with reuse on the run was host-bound (GPU 27-48% busy). The pool filled with never-adopted Continuation checkpoints (every finished branch publishes one, ~151 MB each, 353 lanes vs 520 sequences). Each admission looped bind(fail) -> reclaim (planner ranks all cached points, 250 us-1.2 ms) -> bind up to `kMaxAdmissionReclaimAttempts` = 8; NeedsMaintenance -> start_maintenance Deferred or prepare_maintenance nullopt (address_borrows_ != 0) -> Stale -> ReclaimProgress::Retry spin; `fresh_candidates` is O(P^2); in-flight checkpoint reservations were skipped (2992 skips vs 465 published); reuse 18% vs ~37% potential. `--no-prefix-reuse` ran at 34.23, GPU 96-100% busy, which isolated the stall to cache management.
- Mechanism v1: branch-0-only continuation publication for n>1 (`publish_prefix_continuation = ... && branch == 0`); a retained-point cap in ResourceManager (`retained_limit_`, `trim_retained` after publish/finish, default `max(16, lanes/2)` = 176, Worth{session, reused, retained_at}). Cap 44/88/176 -> 34.87/35.34/36.13; 240 -> 48.74 vs 48.07 (+1.3%).
- Mechanism v2 (final): static cap removed. Engine `balance_retention()` runs each iteration and calls `ResourceManager::trim_retained(session, keep)` when pool free bytes fall below `capacity / kPoolHeadroomDivisor` (20); pool numbers come from `Session::physical_usage().pools` (ContextPoolUsage). Worth = (session, reused, shared, retained_at), least first, so finished lineages (continuations) go before shared prefixes; points held by waiting admissions are skipped; a leased point stays.
- Evidence for v2: static caps collapse once the pool fills (380: 46.15, 420: 38.82, 480: 33.65, 700: 32.86) because checkpoint creation is skipped and admission reclaim stalls the host. Calibrated simulator matched the engine (cap 176: 12.9% reuse / 38.9% hit vs measured 13.8% / 37.8%); sensitivity ~0.6-0.75% throughput per reuse point; pool fits about 290-330 points. Headroom divisor sweep: 1/8 57.65, 1/10 57.62, 1/12 57.69, 1/16 58.10, 1/20 58.49 (chosen, zero steady-state skips), 1/24 59.18 (persistent skips). Lineage-first vs not at cap 330: 58.99 vs 57.33.
- Files: src/runtime/engine/context_cache/resource_manager.h (retained_points, trim_retained, Worth, remove_shared, held_by_waiting), engine_core.h (balance_retention), tests/runtime/test_resource_manager.cpp (test_trim_retained_releases_least_worth_first).

**A3. MoE grouped GEMM and shared expert (36.13 -> 48.07 together with the sampler; run21 +1.0).**
- Evidence: routed MoE 1463 us at 4096 tokens (~5% of FP4 peak); ncu/debug switches: gate/up 544 us (SM 83%, DRAM 30%), down 859 us (SM 73%, DRAM 12%, L1TEX scoreboard stalls 40%); skipping only the epilogue: gate/up 544 -> 330, down 859 -> 129 us. RED microbench at 4096 tokens: scalar atomicAdd 750 us vs `red.global.add.v4.f32` 188 us (148 blocks).
- Mechanism: down epilogue uses `red.global.add.v4.f32` with a per-warp smem transpose (each lane adds four consecutive output rows of one token) and pipelined `tcgen05.ld` (wait, issue next chunk, process current). Gate/up TileN=64 (two accumulator stages; TMEM 512 columns, scale-factor copies take 96/64, kAccStages=2 only if 2*kAccCols+SF<=512), down TileN=128 above 1536 tokens (64 vs 128 a wash). `all_slice_tokens` 4096 -> 8192 (new ShapePolicy field in topk_softmax_swiglu_moe/dispatch.h). Dense control gate/up: shared-expert SwiGLU gate/up on dense tcgen05 (`launch_control_shared_gate_up_dense`), `shared_mma` keeps router logits and the shared-gate row. Run21 added the single-thread gather and an 8-warp epilogue (kernel became 12 warps: 4-11 epilogue).
- Effect: routed MoE at 4096 tokens 1463 -> ~660 us (down 859 -> ~250 us). Tile sweep (us, uniform routing, 4096 tokens): gate/up 673/461/373/563 and down 455/279/224/247 for TileN 16/32/64/128.
- Files: src/ops/moe/kernels/cuda/block_scaled/nvfp4/b200/grouped_tcgen05.cuh; .../routed_swiglu/cuda/nvfp4/b200/shapes/h3072_i1024_e256_k8/dispatch.cpp; src/ops/moe/topk_softmax_swiglu_moe/{dispatch.h, nvfp4/b200/shapes/.../dispatch.cpp}; src/ops/moe/kernels/cuda/block_scaled/{launch.h, shared.cu}; tests/ops/moe/test_block_scaled_122b.cpp (28 cases).

**A4. Sampler rewrite.** `sample_row_kernel` 4.8 ms per flight -> ~0.28 ms in the microbench (about 0.5-0.7 ms in serving, 1 call per flight); results exactly equal to the old kernel on 512 test rows; column<=16 route stays multi-block partial top-k.

**A5. Attention: planner, K/V loader, per-segment envelope (run17 49.95, run22 55.99).**
- Evidence: the engine issues one `causal_softmax_attention` call per prefill segment per layer (~13-15 B=1 segments per flight plus one decode batch). run15 profile: fmha 12.0% (~13.5 ms/flight) + merge 3.3% (~3.8 ms). Per-CTA fixed cost ~25 us (Q prep), ~5.6 us per key tile. The default plan with `--envelope-max 2042` picked 9 splits + merge at W=64 (52 us vs 38 us single split).
- Mechanism: (1) `select_sm100_attention_plan` (b200/plan.cpp): splits = `target < kMinSplitTarget(3) ? 1 : max(2, min(target, key_tiles/2))`, because a merge costs a launch plus a pass over FP32 partials; the single-vs-split topology depends only on the call shape (never on the envelope) because captured prefill-plan graphs are updated across envelope profiles with `DecodeGraphExecutable::update_child`. An envelope-dependent topology produced `cudaErrorGraphExecUpdateFailure` (fixed; regression case `chunk_limits{160,2048,160,1024,160}`). (2) fmha K/V loader (fmha.cuh, kv_policy.cuh, common/quant/fp8.cuh): batched `load_kv` then `store_kv` with `TileRegs`, `VRaw`, `FusedV`, fp8 `load_k_scale_pair`/`decode_k_scales`/`decode_v8`, so the dependent dequant no longer sits between consecutive V loads; 168-register ceiling for the 320-thread CTA. (3) Mixed (eager) traversal uses a per-segment envelope `{segment.keys, segment.keys}` (`SequenceSegment::keys = main_frontier + columns`); graph-captured packed traversals keep the shared envelope.
- Effect (op bench d256-h32-kv2, fp8, envelope-max 2042, eager, shared noisy GPU; old single split -> new, us): W/ctx 223/0 54.8 -> 40.2; 536/0 89.8 -> 54.8; 1024/0 147.7 -> 94.6; 223/300 81.4 -> 51.7; 536/300 116.4 -> 64.1; 1024/300 197.5 -> 121.6. nsys kernel-only: fmha 27.1/38.4/50.5/99.5 us for the four shapes, merge 17.4 us. Attention per flight 13.5 + 3.8 ms -> ~3.75 ms by the sum_11 profile. Tests: fp8 52 cases, all five KV formats 828 envelope-update cases.
- Files: src/ops/softmax_attention/dense/causal_cache/b200/{plan.cpp, fmha.cuh, kv_policy.cuh}; src/models/qwen3_5/execution/{text.h, text.cpp}, program/{inputs.cpp, graphs.cpp}; tests/ops/softmax_attention/causal/cases.cpp; tests/models/qwen3_5/test_packed_forward_contract.cpp; bench geometry `d256-h32-kv2` added.

**A6. Causal conv varlen rewrite (run17 -> run18, +1.0 req/s).** conv `varlen_output_kernel` was 5.4% of kernel time (run15); after the rewrite about 3.9 ms/flight (~1 us per 32-token walk, ~2 TB/s). `std::numeric_limits<int32_t>::max()` replaced by `INT_MAX` in the kernel (nvcc constexpr-host error).

**A7. Host-path trimming (run21 52.01 partly, run38 59.27, run42 59.28).**
- Evidence: host gap ~9 ms/flight at run15 (6-8 ms, ~6%, by run23). Every `ExecutionCallScope`/`EnginePhaseScope` took a snapshot of resident requests for per-request `RequestHostTiming` (O(lanes) per scope plus O(n*m) decode membership).
- Mechanism: `held_completions_` published by `flush_completions()` at the end of the worker-loop body (after `executor->flush()`); batched decode reserve (`units.size() > 1 && session->reserve_units(units)`); cached shared exposure snapshot (`ExposureHandle`) invalidated by `slots_epoch_` (bumped at 5 `slots_` mutation sites) and O(1) decode membership; replay lanes ordered apart in `reserve_planned_units` and given `unit.tokens = prefill_chunk`.
- Effect: NOEXPOSE upper bound 59.06 vs control 58.49 (accounting cost <= ~1%); cheap accounting 59.27; completion flush after submit 59.28 with mean TTFT 424 -> 372 ms. Batched reserve and publish-after-submit were each estimated at 1.2-1.7 ms/flight; the A/B of batched reserve showed no effect on preemption dynamics.
- Files: src/runtime/engine/{engine_core.h, engine_schedule.inl, engine_metrics.inl, engine_execution.inl}, src/models/qwen3_5/session/request.cpp (reserve_units).

### Phase B (starts at 59.45 req/s; Stop hook reported the target unmet)

**B1. MoE gate/up: remove the gather4 producer bottleneck (59.45 -> 65.59).**
- Evidence: ncu/trace of the grouped kernel: 16 `tma_gather4` instructions per stage cost ~94 cycles each, so the producer spent ~83% of the kernel in issue (~1870 cycles/stage) and starved the MMA warp; the epilogue's IEEE `expf`/division chains were next. Gate/up ~453 us; routed op ~765 us at 3700 tokens, skew 0.4.
- Mechanism: a separate `gather_input` kernel (count/scan/index -> gather_input -> gate/up) pre-gathers token rows so gate/up uses contiguous 2D TMA loads; `silu_approx` in the SwiGLU epilogue (`x * __fdividef(1, 1+__expf(-x))`, worst relative error 3.2e-7, BF16 rounding differs on 0.0004% of inputs; later `rcp_approx_ftz`/`exp_approx_ftz` form is bit-identical to it over all 2^32 float inputs and saves 3 instructions); smem-offset change; NVFP4 activation quantizer divides by a shared reciprocal (Markstein: `q = x*r; fma(fma(-q,s,x),r,q)`, `r = rcp.rn(s)`), bit-identical to `__fdiv_rn` for E4M3 scales (checked on 5.37e9 samples; the division itself moved the op only ~1 us, so it is a cleanup rather than the main gain).
- Effect: gate/up 453 -> 212 us; routed op 765 -> 487 -> 481.5 us; est. -13.4 ms of ~106 ms per flight; e2e 65.59 (single run). `ninfer_moe_block_scaled_122b_test` 28 cases OK.
- Files: src/ops/moe/kernels/cuda/block_scaled/nvfp4/b200/grouped_tcgen05.cuh and the gather/quantize kernels; src/ops/common/quant/nvfp4.cuh; src/ops/common/numeric/math.cuh (silu_approx).

**B2. MoE combine: ordered, vectorized, deterministic (-> ~67.4).**
- Evidence: HEAD combine was `.atomic_output = tokens >= 128` fp32 `red.global.add(.v4)`: L2-atomic bound (~2.4-3 TB/s) and run-to-run nondeterministic (section 5).
- Mechanism: the down GEMM writes `contributions[slot][hidden]` in fp32 and a gather kernel sums slots in fixed order (`reduce` for the routed op, `stage_routed` for the complete op with shared experts); the 122B shape is always ordered from this point. Cost: contributions buffer [A][3072] fp32 = ~805 MB at an 8192-token slice (slightly smaller context pool).
- Effect: routed op 480.5 -> 437.7 us at 3700 tokens (faster and deterministic); complete op 650.7 us (gate/up 191, down 127, stage_routed 65.6 at the HBM roofline for the fp32 contributions, shared GEMMs 73 x2, shared_mma 50.6, route_selection 18, scale_shared 13.5, gather 13.5 us). Runs 1-4: 67.19-67.60.

**B3. Dense GEMM: BF16 TileN=256 and vectorized Add epilogue (-> ~70.6).**
- Evidence: dense sm100 GEMMs 17-22% of kernel time, BF16 at ~65-69% of BF16 peak (1.55 PFLOP/s isolated), only TileN=128 in use (8964 launches, 149.8 us average = 82% of dense time).
- Mechanism: `sm100_gemm_kernel` (persistent, cta_group::1, swap-AB: weights are MMA-M, tokens are N; TMA producer, MMA, TMEM-allocator and 4 epilogue warps) gets a TileN=256 path (accumulator double-buffered as 2x256 TMEM columns) for rows >= 1024; the `PlainEpilogue` of the Add (residual) GEMMs (residual aliases output, staged through per-warp smem; measured case: shared-expert down 3072x1024) is vectorized.
- Effect: isolated BF16 in_proj at 4600 tokens 430 -> 373 us; Add-epilogue GEMM 3072x1024 129 -> ~30 us; runs 5-6 70.79/70.40. A 2-CTA (`cta_group::2`) BF16 kernel was judged not worth it (best-in-class only ~16% above the 256 tile). Test fix: the first 256-tile cases did not reach the new path (threshold rows>=2048 vs test rows 1536), threshold lowered to 1024.
- Files: src/ops/common/cuda/sm100/gemm_mainloop.cuh, src/ops/linear/kernels/sm100/{bf16_gemm.cu, gemm.h, gemm_launch.cuh}, src/ops/linear/add/*.

**B4. Small-kernel and launch-bound tuning (-> ~72.3, build #5).**
- Causal-conv varlen restructure: `__launch_bounds__` minBlocks=5, compile-time strides, ftz silu; T=3442: 89.8 -> 73.1 -> 48.7 us, ncu instructions 79M -> 20M (the first restructure left occupancy at 25%/106 regs and gave nothing until the register cap).
- GDN `prepare_kernel` launch bounds (varlen op T=3442: 435.7 us before; minBlocks 5 -> 418.6, 6 -> 407.8, 7 and 8 -> 394.9 us; 8 chosen), rmsnorm approximations, router column handling; complete MoE op at 3700 tokens 648.6 -> 577.0 us. Tests: gated_delta_net (chunked, varlen, replay_record, replay_fold, control) 5/5, causal_conv1d_varlen 8, causal_conv1d_silu 205, normalization 569.
- Runs 7-9: 72.54/72.46/71.94.

**B5. Grouped (ragged) causal attention and grouped FP8 KV write (-> 75.75 -> 76.09, builds #6, #7).**
- Evidence: prof3: fmha 9.57% plus merge, ~15 per-segment B=1 launches per layer (each ~26-31 us, low occupancy, wave quantization).
- Mechanism: one 1-D-grid launch `fmha_group_kernel<Kv>` covers all groups of the mixed (eager) traversal (a group = `batch` consecutive entries of `width` columns on the shared token axis). A group table (<= 72 groups, `cta_base` ascending) rides in `__grid_constant__` params; each CTA maps to (group by binary search, entry, tile with later tiles first, kv head). Every key range is one split and writes BF16 directly: no partials, no merge, no workspace. Numerics differ from split+merge only in FP32 summation order, and groups stay isolated. Packed prefill (possibly graph-captured) keeps per-segment calls; non-B200/non-tcgen05 targets fall back to per-group `causal_softmax_attention`. A grouped FP8 KV write kernel (one warp per (column, kv head), binary search in the group table, `valid_columns` clamp; fix for `local < 0` in `write_fp8_groups_kernel`) removes the per-segment writes (only FP8 storage with 2 KV heads; other dtypes write group by group).
- Effect: fmha_group 5.65% of kernel time (58 us for the segments_and_decode case); build #6 75.75 mean, build #7 76.09 mean (TPOT 42.1-42.5 -> 41.5-41.9 ms). Tests: `causal_groups/*` OK for bf16, int8, nvfp4, k8v4 and fp8.
- Files: src/ops/softmax_attention/... (fmha group path, group table), src/ops/kv_cache/kernels/write.{cu,h}, write/fp8.cuh; include/ninfer/ops/softmax_attention.h; bench/ops/causal_softmax_attention_bench.cu; src/models/qwen3_5/execution/text.cpp.

**B6. FP8 W8A8 tcgen05 projections for the GDN layers (76.09 -> 83.95, +10.3%; estimated 6-8% beforehand).**
- Evidence: dense sm100 GEMMs were 17-22% of kernel time; GDN in_proj (n=20480 = {12288, 8192}, k=3072) and out_proj (n=3072, k=8192) were BF16 tcgen05 GEMMs at ~65-69% of BF16 peak; the published checkpoint keeps GDN in BF16 and NVFP4 for GDN was rejected for accuracy (PPL gate +-1%; converter has no BF16->NVFP4 RTN, only `stored_scale_mse`). Whether an FP8 stage would be MMA-bound or stage-feed bound was unknown until a prototype was timed.
- Mechanism: weights are E4M3 codes [n,k] plus FP32 per-row scales (`FP8_E4M3FN_ROW_FP32`, RowScale layout: codes plane, then scales at a 256-byte-aligned offset), produced by converter method `fp8_row_fp32_maxabs` with `activation_policy="AllowA8"`. Activations are quantized per token by `launch_fp8_a8_quantize` (scale = amax/448, zero token -> scale 0 and codes 0). `GemmKind::Fp8` in gemm_mainloop.cuh (a stage is one 128-byte K slab = 128 E4M3 values); the epilogue multiplies the accumulator by `row_scale x token_scale` before the optional residual add and BF16 rounding. `linear_backend()` returns Tcgen05 for `(BF16 || FP8_E4M3FN_ROW_FP32)` at (20480,3072) and (3072,8192); policy `A16Only` is rejected with `invalid_argument`, AllowA8/AllowA4 accepted. Wired through projection, split (qkvz via `linear_split`), add (out via `linear_add` with residual) and the GDN input-conv op (`gdn_input_proj_conv_update/record`, decode materialized route). GDN a/b stay BF16; FP8 shared experts were rejected (~0.4% net).
- Bytes (from the two conversion.json files, 36 GDN layers): qkvz 125,829,120 -> 62,996,480 B and out 50,331,648 -> 25,178,112 B per layer, i.e. -3,167,502,336 B in total.
- Effect: runs 16-18 83.66/84.24/83.94; means over the three runs of builds #7 -> #8: TTFT 313 -> 241 ms, E2EL 1689 -> 1531 ms, TPOT 41.7 -> 40.0 ms; FP8 qkvz GEMM ~165 us per call; `fp8_a8_quantize_kernel` costs 0.66% (K=8192, 14.4 us avg) and 0.37% (K=3072, 4.3 us avg). v2 artifact PPL 3.869522 (+0.0585%).
- Files/tests: src/ops/linear/backend.h; src/ops/linear/kernels/sm100/fp8_gemm.cu (new); kernels/fp8/{fp8_a8.cu, fp8_geometry.h}; add/fp8/b200, split/fp8/b200, projection dispatch; src/ops/common/quant/fp8.cuh; tools/convert/official_recipes.py (`qwen3_5_122b_nvfp4`); tests/convert/test_qwen3_5_122b_nvfp4.py (`test_gdn_projections_are_row_scaled_fp8_with_a8_permission`); tests/ops/linear/{fp8_row_fp32_t4, fp8_b200, split, add, projection} and the B200-only profile `{FP8_E4M3FN_ROW_FP32, 3072, "fp8_tcgen05"}` in the GDN input-conv test. Link detail: the FP8 shape files had to be added to `ninfer_ops` explicitly because `libninfer_ops_non_rdc.a` members are pulled in only when referenced.

**B7. BF16 output head (v2 artifact, 83.95 -> ~86; estimated +2.5%).** Evidence: lm_head `q8_a16_mma_kernel` 2.6-2.7 ms/flight (3.0-3.2% of kernel; legacy mma.sync ~225 TFLOP/s, ~0.87 ms per 128-token tile, DRAM floor ~0.1 ms). Mechanism: recipe `_assign(recipe, "text/output_head", "bf16")` (was Q8 g32 fp16) plus BF16 n=248320,k=3072 tcgen05 route in `linear_backend()` (projection/bf16/b200/shapes/n248320_k3072.cu). Cost: +715,161,600 B. Measured single runs 85.69/86.13 (profiled runs 84.8-85.4).

**B8. In-flight checkpoint capture refinement, "Approach C" (~86 -> 87.3, +1.7% where +3-4% was estimated).**
- Evidence: after the T4 engine each request still ran two prefill units: the window was cut at the last checkpoint, leaving a ~11-token trailing window, and the end-of-window checkpoint was a 144 MiB D2D State copy in `RankProgramImpl::publish_checkpoints` (pre-head gap 0.5-1.0 ms); about 8 extra units per boundary cost ~1-1.5 ms host. Boundary: 5.85 ms idle plus in-flight idle 4.3 ms per ~95 ms flight.
- Mechanism: final prefill windows carry their checkpoints (permit-shrink block removed from `reserve_units` in session/request.cpp); interior columns are written in-kernel via `window_splits`/`execution_splits` of the varlen recurrence plus a small continuation-hidden copy; the Session hands checkpoints to the engine with `Session::take_flight_checkpoints(SequenceHandle)` at the flight's `ExecutionComplete` event (returns `FlightCheckpoints{published, skipped}`; replaces the row-carried `PrefillRowProgress.published_checkpoints/skipped_checkpoints`) and the engine registers them via `publish_flight_checkpoint` -> `DeferredSettlement{publication_only}`. Companion fix: `reserve_for_growth` (geometric growth) in src/models/common/qwen/prefix_identity.{h,cpp} for `ResidentPrefixIdentity::reserve` and `PrefixShortlistDigests::reserve`, and `prepare_identity` in session/commit.cpp, because exact-size `reserve` on every append defeated geometric growth (allocator was 17.7% of boundary samples, ~0.74 ms per boundary).
- Effect: units.prefill = 1 for 99.8% of requests; token reuse 23.4% -> 22.4%; runs 87.28/87.11/87.11; TTFT ~250 -> 215-221 ms.
- Files/tests: src/models/qwen3_5/session/{request.cpp, session.h, session.cpp, internal.h, commit.cpp}, src/runtime/engine/engine_execution.inl; tests/models/qwen3_5/test_native_transactions.cpp (`flight_checkpoints`, new `final_window_checkpoints`, `marker_request(one_window)`).
- A follow-up `reclaim_order` fix (removed an O(owners^2) `points(owner.token)`/`find_owner` scan and a duplicate `checkpoint_metadata` call per victim; 9.6% of boundary samples, ~0.65 ms) changed nothing measurable (87.21/87.27/86.97).

**B9. Register-state chunked GDN step (87.3 -> 90.7, +3.9%).**
- Evidence: varlen GDN recurrence ~11.7 ms/flight average (~17.5 ms on prefill-heavy flights) plus prepare ~2.8 ms was the largest remaining kernel. ncu on the old `recurrence_kernel<64>` (grid (128,17), 313.5 us): L1/TEX 93%, compute 39%, 2.4-way smem bank conflicts, theoretical occupancy 25%, 126 regs, 61,952 B dynamic smem, 2 CTAs/SM; stall cycles per issued instruction (9.16 total): mio_throttle 2.30, short_scoreboard 1.92, wait 1.21, barrier 0.63. About 3.54 us per 16-token chunk step. The old step snapshotted the FP32 state (128x64) to smem each chunk and fed `mma.sync` B fragments with scalar `LDS.32`, so it was shared-memory/MIO bound, not tensor bound. A smem swizzle fixed bank conflicts only (-1.6% on the op, no in-flight gain, +31% instructions) and was reverted.
- Mechanism (src/ops/linear_attention/gated_delta_net/recurrence/chunked/register_state.{cuh,cu}; `chunked/recurrence.cu` `case 64` calls `launch_register_state` unless `NINFER_TARGET_SPARK`; the varlen op reuses the same step through `chunked::registers`): a warp owns 16 value rows of S^T[value,key] in MMA accumulators across chunks; TF32 A fragments come straight from the accumulators in order (c0,c2,c1,c3); K.S is TF32, Q.S uses the BF16-rounded state, V^T arrives through `ldmatrix_x4_t`, 3 cp.async stages, `__launch_bounds__(128,3)`; ~656 instructions per warp-chunk (72 HMMA), tensor pipe 41-46% busy, issue slots ~47%. Arithmetic per chunk (unchanged contract): R = V - prefix.pred; delta = solve.R (TF32); out = scale.(prefix.hist + mqk.delta) (TF32); S <- decay.S + K^T diag(suffix) delta with BF16 delta; FP32 master state kept. ControlChunk{solve[256], mqk[256], prefix[16], suffix[16], padding[32]} (2304 B), QkChunk{q[16*128], k[16*128]} bf16 with swizzled `vector_index`/`square_index`.
- Effect: no-boundary microbench (R=8, T=4270): varlen call 474 -> 288 us, recurrence kernel 353 -> 178 us; production nsys: varlen recurrence 11.7 -> 7.44 ms/flight; harness reg1 90.71 (TTFT 211.7, E2EL 1417).
- Contract kept: varlen output, row destinations and checkpoints bit-identical to calling the single-segment chunked op per sub-segment, and within FP64-reference tolerances (output rel_l2 6e-3, state 4.5e-3), so chunked and varlen share one arithmetic. A reduced-precision (fp16/bf16) State was rejected repeatedly because FP32 master State is a public contract and touches ~10 ops (estimated 7-8% from decode traffic; unexplored).

**B10. Recurrent-route rewrite and grouped prepare (90.7 -> ~94.2, +3.9%).**
- Evidence: the 90.7 build gained less in production than the no-boundary microbench promised. Real flights call the varlen op with R ~9.5 rows, T ~4000-4200 and C ~40 boundaries (4.3 per row; 35% are checkpoints); sub-segment length mean 79, median 29, p10 4, p90 240, 23% shorter than 16 tokens; longest row ~862 tokens. Short sub-segments take the recurrent route (length < `chunked_threshold`, <= 16) and every boundary adds prepare work. A replay microbench driven by recorded flight traces (trace patch needed a `cudaStreamIsCapturing` guard around its D2H copy) reproduced 374 us per call where the boundary-free bench said 288.
- Mechanism: (1) `run_recurrent_rows` (varlen/recurrence.cu; arithmetic fixed by the contract: 4 keys per lane, groups of 4 value rows, `warp_sum`): a warp holds `state[4][4][4]` (16 value rows) and shares key, gate and query loads across the 4 groups; `register_recurrence_kernel` at `__launch_bounds__(128,4)`. The first version was slower (381 vs 374 us, STACK 192) because `regs::State` stayed live across the recurrent branch; scoping it inside each run of chunked sub-segments gave STACK 0 and 336 us. (2) Grouped prepare: `kMaxGroup = 4`, one block per (chunk, QK head) builds the control chunks of the group's value heads from one QK chunk; `kPrepareMinBlocks = 5` (8 spilled: REG 32, STACK 120), 48 registers; per-element arithmetic identical to `chunked::prepare_kernel`.
- Effect: replay bench 374 -> 336 us (recurrent route) -> 273 us (grouped prepare); harness prep1-3 94.36/93.77/94.49; canonical artifact 94.65/94.79/94.01.
- Tests: tests/ops/linear_attention/gated_delta_net/test_varlen.cpp (`random_spec`, `many-rows`, `hv64-many-rows` cases) and test_recurrence.cpp (122B cases); all GDN suites pass against FP64 references (chunked 342 cases, varlen 7, replay_record 279, replay_fold 155, replay records, control).

**B11. Finalization.** Canonical artifact regenerated from the final recipe (id d1c45569..., 72,305,503,232 B; v2 and fp8gdn deleted; original kept in orig/ as the PPL reference). Official ninfer_e2e.sh PASS: 95.2644 req/s, 2000/2000, PPL 3.867882 (+0.0161%), sanity 20/20. 12 diagnostic NVTX ranges removed from the Engine before that run (the three earlier runs on the same artifact predate the removal).

## 3. Profile breakdowns recorded at each stage

All from nsys (kernel table, `--cuda-graph-trace=node`) on GPU 1 under the locked harness unless noted; profiling slows the run slightly (e.g. 83.85 profiled vs 83.95 plain). "ms/fl" = ms of kernel time per mixed flight.

| stage / build | window | flight shape and period | GPU busy | kernel breakdown |
|---|---|---|---|---|
| pipeline HEAD (2.14 req/s) | | ~55 prefill tokens + 1.7 decode rows per flight; prefill ~6.3k tok/s at 2048-token chunks, ~167 ms for one 513-token flight | host 60-80% of wall | control-plane bound (section 2 A1) |
| run7, reuse off (34.2) | | ~255 ms, ~4.7k prefill tokens + ~310 decode rows, ~18.3k prefill tok/s | 96-100% | grouped_gemm 38%, sm100 GEMM 15%, shared_mma 9%, fmha 7%, GDN ~17% |
| run9 (~36) | 8.06 s | ~230 ms; ~3.8-4.4k prefill tokens + ~280 decode rows | 80.5% | grouped_gemm 37.5% (443.6 us avg, 5496 calls), dense 14.1%, shared_mma 9.1% (215 us), fmha 7.2%, GDN decode 6.6%, GDN prefill ~15%, sample_row 2.2% (4.8 ms/fl), lm_head 1.2% (2.7 ms) |
| run15 (~48-50) | 55 flights, kernel sum 6.191 s | ~160 ms; ~3700 prefill tokens (~8 requests, ~14 prefill rows) + ~287 decode rows; kernel 112.6 ms/fl; host gap ~9 ms/fl | ~95% | MoE grouped 24.3% (~27 ms/fl), dense 19.4% (~22), fmha 12.0% (~13.5), merge 3.3% (~3.8), GDN decode 10.6% (~12), GDN recurrence 8.9% (~10), prepare 3.1%, lm_head 2.1% (2.46 ms/launch), shared_mma 2.0%, conv varlen_output 5.4% |
| prof23 (~59) | | ~139.6 ms; ~285 decode rows + 3.4-3.9k prefill tokens in 13-16 segments; prefill 72-85% of GPU time at ~30 us/token | ~91% | MoE 34.1 ms/fl (gate/up 310 us, down 145 us per layer), dense 28.1, GDN recurrence 13.7 (234 us), GDN decode 13.5 (392 us, ~6.4 TB/s), fmha 9.8 + merge 0.8, prepare 4.6, conv 3.9, lm_head 2.6; per request ~17 ms GPU time (target 10.9 ms) |
| prof3, build #5 (~72) | 54 flights, 5404 ms | ~125 ms wall; ~1300 kernels/flight, 1-2.5 us gaps | 96-97% | GDN decode 15.4% (340 us, HBM roofline), recurrence<64> 12.2%, prepare 2.75%, dense 21.9% (BF16 TileN256 9.73, TileN128 8.04, SwiGLU 2.44, NVFP4 1.67), fmha 9.57%, MoE grouped 16.5% (gate_up ~71% HBM peak), stage_routed 3.0%, lm_head 2.7%, shared_mma 2.1%, route_selection 0.9% |
| profile #4, build #8 FP8 GDN (~84) | kernel 9390 ms of a 10447 ms span | flight shape not recorded in the summaries (parent notes for the same stage: ~119 mixed flights per 15 s window, ~83 ms GPU each, ~6.2 ms GPU idle per boundary) | idle 1057 ms = 10.1% | GDN decode 16.73%, GDN recurrence 14.88%, MoE grouped 18.79% (10.53 + 7.27), dense 17.09%, fmha_group 5.65%, stage_routed 3.58%, prepare 3.33%, lm_head 3.03-3.18% (2.6-2.7 ms), shared_mma 2.5-2.7%. Gaps: <5 us ~250 ms; 100 us-1 ms 106 ms; >1 ms 672 ms (6.4%) = 116 boundary gaps (613 ms, ~5.3 ms each: increment_prefix_counts_kernel -> embed_gather_q8_kernel) + 52 gaps of ~1.1 ms (sample_row -> scatter_columns) |
| v2 steady state (~85-86) | 41 flights | period 94.7 ms: kernel 84.6 + in-flight idle 4.30 (micro-gaps ~1.9, pre-head memcpy 0.5-0.65, sample->commit round trip 1.17) + boundary idle 5.85 | ~95% | MoE grouped 16.10 ms/fl (19.0%, 96 calls), dense 15.32 (18.1%, 229 calls), GDN decode 14.26 (16.9%, 36 calls, ~6.1 TB/s = 73% HBM), varlen recurrence 13.30 (15.7%, 36), fmha_group 5.07 (6.0%, 12), stage_routed 3.20, prepare 2.98, shared_mma 2.24, varlen_output 1.48, sample_row 0.71; prefill ~66% of GPU time |
| Approach C stage (~87.3) | | period 84-90 ms = kernel ~80.2 + boundary 6.7-6.8; ~8.1 requests/flight; ~3.2k computed prefill tokens/flight (mean prompt 536, computed 419, reuse 21.7%) | 90-91%; in-flight gaps 2.3-2.6% | GDN decode ~10.5 ms/fl (79% HBM), varlen recurrence ~11.7 avg (~17.5 on prefill flights) + prepare 2.8, MoE grouped ~14, dense ~14 (~58% of FP8 peak) |
| register-state (~90.7) | | | | varlen recurrence 11.7 -> 7.44 ms/fl |
| final (~95) | | ~80 ms kernel + ~6.7 ms host boundary (parent summary) | | recurrence replay-bench 374 -> 273 us per production-shaped call |

Host boundary (v2 stage, NVTX, sample_row end -> first kernel of the next flight, mean 7.08 ms; engine thread running 98% of boundary gaps; CUDA API calls only ~0.65 ms of ~6.0 ms): admission 1.094, reserve 1.036, commit_output 1.016 (17 ranges), settle 0.474 (78 calls x ~6 us), fork 0.341, plan 0.029, unattributed ~3.1 ms (Session::commit 0.4-0.55, poll/retire ~0.3, start_mixed -> first kernel ~1.5 incl. ~116 pre-kernel API calls 0.77 ms). Engine-thread CPU samples in boundary windows: begin_iteration 38% (admit_request 14.6%, fork_ready_groups 6.2%, Session::finish 5.4%, apply_settlement 4.1%), submit_iteration 26% (reserve_planned_units 15%, start_mixed 8.7%), poll_execution 25% (commit_pending 15.3%), allocator 17.7%.
Per-item boundary shares at the 87 stage: reclaim_order 8%, ContextPool::plan 5%, fork 6%, reserve/assess 15%, malloc churn 16% of ~6.7 ms.
Closed-loop constants: ~17 flights per request (1 parent prefill + ~15 decode + 0.5-1 arrival wait); 91.5 req/s needs E2EL ~1.42 s, ~10.9 ms GPU time per request (about 20 us per prefill token); GDN decode state traffic ~300 MB per row-step is a floor of ~1.4-1.5 ms per request.

## 4. Experiments tried and rejected, or measured as irrelevant

| idea | number that decided it |
|---|---|
| GDN value tile 32 / 16 (varlen recurrence) | 50.59 / 29.61 req/s vs 50.98 (run19/20); tile 64 kept |
| MoE gate TileN=128 | 380-433k cycles vs ~350k for TileN=64 (gate/up 563 vs 373 us at 4096 tokens); plan.cpp reverted |
| MoE down tile 64 vs 128 | 48.24 vs 48.07 req/s: wash, default kept |
| MoE scale-loader prefetch; L2 policy switch; L2 prefetch | no gain / no effect / slower gate/up (513 vs 374 us was a foreign GPU load artifact); all reverted |
| quantizer-epilogue division as the gate/up bottleneck | op moved ~1 us; real cause was gather4 issue (B1) |
| retained-point caps 290-700 and cap/lineage variants | 57.78 (290), 58.99 (330), 46.15 (380), 38.82 (420), 33.65 (480), 32.86 (700); replaced by headroom controller |
| host cache tier (host_capacity > 0) | 83.35 vs 85.7-86.1; cost model + demotion stalls; default stays 0 |
| lane count | 520 lanes: 75.66 req/s at the v2 stage (TTFT 84.7 ms, E2EL 1701 ms, host 92%); 256/288/320 lanes: 58.55/59.17/59.15 vs 58.49-59.27 at the 59 stage; 300 lanes: 85.57 vs 85.7-86.1 at v2: flat, 361-363 lanes is near the sweet spot |
| host exposure accounting off (NOEXPOSE) | 59.06 vs 58.49: bounds the cost at ~1%, cheap accounting got 59.27 |
| prefix-cache capacity or policy changes | simulation: 22.2-22.9% reuse at 256 entries, 28-29.7% at 512, 30-32% unlimited; pool-capacity-bound (~290-301 shared checkpoints of 144 MiB); in-flight dedup of siblings worth ~2.5% of chars |
| prefill/decode GDN kernels on two streams | no gain (both saturate CTA slots) |
| GDN chunk-size change | would change chunked and varlen together under the bit-identity contract; deprioritized |
| fp16/bf16 GDN State, bf16 MoE contributions, bf16 atomics | contract changes (FP32 master State is public; ~10 ops touched); GDN decode already at 79% of HBM peak; est. 7-8% unexplored |
| atomic MoE combine | slower (480.5 vs 437.7 us) and nondeterministic |
| NVFP4 for GDN projections | accuracy risk against the +-1% PPL gate; converter has no BF16->NVFP4 RTN |
| FP8 shared experts | ~0.4% net |
| 2-CTA (`cta_group::2`) BF16 GEMM | best-in-class only ~16% above the TileN=256 result |
| swizzle in varlen recurrence | op 394.8 -> 388.5 us (-1.6%), +31% instructions, no in-flight gain; also touches a kernel shared with dgx_spark; reverted |
| reclaim_order O(owners^2) fix | 87.21/87.27/86.97 vs 87.1-87.3: no measurable change |
| async/pipelined launch-ahead scheduling | would hide ~7 ms/flight (~8%) but needs device-side token feedback, ghost-row and lane-quarantine redesign; not pursued |
| multi-step decode, half-flight A/B alternation, PDL, swapping begin/submit_iteration | ~neutral to +1%, or alternating decode-only/prefill-heavy flights |
| multi-stream fmha; BF16 lm_head materialized at load time (considered at the 59 stage) | stream pool + per-segment workspace needed; the BF16 head changes numerics and weight materialization (it was later done through the recipe, B7) |
| ncu under serve | captured only warm-up kernels or died with LaunchFailed; standalone benches were used instead |

## 5. Correctness bugs and fixes

Product code (fixed):
1. **submit_samples race** (T4 work): `sample_group_id`/`members` were set after the request became visible in `pending_`; now linked inside `create_request` under the lock with explicit group sealing.
2. **free_lane() returned the same lane repeatedly** in the old engage code; the new claim code scans distinct free lanes.
3. **logic_error "fresh admission exceeded the FIFO bypass allowance"** (run5): `claim_fork_children` called `scheduler_.admitted()` once per child; `Scheduler::admitted_with` makes a group one admission.
4. **No forks happened** (run1): eligibility required no penalties and no rewrite checkpoints; replaced by the sample_count rule (A1).
5. **CUDA graph update failure** `cudaErrorGraphExecUpdateFailure` (update result 2) at `causal_append/.../t1/base17/bound2048/envelope_updates`: the attention planner made single vs split+merge depend on the envelope's key tiles; topology is now shape-only; regression case added; 828 cases pass.
6. **Replay lane without a unit permit** (latent bug from the pipeline's T06 stage, verified by preemption_real): `reserve_planned_units` skipped the replay lane because it is not in the prefill list; it is now ordered apart and given `unit.tokens = prefill_chunk` without spending the prefill budget. preemption_real passes all 4 scenarios (e.g. replay: 5 preemptions, 5 restores, 16296 replayed tokens).
7. **Zero-suffix reuse regression** (grammar_real and tools_real snapshot hung): a retained continuation at frontier == prompt_tokens leaves an empty window that packed/mixed paths reject. `build_lane_table` now gives a non-forking lane `prefill_limit = max(min(chunk, suffix-computed), 1)`, and the Session runs the empty window alone (`operation.packed = Prefill && rows.size() > 1`, `start_prefill` handles `end == begin`), sampling the first token from the source's `continuation_hidden`. A mixed-flight crash "packed prefill window cannot share an execution" was fixed by the same `alone` change; new zero-suffix real test (`ninfer_qwen3_5_zero_suffix_real_test`).
8. **Scoring/PPL nondeterminism (root cause, fixed by B2):** six repeated `score_tokens` runs on 1537 tokens differed at ~1530 positions (max_abs_diff up to 7.76 nats; "whole" mean log-prob -0.173364 vs -0.177435 in two processes). Cause: HEAD's `.atomic_output = tokens >= 128` fp32 `red.global.add` combine in the grouped GEMM down epilogue (b200 h3072_i1024_e256_k8 dispatch); with `atomic_output=false` repeated scores are bit-identical. The NVFP4 model amplifies rounding noise (rare multi-nat spikes at ~0.08% of positions) and the batch-wide NVFP4 activation scale couples rows. Ordered combine is now the only mode for the 122B shape.
9. **Grouped FP8 KV write chunk bug:** a later chunk's binary search could land on a group whose `token_base` exceeded the column (negative `local`); `local < 0` check added in `write_fp8_groups_kernel`.
10. **Growth/allocator churn:** exact-size `reserve` calls in prefix identity defeated geometric growth (B8); `reserve_for_growth` added.
11. **Stale tests corrected:** `score_real` assertion `!=` -> `<` (context_device_bytes vs pool-managed `context_capacity_bytes`), `tools cancel` (parser retains an Incomplete call on Cancelled), `preemption_real` cancel-replay assertion made per-request, `kRecutBudget` left at 0.12 although the packed-prefill re-cut noise floor is rel_l2 ~0.127.
12. Other build/test-side fixes: `__int_as_float` in host-pass initializers; `INT_MAX` in a conv kernel; missing `typename` in `resource_manager.h:1367` (dependent `ReclaimCursor::OwnerProtection`); FP8 shape files added to `ninfer_ops`; 256-tile test threshold; b200 fp8 attention bench geometry `d256-h32-kv2`.

Pre-existing, not caused by this work and not fixed (all 22 ctest failures were triaged this way): `ninfer_platform_targets_test` and `ninfer_apps_help_test` (CMake references scripts deleted in commit 934ce1a4); `ninfer_spark2_5_hf_fixture_build` and dependents; NVFP4 `mma.sync`/KDA/QSA/non-122B-MoE cases throwing "... is not available on b200"; `ninfer_glm5next_session_test` worker spin (reproduces with HEAD's engine sources); native-transactions stages `rewrite_segments_share_prefill_unit` (fixture geometry) and `capped_prefill_matches_uncapped` (K 0.136/V 0.130 vs 0.12 budget on the canonical artifact, 0.130/0.124 on the original; unconfirmed on HEAD); `ninfer_kimi_delta_attention_bench` link error (undefined `kda_input::mxfp8_execution_workspace`; builds need `ninja -k 0`); `ninfer_context_store_test` flaky (~1 in 9, test race); locked `ppl_check.sh` parses "35" from the artifact path so its pass is not numerical evidence (documented in docs/b200-delivery.md).

## 6. Caveats on this reconstruction

- Rows marked approx come from my summaries, not from a harness JSON I re-read; rows marked exact were re-extracted from harness output lines printed in the transcript (or from bench_results/benchmark_*.json). Run1, run3, run7 and the 44/88 cap values are approx.
- Several steps bundle more than one change (run13: MoE epilogue + sampler + slice cap + dense control; run21: gather + 8-warp epilogue + completion batching + batched reserve; run22: fmha loader + planner fix + per-segment envelope + cap 240; #5 builds). No isolated e2e A/B exists for each; op-level numbers come from benches/nsys on a shared GPU (several early benches were noisy because another tenant used GPU 1).
- "est." numbers (e.g. -13.4 ms/flight for B1, 6-8% for FP8 GDN, +3-4% for Approach C) were estimates; only the e2e deltas in section 1 are measured. FP8 GDN beat its estimate (+10.3%); Approach C fell short (+1.7%).
- The v2 artifact PPL (3.869522) and the canonical artifact PPL (3.867882) differ although the recipe is the same; the summaries do not explain the difference.
- Which staged/unstaged lines belong to the pipeline's T3 versus my edits cannot be separated from the summaries; the T3 Session contract (`captures_in_flight`, `fork_batch`, in-flight checkpoint rows) was squash-merged by me from dev/task-0002, and the engine side (T4) is mine.
- No re-measurement of HEAD is in the summaries beyond the pipeline's own benchmark_20261117_000202 (2.14 req/s).
