#pragma once

#include "core/arena.h"
#include "core/device.h"
#include "core/tensor.h"
#include "core/record_view.h"


#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the transient arena capacity required by gated_delta_net for the given geometry. It is
 * zero when the private implementation requires no transient storage. The state/head dimension is
 * fixed at 128; `value_heads` must be at least `qk_heads` and divisible by it. The query covers
 * every T in the inclusive interval and throws for an invalid profile or interval.
 */
[[nodiscard]] std::size_t gated_delta_net_workspace_capacity_bytes(std::int32_t qk_heads,
                                                                   std::int32_t value_heads,
                                                                   std::int32_t min_tokens,
                                                                   std::int32_t max_tokens);

/**
 * Applies the Gated DeltaNet recurrence independently for each value head h. Let
 * G=value_heads/qk_heads; its Q/K head is qh=floor(h/G). Starting from S_h, for t in increasing
 * order:
 *
 *   alpha       = exp(g[h,t])
 *   delta       = beta[h,t] * (v[:,h,t] - alpha * S_h * k[:,qh,t])
 *   S_h         = alpha * S_h + outer(delta, k[:,qh,t])
 *   ideal[:,h,t] = scale * S_h * q[:,qh,t].
 *
 * Shapes/dtypes are contiguous q/k BF16 [128,Hqk,T], v/out BF16 [128,Hv,T], g/beta FP32 [Hv,T],
 * and state FP32 or BF16 [128,128,Hv], where Hqk>=1, Hv>=Hqk, and Hv%Hqk==0. `scale` is
 * 1/sqrt(128). When `normalize_qk` is true, the Op consumes raw q/k and applies
 * x / sqrt(sum(x^2) + 1e-6) independently to every 128-element row before using it. When false,
 * q/k are consumed as supplied. The outputs approximate the complete recurrence above from
 * represented inputs. The call publishes only the state after token T-1, rounded to nearest even
 * when the state is BF16; output storage is BF16. Scratch capacity is independent of
 * normalize_qk.
 * `execution` supplies the stream and physical SM count for launch decomposition.
 * Inputs and out do not overlap state or one another. `ws` supplies transient storage reported by
 * gated_delta_net_workspace_capacity_bytes; scratch is scoped to the call. T may be any positive
 * value.
 *
 * This overload reads and writes the same `ssm_state`, publishing the state after all T tokens.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     Tensor& ssm_state, Tensor& out, CudaExecutionView execution);

/**
 * Distinct-state form of the same recurrence. `ssm_state_out` receives the final state;
 * `ssm_state_in` and `ssm_state_out` share one dtype and may be disjoint or exactly the same
 * storage. No other arguments may overlap either state.
 */
void gated_delta_net(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                     const Tensor& beta, float scale, bool normalize_qk, MemoryArena& ws,
                     const Tensor& ssm_state_in, Tensor& ssm_state_out, Tensor& out,
                     CudaExecutionView execution);

/**
 * One-token update for B independent records. q/k are contiguous BF16 [128,Hqk,1,B],
 * v/out BF16 [128,Hv,1,B], and g/beta FP32 [Hv,1,B]. Source and destination fields are dense
 * [128,128,Hv] fields of one dtype, FP32 or BF16, within caller-owned records, selected by device
 * I64 [B] arena-relative byte offsets; a BF16 destination holds the updated state rounded to
 * nearest even. B is in [1, kMaximumConcurrency]: one row per decode lane of the execution model.
 * Each selected field must be 16-byte aligned and fully inside its arena.
 * A row may update in place. Destinations are distinct and do not overwrite other active sources.
 * Input/output tensors do not overlap selected state fields. This form uses no arena allocation.
 */
void gated_delta_net_batch_update(const Tensor& q, const Tensor& k, const Tensor& v,
                                  const Tensor& g, const Tensor& beta, float scale,
                                  bool normalize_qk, ConstStateFieldView source_state,
                                  StateFieldView destination_state, Tensor& out,
                                  CudaExecutionView execution);

/**
 * Row table for gated_delta_net_varlen. R rows are consecutive token ranges of one packed
 * sequence axis: `token_offsets` is device I32 [R+1], strictly increasing, with [0]=0 and [R]=T.
 * `source` and `destination` select dense [128,128,Hv] fields (one layer) of one dtype, FP32 or
 * BF16, with device I64 [R] record byte offsets. Row r starts from source record r and publishes
 * its final state to destination record r. A source may exactly alias its own row's destination.
 * Destinations of different rows do not overlap each other, any source other than the row's own,
 * or any checkpoint record.
 */
struct GdnVarlenRows {
    Tensor token_offsets;
    ConstStateFieldView source;
    StateFieldView destination;
};

/**
 * Traversal boundaries inside rows, sorted by (row, column). Entry i splits row rows[i] after
 * columns[i] tokens of that row: `rows` is device I32 [C] and non-decreasing, `columns` is device
 * I32 [C] with values in [1, tokens(row)-1], strictly increasing within a row. Entry i with
 * destination_offsets[i] >= 0 also writes the state after that column to the checkpoint record at
 * that arena-relative byte offset; -1 makes it a traversal boundary only. C=0 uses empty tensors.
 */
struct GdnVarlenBoundaries {
    Tensor rows;
    Tensor columns;
    Tensor destination_offsets; // device I64 [C]
};

/**
 * Returns the transient arena capacity required by gated_delta_net_varlen for every call whose
 * rows, tokens and boundaries do not exceed the arguments. The profile must be (Hq,Hv) in
 * {(16,32),(16,48),(16,64)} with max_rows in [1,1024], max_boundaries in [0,1024] and
 * max_tokens >= max_rows; otherwise std::invalid_argument is thrown.
 */
[[nodiscard]] std::size_t gated_delta_net_varlen_workspace_capacity_bytes(
    std::int32_t qk_heads, std::int32_t value_heads, std::int32_t max_rows,
    std::int32_t max_tokens, std::int32_t max_boundaries);

/**
 * Applies the recurrence of gated_delta_net to R independent rows of a packed token axis in one
 * call. q/k are BF16 [128,Hq,T], v/out BF16 [128,Hv,T], g/beta FP32 [Hv,T]. Each row is traversed
 * as consecutive sub-segments cut at its boundaries. The call is bit-identical to invoking the
 * distinct-state gated_delta_net once per sub-segment in order, with sub-segment k reading the
 * state that sub-segment k-1 produced (the row source for k=0), the last sub-segment writing the
 * row destination and every boundary with destination_offsets[i]>=0 writing its checkpoint
 * record (checkpoint records live in destination.records.storage and are laid out by
 * `checkpoint_field`, dense [128,128,Hv] of the row state dtype, one layer). Chunks start at the
 * row start and at every boundary. Intermediate state of a row may be staged in that row's
 * destination record.
 *
 * `row_count` is R in [1,1024] and `boundary_count` is C in [0,1024]; T is q.ne[2]. Row and
 * boundary tables are consumed on the device and are not read back. A profile or geometry outside
 * the supported domain throws std::invalid_argument naming Hq, Hv, R, C and T. `ws` supplies
 * the scratch reported by gated_delta_net_varlen_workspace_capacity_bytes, scoped to the call.
 * Inputs and out do not overlap each other, any source, destination or checkpoint record.
 */
void gated_delta_net_varlen(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& g,
                            const Tensor& beta, float scale, bool normalize_qk, GdnVarlenRows rows,
                            GdnVarlenBoundaries boundaries, StateFieldLayout checkpoint_field,
                            std::int32_t row_count, std::int32_t boundary_count, MemoryArena& ws,
                            Tensor& out, CudaExecutionView execution);

struct GdnRecurrenceRecords {
    Tensor key;   // BF16 [128,Hq,T,B], raw represented key words.
    Tensor value; // BF16 [128,Hv,T,B], raw represented value words.
    Tensor gate;  // FP32 [2,Hv,T,B], raw {g,beta} words.
};

/**
 * Evaluates B independent normalized Gated DeltaNet recurrences without modifying initial state.
 * q/k are BF16 [128,Hq,T,B], v/out BF16 [128,Hv,T,B], and g/beta FP32 [Hv,T,B].
 * initial_state selects dense FP32 or BF16 [128,128,Hv] fields using device I64 [B] record byte
 * offsets. Every selected field is 16-byte aligned and in bounds. B=1..8, T=2..16; (Hq,Hv) is
 * one of (16,32), (16,48), (16,64), (8,24). scale is 1/sqrt(128).
 *
 * valid_columns is empty for dense rows or device I32 [B] with values in [1,T]. Each valid
 * transition copies raw k, v and {g,beta} into the corresponding record fields. Invalid record
 * suffixes remain unchanged; invalid output suffixes are exact BF16 zero. Inputs, selected state
 * fields, records and output do not overlap. Valid outputs are bit-identical to normalized
 * recurrent one-token execution from the same represented inputs and initial state, so a BF16
 * state continues every token from its value rounded to nearest even.
 */
void gated_delta_net_replay_record(const Tensor& q, const Tensor& k, const Tensor& v,
                                   const Tensor& g, const Tensor& beta, float scale,
                                   ConstStateFieldView initial_state, const Tensor& valid_columns,
                                   GdnRecurrenceRecords records, Tensor& out,
                                   CudaExecutionView execution);

} // namespace ninfer::ops
