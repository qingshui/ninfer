import os
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
import sys
sys.path.insert(0, HERE)
from viz_core import Scene
from viz_charts import header, footer, save, md_table
from viz_flow import column, kv_table, strip

BEFORE = [
    (["Load chunk tiles", "K, Q, V by cp.async, 2 stages"], None, "16-token chunk, 256 threads"),
    (["Snapshot the FP32 state to shared memory", "128x64 state, 32 KB written every chunk"], "s2",
     "The state lives in registers between chunks and is copied to shared memory for each chunk step"),
    (["MMA with operands read from shared memory", "scalar LDS.32 B fragments, 2.4-way bank conflicts"], "s2",
     "ncu: L1/TEX 93%, compute 39%, mio_throttle 2.30 and short_scoreboard 1.92 of 9.16 stall cycles per instruction"),
    (["Update the state and synchronize", "4 __syncthreads per chunk"], "s2", "barrier stall 0.63 cycles per instruction"),
]
AFTER = [
    (["Load chunk tiles", "K, Q, V by cp.async, 3 stages, V^T by ldmatrix"], None, "16-token chunk, 128 threads"),
    (["State stays in MMA accumulators", "each warp owns 16 value rows, 64 FP32 per lane"], "s1",
     "S^T[value, key] is never copied to shared memory"),
    (["MMA with operands taken from registers", "TF32 A fragments straight from the accumulators"], "s1",
     "72 MMAs per warp per chunk; K.S in TF32, Q.S on the BF16-rounded state"),
    (["Update the state and synchronize", "1 __syncthreads per chunk"], "s1", "FP32 master state is kept"),
]
ROWS = [
    ("Threads per CTA", "256", "128"),
    ("Dynamic shared memory per CTA", "61,952 B", "40,704 B"),
    ("__syncthreads per 16-token chunk", "4", "1"),
    ("cp.async stages", "2", "3"),
    ("CTAs per SM", "2 (25% occupancy, measured)", "asks for 3 (single) / 4 (varlen)"),
    ("Profile of the chunk step", "L1/TEX 93%, compute 39%", "tensor pipe 41-46%, issue ~47%"),
]


def build(outdir):
    W = 880
    sc = Scene(W, 640, "Gated DeltaNet chunk step before and after",
               "Schematic of the varlen GDN recurrence chunk step with a shared-memory state snapshot and with a register-resident state")
    y = header(sc, "Gated DeltaNet recurrence: one 16-token chunk step", "Schematic, not to scale. Orange boxes are replaced by the blue ones.")
    sc.text(24, y + 16, "Before: HEAD recurrence_kernel<64>", 13, "ink1", "start", True)
    sc.text(450, y + 16, "After: register-state chunk step", 13, "ink1", "start", True)
    yb = column(sc, 24, y + 28, 406, 52, BEFORE)
    column(sc, 450, y + 28, 406, 52, AFTER)
    yt = kv_table(sc, 24, yb + 30, 832, ("", "HEAD", "Final"), ROWS, 0.40, 0.64)
    strip(sc, 24, yt + 8, 832, "Isolated bench:", "recurrence kernel 353 to 178 us (8 rows, 4270 tokens, no boundaries)")
    strip(sc, 24, yt + 40, 832, "Production flights:", "varlen recurrence 11.7 to 7.44 ms per flight (nsys); end to end 87.3 to 90.7 req/s", "b100")
    strip(sc, 24, yt + 72, 832, "Controlled A/B:", "final engine with HEAD's GDN kernels 85.57 req/s, with the final kernels 94.11 (+10.0%)", "b100")
    y = footer(sc, yt + 114, [
        "Numbers come from ncu, nsys and microbenchmarks run on GPU 1 during development and are recorded in notes; they were not re-measured for this report.",
        "The controlled A/B was run for this report: the four GDN recurrence files reverted to HEAD, everything else final, three interleaved rounds.",
        "Contract: varlen output, row destinations and checkpoints equal the per-sub-segment single-segment op bit for bit and stay within FP64-reference budgets.",
        "The new chunk step is not bit-identical to HEAD's: it uses two TF32 accumulation chains."])
    sc.h = y + 14
    svg, png = save(sc, outdir, "fig10_gdn_chunk_step")
    table = md_table(["item", "HEAD", "Final"], [list(r) for r in ROWS] + [
        ["Recurrence kernel, isolated bench (us)", "353", "178"],
        ["Varlen recurrence per production flight (ms), development notes", "11.7", "7.44"],
        ["Kernels in the reverted files per request (ms), nsys, final engine", "2.442", "1.114"],
        ["Throughput, interleaved A/B on the final engine (req/s)", "85.57", "94.11"]])
    return svg, png, table


if __name__ == "__main__":
    print(build("/tmp/figtest")[2])
