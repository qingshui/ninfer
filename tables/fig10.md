| item | HEAD | Final |
| --- | --- | --- |
| Threads per CTA | 256 | 128 |
| Dynamic shared memory per CTA | 61,952 B | 40,704 B |
| __syncthreads per 16-token chunk | 4 | 1 |
| cp.async stages | 2 | 3 |
| CTAs per SM | 2 (25% occupancy, measured) | asks for 3 (single) / 4 (varlen) |
| Profile of the chunk step | L1/TEX 93%, compute 39% | tensor pipe 41-46%, issue ~47% |
| Recurrence kernel, isolated bench (us) | 353 | 178 |
| Varlen recurrence per production flight (ms), development notes | 11.7 | 7.44 |
| Kernels in the reverted files per request (ms), nsys, final engine | 2.442 | 1.114 |
| Throughput, interleaved A/B on the final engine (req/s) | 85.57 | 94.11 |
