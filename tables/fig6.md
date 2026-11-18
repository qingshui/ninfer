|  | HEAD | Final |
| --- | --- | --- |
| Lanes per group of n samples | n + 1 (hidden helper) | n |
| Checkpoint capture | context operation, State copy, quiescent iteration | written in the final prefill window, published at retirement |
| Fork | one State copy per sample, one context operation | one fork_batch call, shared frozen StateAnchor |
| Prefill tokens per flight (median) | 62 | 3,837 |
| Decode rows per flight (median) | 1.6 | 312 |
