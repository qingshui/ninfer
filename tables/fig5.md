| category (ms per request) | HEAD | Final, original artifact | Final, canonical artifact | HEAD / final |
| --- | --- | --- | --- | --- |
| MoE routed experts | 156.38 | 2.75 | 2.60 | 60x |
| Dense linear (GEMM) | 97.23 | 3.65 | 2.26 | 43x |
| GDN decode state update | 2.87 | 1.74 | 1.70 | 2x |
| GDN prefill (varlen/chunked) | 24.43 | 1.59 | 1.46 | 17x |
| Full attention | 32.39 | 0.80 | 0.74 | 44x |
| MoE routing and shared expert | 50.97 | 0.51 | 0.48 | 106x |
| Norm, sampling, other | 9.98 | 0.44 | 0.41 | 25x |
| Total GPU kernel time per request | 374.26 | 11.47 | 9.64 | 39x |
| GPU idle share of the window | 17.3% | 8.1% | 10.2% | - |
