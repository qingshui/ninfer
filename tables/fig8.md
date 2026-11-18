| item | before | after |
| --- | --- | --- |
| Gate/up GEMM, 3700 tokens, skew 0.4 (us) | 453 | 212 |
| Routed op, 3700 tokens, skew 0.4 (us) | 765 | 481.5 |
| Routed op, atomic combine against ordered combine (us) | 480.5 | 437.7 |
| Complete op with shared experts, ordered combine (us) | - | 650.7 |
| End to end, locked harness (req/s) | 59.45 to 65.59 to about 67.4 |  |
