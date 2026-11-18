### V1 gdn_head

| run | req/s | mean E2EL (s) | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | completed/failed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 85.87 | 1.497 | 223 | 1294 | 39.32 | 2000/0 |
| 2 | 85.29 | 1.509 | 219 | 1222 | 39.93 | 2000/0 |
| 3 | 85.55 | 1.503 | 223 | 1257 | 39.81 | 2000/0 |
| mean | 85.57 | 1.503 | 222 | 1258 | 39.69 | |

range 85.29 to 85.87 req/s, sample stdev 0.29, n=3

### V2 moe_atomic

| run | req/s | mean E2EL (s) | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | completed/failed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 93.70 | 1.371 | 209 | 1159 | 36.06 | 2000/0 |
| 2 | 93.48 | 1.374 | 205 | 1095 | 36.22 | 2000/0 |
| 3 | 92.70 | 1.386 | 203 | 1164 | 36.53 | 2000/0 |
| mean | 93.29 | 1.377 | 206 | 1139 | 36.27 | |

range 92.70 to 93.70 req/s, sample stdev 0.52, n=3

### control (final build, same session)

| run | req/s | mean E2EL (s) | mean TTFT (ms) | p99 TTFT (ms) | mean TPOT (ms) | completed/failed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 94.00 | 1.367 | 207 | 1100 | 35.86 | 2000/0 |
| 2 | 93.92 | 1.368 | 202 | 1099 | 36.20 | 2000/0 |
| 3 | 94.41 | 1.361 | 205 | 1133 | 35.93 | 2000/0 |
| mean | 94.11 | 1.365 | 205 | 1111 | 35.99 | |

range 93.92 to 94.41 req/s, sample stdev 0.26, n=3

### Deltas

- V1 gdn_head vs interleaved control: -8.54 req/s (-9.07%), mean E2EL +0.138 s, mean TTFT +17 ms
- V1 gdn_head vs final build runs earlier today (final_c1..3): -8.88 req/s (-9.40%), mean E2EL +0.143 s, mean TTFT +19 ms
- V1 gdn_head minus control per round: -8.13, -8.62, -8.86 (mean -8.54 req/s)
- V2 moe_atomic vs interleaved control: -0.82 req/s (-0.87%), mean E2EL +0.012 s, mean TTFT +1 ms
- V2 moe_atomic vs final build runs earlier today (final_c1..3): -1.15 req/s (-1.22%), mean E2EL +0.017 s, mean TTFT +3 ms
- V2 moe_atomic minus control per round: -0.30, -0.44, -1.70 (mean -0.82 req/s)
- control vs final build runs earlier today (final_c1..3): -0.34 req/s (-0.36%), mean E2EL +0.005 s, mean TTFT +2 ms

final_c1..3 earlier today: 95.27, 94.22, 93.85 (mean 94.45, range 93.85 to 95.27)
