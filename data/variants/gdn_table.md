| kernel | V1 ms/flight | V1 ms/request | final ms/flight | final ms/request |
| --- | --- | --- | --- | --- |
| recurrence_kernel | 16.905 (n=1626) | 1.795 | - | - |
| register_recurrence_kernel | - | - | 8.425 (n=1890) | 0.939 |
| prepare_kernel | 5.744 (n=1914) | 0.610 | 1.300 (n=1998) | 0.145 |
| chunk_recurrence_kernel | 0.080 (n=288) | 0.008 | - | - |
| chunk_register_kernel | - | - | 0.015 (n=108) | 0.002 |
| varlen_output_kernel | 1.745 (n=1626) | 0.185 | 1.598 (n=1890) | 0.178 |
| varlen_state_kernel | 0.364 (n=1626) | 0.039 | 0.357 (n=1890) | 0.040 |
| plan_kernel | 0.267 (n=1626) | 0.028 | 0.259 (n=1890) | 0.029 |
| recurrent_batch_update_kernel | 15.374 (n=1590) | 1.632 | 15.269 (n=1926) | 1.702 |
| gdn_projected_conv_kernel | 1.104 (n=1590) | 0.117 | 1.109 (n=1926) | 0.124 |
| GDN kernels total | 41.58 | 4.415 | 28.33 | 3.158 |
| all kernels total | 104.89 | 11.136 | 86.53 | 9.645 |
# V1: window 5.08 s, flights 44.2, requests 416, util 0.912
# final: window 5.15 s, flights 53.5, requests 480, util 0.898
