# Eval results

One row per (run, set). mean@k ± stderr over tasks. Written by `evals/run.py`.

| date | run | checkpoint | mode | harness@version | set | n | mean@k | ± | invalid | trunc | resp tok |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | 2wiki_dev | 400 | 0.7991 | 0.0167 | 0 | 0.001 | 586 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa1_0k | 80 | 0.9969 | 0.0031 | 0 | 0.000 | 230 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa1_4k | 80 | 0.9688 | 0.0129 | 0 | 0.000 | 341 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa1_8k | 80 | 0.9406 | 0.0195 | 0 | 0.003 | 581 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa2_0k | 80 | 0.8938 | 0.0230 | 0 | 0.000 | 1409 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa2_4k | 80 | 0.6687 | 0.0402 | 0 | 0.003 | 1876 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa2_8k | 80 | 0.6250 | 0.0390 | 0 | 0.006 | 2246 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa3_0k | 80 | 0.5906 | 0.0411 | 0 | 0.009 | 3273 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa3_4k | 80 | 0.3344 | 0.0390 | 0 | 0.094 | 4284 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa3_8k | 80 | 0.3031 | 0.0389 | 0 | 0.150 | 4837 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa4_0k | 80 | 0.7719 | 0.0410 | 0 | 0.000 | 528 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa4_4k | 80 | 0.8281 | 0.0363 | 0 | 0.003 | 581 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa4_8k | 80 | 0.8063 | 0.0356 | 0 | 0.006 | 811 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa5_0k | 80 | 0.8250 | 0.0399 | 0 | 0.000 | 804 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa5_4k | 80 | 0.7906 | 0.0423 | 0 | 0.006 | 1018 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | babilong_qa5_8k | 80 | 0.8187 | 0.0403 | 0 | 0.006 | 1051 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | hotpotqa_distractor_dev | 400 | 0.7481 | 0.0176 | 0 | 0.000 | 653 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | musique_ans_dev | 400 | 0.6140 | 0.0201 | 0 | 0.001 | 1467 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | omanic_heldout | 967 | 0.5429 | 0.0139 | 0 | 0.001 | 3032 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | ruler_vt_4k | 160 | 1.0000 | 0.0000 | 0 | 0.000 | 605 |
| 2026-10-16 | parity_base_direct | Qwen/Qwen3-8B | direct | — | ruler_vt_8k | 160 | 1.0000 | 0.0000 | 0 | 0.000 | 630 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | 2wiki_dev | 400 | 0.8006 | 0.0169 | 0 | 0.000 | 568 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa1_0k | 80 | 1.0000 | 0.0000 | 0 | 0.000 | 231 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa1_4k | 80 | 0.9781 | 0.0142 | 0 | 0.000 | 372 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa1_8k | 80 | 0.9437 | 0.0189 | 0 | 0.000 | 441 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa2_0k | 80 | 0.8906 | 0.0263 | 0 | 0.000 | 1273 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa2_4k | 80 | 0.6687 | 0.0385 | 0 | 0.009 | 2086 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa2_8k | 80 | 0.6031 | 0.0442 | 0 | 0.000 | 2058 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa3_0k | 80 | 0.6312 | 0.0442 | 0 | 0.016 | 3208 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa3_4k | 80 | 0.3625 | 0.0407 | 0 | 0.084 | 4057 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa3_8k | 80 | 0.2969 | 0.0370 | 0 | 0.144 | 4807 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa4_0k | 80 | 0.7750 | 0.0411 | 0 | 0.000 | 524 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa4_4k | 80 | 0.8156 | 0.0369 | 0 | 0.003 | 568 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa4_8k | 80 | 0.8094 | 0.0351 | 0 | 0.009 | 817 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa5_0k | 80 | 0.8406 | 0.0396 | 0 | 0.000 | 838 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa5_4k | 80 | 0.8031 | 0.0416 | 0 | 0.006 | 974 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | babilong_qa5_8k | 80 | 0.8313 | 0.0395 | 0 | 0.003 | 1026 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | hotpotqa_distractor_dev | 400 | 0.7476 | 0.0177 | 0 | 0.000 | 632 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | musique_ans_dev | 400 | 0.6208 | 0.0200 | 0 | 0.000 | 1484 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | omanic_heldout | 967 | 0.5419 | 0.0139 | 0 | 0.002 | 3057 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | ruler_vt_4k | 160 | 1.0000 | 0.0000 | 0 | 0.000 | 605 |
| 2026-10-16 | parity_base_pi_vanilla | Qwen/Qwen3-8B | pi | vanilla@ffaf8d9b | ruler_vt_8k | 160 | 1.0000 | 0.0000 | 0 | 0.000 | 631 |
