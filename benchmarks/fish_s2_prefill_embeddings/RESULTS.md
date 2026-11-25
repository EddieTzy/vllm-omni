# C32 流式克隆测量

| 指标 | A | B | B2 | A2 | 基线均值 | 候选均值 | 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 音频秒/秒 | 23.396 | 22.799 | 23.875 | 23.539 | 23.468 | 23.337 | -0.56% |
| 首 PCM P50（ms） | 1670.770 | 1691.554 | 1787.420 | 1691.870 | 1681.320 | 1739.487 | +3.46% |
| 首 PCM P95（ms） | 1750.973 | 1743.953 | 1915.841 | 1784.903 | 1767.938 | 1829.897 | +3.50% |
| 平均请求耗时（秒） | 3.770 | 3.783 | 3.824 | 3.778 | 3.774 | 3.804 | +0.79% |
| 平均请求 RTF | 1.100 | 1.106 | 1.112 | 1.109 | 1.104 | 1.109 | +0.48% |

## Profile and correctness

| Serving arm | Clone calls | Eligible calls | Host median (µs) | Raw captures |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 96 | 96 | 664.5 | 3 |
| Candidate | 96 | 96 | 7.1 | 3 |

The host median excludes the three traced calls and excludes hashing. It times the marked region from the serving thread. Pending GPU work can affect the host wait.

| Arm | Trace | Marked region (µs) | D→H bytes | H→D bytes |
| --- | ---: | ---: | ---: | ---: |
| A | trace-4.json | 1200.3 | 1,280,000 | 1,280,000 |
| A | trace-40.json | 1126.1 | 1,290,240 | 1,290,240 |
| A | trace-78.json | 1497.3 | 1,285,120 | 1,285,120 |
| B | trace-4.json | 97.9 | 0 | 0 |
| B | trace-40.json | 90.7 | 0 | 0 |
| B | trace-78.json | 101.0 | 0 | 0 |

94 occurrences matched across the two diagnostic processes by embedding SHA-256, tensor shape and scheduled span. In warmup, 30 of 32 matched. In each subsequent wave, all 32 matched (64/64 post-warmup). The two processes saw different prompt lengths on their first call (258 and 265). The traces do not establish the cause of the two cold-run differences. See [the per-wave counts](embedding-phases.json). The CUDA controls matched in 8 deterministic cases. The upstream Fish prompt and regression tests passed (16/16), as did pre-commit on the changed model file.

## Decision

Keep the fast path as a measured experiment. All three sampled candidate regions removed the staging copies. Across the two fresh processes per arm, throughput and time to first audio did not show a consistent serving improvement. No upstream PR was opened for this change.
