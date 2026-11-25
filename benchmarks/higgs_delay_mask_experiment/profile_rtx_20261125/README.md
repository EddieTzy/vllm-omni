# Real-service C32 mask stage attribution — RTX PRO 6000, 2026-11-25

Private, identical diagnostic instrumentation of the baseline and candidate;
no profiling instrumentation is part of upstream PR #11119.

Measured baseline 649e7b3b9a5b93f9a33f753942df149ba939e523 already includes
routing PR #11068. Candidate model bytes match PR head
0ee2017a6a59808544f7cacee49cd9d2dfce48e8 (see validation.json).
The PR was rebased on a57d6d1a73c694476ae1cc1447bcd765dfd47dd7;
intervening changes only affect WebSocket ping/pong, not this HTTP workload.
Same RTX PRO 6000 Blackwell 96GB, pinned runtime/model, public corpus and
reproduction/c32.yaml as the unprofiled serving experiment in the parent folder.
A/B/B2/A2, two fresh boots per variant, 32 warmup +32 diagnostic measured
requests per boot; all 128 measured requests succeeded. No other concurrency.

## Selection and timing

Capture CPU wall time and CUDA event intervals for real service stages.
Select single-token-per-request steps with exactly 32 requests, CPU-visible
previous tokens all audio, and exactly 32 audio rows in the mask. Drop the first
16 eligible C32 occurrences, profiler steps and the next two steps. Timings
include eligible steps from both warmup and diagnostic traffic; they are not
restricted to the post-warmup client window. Retained steps A/B/B2/A2:
97/96/101/107. Query completed events asynchronously; do not synchronize each
normal timing region. CUDA intervals include CPU launch gaps and are not pure
kernel busy time. Nested inclusive stage timings must not be added together.

Below: mean of the two per-boot CPU-wall medians, milliseconds. Full per-boot
sample counts, means, medians, p95 and CUDA intervals are in summary.json.

| Region | Baseline | Fused mask |
| --- | ---: | ---: |
| talker.forward | 23.1629 | 22.2722 |
| talker.audio_head | 0.2181 | 0.1937 |
| mask.delay_pattern | 6.2305 | 0.1313 |
| audio.sample_codes | 0.9670 | 0.9148 |
| audio.ras | 2.7094 | 2.5373 |
| audio.state_update | 2.2350 | 2.1992 |
| talker.sample_step | 15.9686 | 9.6047 |

Mask latency drops 97.89%; sampling-step latency drops 39.85%. Successive
forward-start intervals (valid adjacent selected steps) have boot medians
44.802/45.111 ms baseline versus 37.284/36.278 ms candidate; their boot-median
means are 44.9565 versus 36.7806 ms (-18.19%). These instrumented cycle timings
are diagnostic, not replacement end-to-end throughput measurements.

Forward, head, RAS and state-update code are unchanged by the PR. Their measured
variation also reflects serving schedules, attention context, CPU contention
and run variability. Output/sample histories in these unseeded diagnostic runs
are not paired. Do not attribute the whole cycle difference solely to the mask,
or treat these phase medians as an exact decomposition of the earlier +19.44%
unprofiled audio-throughput result. Codec timings are excluded from this table
because codec input work/batching is not matched.

## CPU/CUDA trace evidence

Traces surround the actual MRv2 sampler.sample_step, including head, mask,
audio sampling, RAS and state updates. Capture at C32 occurrence 64 and 128
when reached: A/B/B2 each reached one capture; A2 reached both. All five traces
have 32 previous audio-token IDs (128016). Capture count is not padded with
additional requests. Profiler-perturbed steps are excluded from phase timing.

In steady baseline trace A, mask has 480 aten::copy_ operations, 480 H2D
copies and 480 cudaStreamSynchronize calls, plus 1952 aten::select operations.
Its H2D device busy time sums to only 0.153 ms: most wall time is dispatch and
synchronization, rather than arithmetic. Baseline A2's different delay states
produce 480/484 copies, 144/450 synchronizations and 168/17 fill kernels.
This spread is retained; traces are not paired identical delay-state snapshots.

Both candidate traces show one state-table H2D copy, one stream synchronization
and exactly one _mask Triton kernel, grid [32,8,1], 1.12 microseconds device time.
Total mask-region CPU median remains ~0.13 ms because it also includes Python
state collection, table construction/transfer and launch overhead. Do not
quote kernel-only time as the whole mask latency. Trace parser links ATen
kernels via External id and Triton kernels via CUDA driver correlation id.

The optimized layer is the audio delay-pattern logits processor inside
_dispatch_audio_codes, after the codebook head and before sampling. It is
not a transformer forward/attention/FFN change and does not fuse RAS or request
state updates. Existing 29 regression tests and controlled 32/32 full-PCM parity
remain the correctness evidence; profiling adds no broader quality claim.

## Artifacts and reproduction

profile_support.py and run_profile.py build isolated source copies, insert
identical phase contexts, and launch each arm sequentially. Never modify the PR
source to enable profiling. Adjust ROOT if reproducing on another machine;
use the parent's pinned runtime, launcher, model and YAML/corpus. summarize.py
consumes the extracted per-arm JSONL and Chrome traces; run:

    python summarize.py /path/to/extracted/evidence

The compressed bundle contains only explicitly selected JSONL/trace JSON files,
not service logs, prompt strings, generated audio, or credentials. Torch CPU
operators record shapes but no tensor contents; trace metadata stores only the
32 previous audio-token IDs. Traces may contain benign runtime paths. Services
were stopped after all four boots; GPU memory returned to 0 MiB.

Evidence bundle SHA256: d3a370dd7abbd1be47ad6ca9132251a52ab8858c5d7af60c8beb6368a05b4a49
