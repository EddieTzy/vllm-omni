# Higgs v2 batched RAS repeat detection — RTX PRO 6000

Experimental, **no upstream PR**: component gain did not establish serving gain.
Branch `perf/higgs-v2-ras-repeat`; measured code commit d509d6e. All new follow-up
experiments independently start from mask PR11119 head0ee2017, including merged
routing; RAS is not stacked with LM mode mask, bounded history or state fusion.
Runtime uses649e+identical mask model bytes. Rebased PR's a57 changes WebSocket
ping/pong, not this HTTP workload. Same model/runtime/C32 YAML/public corpus as
`perf/higgs-v2-logits-mask`, never combine percentage gains across experiments.

The change stacks each eligible seven-frame history window, compares all
codebook repeat counts together, and transfers the small flags table once.
Request/codebook ordering of softmax and multinomial is preserved. GPU sampling
is still per request. No persistent state pool or new public config.

Validation:30 CPU/GPU component cases compare flags, sampled codes and RNG;
43 initial regression tests passed. Actual model, controlled admission and first
audio RNG:32/32 full PCM and code histories equal, all200normalized sample/RNG
events equal. This is not a broad ASR, speaker or perceptual quality guarantee.
N32 component medians: no-repeat detection1.8888->.1310ms; repeat-heavy whole
RAS6.1686->4.5589ms. These synthetic component workloads do not prove serving
speedup. N1 component timing is not C1 serving.

C32 A/B/B2/A2, two fresh boots per variant,32 warmup+64 measured requests each,
all256measured successful, same unseeded HTTP client and pinned configuration:

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| audio seconds/s |24.4329|22.8563|-6.45%|
| mean E2E s |3.6682|3.7123|+1.20%|
| mean request RTF |.89573|.88216|-1.52%|
| first PCM P50 s |3.57661|3.71442|+3.85%|
| first PCM P95 s |5.06970|5.13055|+1.20%|

Boot throughput baseline22.20995/26.65583, candidate23.26700/22.44561. Preserve
this variance and negative result; do not infer a confirmed regression cause.
E2E includes queueing/talker/codec; first PCM is near completion here. RTF is
mean per-request E2E/audio duration, not throughput's inverse. No concurrency
other than the existing C32 was measured. Four disconnect/WAV recovery probes
passed (not proof of server cancellation acknowledgement or memory leak absence).

First fixed controller refused a pre-existing inherited directory before any
service boot; corrected using fresh r1 labels and retained failure log. Initial
pytest command named a nonexistent old-module path and collected no tests;
corrected command passed43tests. No performance run was discarded.

JSON artifacts preserve each boot and the paired controls. Stage profiling is
pending separately, never substituted for unprofiled serving numbers. Do not
submit this candidate until it demonstrates reproducible deployment value.
AI assistance: Codex assisted with implementation, experiments and writing.
