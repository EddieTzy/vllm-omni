# Higgs v2 routing profiling for PR #11068

Completed on 2026-11-24. Four independent boots in A/B/A2/B2 order reproduced the reduction in talker-forward overhead. A and A2 are the original routing; B and B2 contain only the three-line uniform-audio routing change. These private diagnostic runs supplement, rather than replace, the original uninstrumented throughput benchmark.

## Source and environment

- Baseline: `4dafb464442956996be9633ed036cd24ebfef1d0`.
- Measured routing revision: `30dcfdc205d17072734328e74c5ca96a882ec548`, whose parent is the baseline above.
- The patched talker is byte-identical to the talker in current PR commit `c70831d97aea6a1ad038706b2b58af6609cefb26`.
- Compared all 2,381 runtime Python files in each source snapshot: only `higgs_audio_v2_talker.py` differs. The controller verifies that removing precisely the three published routing lines reconstructs the original file. Identical instrumentation is appended in independent diagnostic source copies; no PR source is modified.
- Original talker SHA256: `df54c3435cafadef9f600babacd90ae3e9ab4bc4064bf0afe6b437772e25f336`.
- Patched talker SHA256: `9dda0618f1a76b6f4b8c750ad6d4b6c095058945f05ec78b27ee35be9663a37a`.
- One NVIDIA B200, 183,359 MiB, driver 580.173.02; torch 2.14.0, vLLM 0.34.0, FlashInfer 0.7.3, transformers 5.22.0. Existing pinned runtime and offline models reused.
- Model: `bosonai/higgs-audio-v2-generation-3B-base@d80c511612b3040ff2877ce3d408747df1739f11`.
- Codec: `bosonai/higgs-audio-v2-tokenizer@815a03f1b57dcb5ae918db415f22cd0c8a6f0f54`.

## Protocol

Same C32 two-stage MRv2 eager configuration as the original benchmark: TP=PP=1, BF16, max_num_seqs=32 in both stages, async_chunk=false, no prefix caching or async scheduling; identical sampling defaults. The existing streaming PCM client omits request seeds to permit co-batching.

Each boot sends one wave of 32 warmups followed by one wave of 32 measured requests. Both waves use the first 32 entries of the existing interleaved EN/ZH public corpus. This is a smaller diagnostic workload than the original two measured waves. All 256 requests, including warmups, completed successfully; 128 measured requests were retained by the client on the GPU host.

Top-level wrappers measure talker.forward, compute_logits and codec.forward using host clocks and CUDA events on the current stream. Event results are harvested without synchronizing every forward. Two real 32-row forwards per boot are separately profiled with torch.profiler; all eight trace metadata records confirm that every input ID is the audio token 128016. Routing helpers are annotated only while collecting those traces.

The non-trace forward summary selects exactly 32 input tokens, discards the first 16 occurrences of that shape, and excludes traced forwards and their next two forwards. This stage summary covers real forwards from both client waves after those exclusions; it is not restricted to the client's measured wave. Consecutive selected forwards also define a decode-cycle interval from one forward start to the next. The remainder is that interval minus the current forward's host duration; it includes logits, sampling, runner/scheduler work, instrumentation and any waiting. It is **not** an isolated sampling measurement.

## Non-trace stage measurements

All times are milliseconds. Each cell lists the two independent boots, rather than pooling their observations.

| Measurement | Original A / A2 | Patched B / B2 |
|---|---:|---:|
| 32-token forward samples | 98 / 97 | 89 / 107 |
| Forward host mean | 33.866 / 34.365 | 16.702 / 17.062 |
| Forward host median | 34.639 / 35.248 | 17.069 / 17.386 |
| Forward host P95 | 35.488 / 36.484 | 17.503 / 17.715 |
| Forward CUDA-event interval median | 34.669 / 35.278 | 17.122 / 17.415 |
| Consecutive selected decode cycles | 94 / 93 | 85 / 104 |
| Decode-cycle mean | 52.251 / 52.356 | 33.480 / 33.966 |
| Decode-cycle median | 51.641 / 52.167 | 34.193 / 34.582 |
| Decode-cycle P95 | 54.462 / 54.555 | 35.920 / 36.215 |
| Outside-forward remainder median | 16.942 / 16.933 | 16.945 / 17.083 |

The average of the per-boot forward medians drops from 34.943 to 17.227 ms, about 50.7%. The decode-cycle medians drop from about 51.9 to 34.4 ms, while the outside-forward remainder medians remain near 17 ms. This localizes the observed step-time improvement to talker.forward and supports the routing-overhead explanation for the separately measured serving gain. It does not prove a request-level additive decomposition of the original 50.8% throughput result.

CUDA-event intervals include CPU launch gaps and same-stream waits; they are not summed GPU kernel compute time. Similar host and event intervals therefore do not imply that the GPU was continuously executing kernels.

## Operator evidence

Every sampled trace in each variant has the counts below. These are counts over one whole 32-row all-audio talker forward, including common preprocessing and profiler boundary effects.

| Operation | Original | Patched |
|---|---:|---:|
| Routed norm helper calls | 56 | 56 |
| Routed MLP helper calls | 28 | 28 |
| `aten::any` | 168 | 0 |
| `aten::all` | 0 | 1 |
| `aten::item` / `_local_scalar_dense` | 168 / 168 | 1 / 1 |
| `aten::nonzero` | 169 | 1 |
| `aten::index` | 84 | 0 |
| `aten::_index_put_impl_` | 84 | 0 |
| `cudaStreamSynchronize` runtime calls | 339 | 4 |
| `cudaMemcpyAsync` runtime calls | 396 | 5 |
| `cudaLaunchKernel` runtime calls | 1,739 | 788 |

The remaining nonzero is outside the annotated layer-routing helpers. Each forward still invokes all 56 norm and 28 MLP helpers; this change removes dispatch/indexing overhead rather than skipping the audio network.

The Chrome traces show substantial CPU launch/synchronization overhead: summed kernel durations are about 5.99–6.14 ms in original traces and 3.985–4.001 ms in patched traces. These traced kernel sums exclude idle gaps and are not compared directly with the non-trace forward wall times. Annotated CPU routing spans also shrink, but their absolute durations are profiler-perturbed and are not used as production timing claims. For annotations, use Chrome `cat=user_annotation`; this torch build's key_averages view reports misleading zero CPU totals for those custom ranges.

## Other stages and limits

Descriptive codec.forward host medians are 10.383 / 9.892 ms for A/A2 and 10.370 / 10.250 ms for B/B2. Generated lengths and batching differ, so these are **not matched codec workloads** and do not establish codec equivalence or speedup. Codec source is unchanged. compute_logits host medians remain about 0.069–0.071 ms. The outside-forward interval above is the more useful contextual control, and it must not be mislabeled as pure sampler time.

This run adds profiling evidence, not a new quality experiment. Existing controlled output-equivalence checks remain the correctness evidence. No exact request critical-path breakdown, general hardware claim, or new production throughput number is asserted. There are only two boots per variant, and per-step samples within a boot are not independent benchmark replicates.

All four logs contain the same startup chat-template warmup warning about a missing system message; the speech endpoint became healthy and all diagnostic speech requests succeeded. Full logs are retained.

## Artifacts and reproduction

- [Download complete evidence archive](evidence.tar.gz): all eight Chrome traces, JSONL events, complete summaries, logs, source snapshots, exact configuration, corpus, and reproduction scripts.
- [Readable metrics](metrics.json), [source identity](source-identity.json), [manifest](manifest.json), [environment](environment.json), [hardware](hardware.csv), and [validation](validation.json).
- [Controller](scripts/run_profile.py), [instrumentation](scripts/profile_support.py), [summarizer](scripts/summarize.py), [launcher](scripts/run-server.sh), and [streaming client](scripts/screen.py).
- [Exact deployment config](reproduction/c32.yaml) and [corpus](reproduction/public-corpus.json).

Archive SHA256: `054a8fc25293978eadb226565304fc0f5f1ff240c1595fa264dc335660d6c3f9`.

Download and extract the archive in a fresh directory. Raw data are under `b200-results/results/higgs-routing-profile/`; the summarizer is under `b200-setup/higgs-routing-profile/`:

```bash
python b200-setup/higgs-routing-profile/summarize.py \
  b200-results/results/higgs-routing-profile
```

The GPU-host controller reuses the documented prepared runtime, model snapshots, client and deployment config at `/home/tzy/zyb-fish-pr`. It refuses to overwrite existing result/source directories and refuses to start a boot while another GPU compute process is present. A rerun needs fresh isolated result/source names, while retaining the same inputs and configuration. This is an environment-specific reproducer, not a turnkey installer.

All four diagnostic serving containers were stopped after collection; GPU memory was verified at 0 MiB with 0% utilization. This evidence branch adds only experimental artifacts; the PR's production-code branch is separate.

AI assistance: Codex assisted with instrumentation, experiments, validation and documentation.
