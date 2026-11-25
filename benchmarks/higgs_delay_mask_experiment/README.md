# Higgs v2 delay-pattern mask experiment

This perf branch isolates the BOS/EOS logits mask change. Its base already contains merged audio-routing PR #11068; it does not contain the experimental decode-state fusion. Upstream PR [#11119](https://github.com/vllm-project/vllm-omni/pull/11119) is open; its branch contains only the three implementation/test files.

[Full results and limitations](RESULTS.txt) · [Serving arms](serving_results.json) · [Paired output check](paired_results.json) · [Validation](validation.json) · [Source/model pins](manifest.json)

## Result

One RTX PRO 6000 Blackwell Server Edition, 96GB, SM120, driver 580.173.02. PyTorch 2.14.0+cu130, vLLM 0.34.0, FlashInfer 0.7.3, Triton 3.8.0, transformers 5.22.0. Base `649e7b3b9a5b93f9a33f753942df149ba939e523`; measured code `273179aa656aba22a10ff760fbc927b389c857a4`.

Two independent boots each, A/B/B2/A2. Each boot has 32 warmups and 64 measured C32 requests in two waves. No speech-request seed, no profiler. Both original and candidate use the same eager MRv2 config. No additional concurrency levels were measured.

| Boot-mean metric | Baseline with routing | Routing + mask kernel | Change |
| --- | ---: | ---: | ---: |
| Audio seconds / wall second | 19.6806 | 23.5063 | +19.44% |
| Mean request E2E, seconds | 4.8095 | 4.1312 | -14.10% |
| Mean per-request RTF | 1.1333 | .9956 | -12.15% |
| First-audio P50, seconds | 4.7528 | 4.1907 | -11.83% |
| First-audio P95, seconds | 6.7390 | 5.3993 | -19.88% |

All 256 measured requests succeeded. All four HTTP disconnect/recovery probes returned valid follow-up audio. Sampled memory peaks include configured pools and warmup; no memory-saving claim. Both candidate boots are retained, including their throughput spread. Unseeded output lengths vary and two boots are descriptive evidence only.

The prototype component comparison is separate from serving: at N32, old loop / PyTorch broadcast / Triton median 5.681 / .224 / .040ms including metadata H2D. These are not real-model phase timings. The integrated kernel and existing MRv2 module pass 29 tests. A separate matched admission/RNG diagnostic gives 32/32 full PCM matches, 32/32 code-history matches and 200/200 matching sampling/RNG events.

## Reproduction

This is an environment-specific reproducer using the already prepared runtime and pinned public model/codec snapshots in `manifest.json`. The GPU-host root is `/home/tzy/zyb-fish-pr`, mounted at `/data`; both sources are exported snapshots. It is not an environment installer.

Use [c32.yaml](reproduction/c32.yaml) for both revisions and [public-corpus.json](reproduction/public-corpus.json) for every arm. The config increases the shipped max sequences from 4 to 32 for measurement only; no deployment defaults change in the implementation. It retains stage GPU fractions .6/.25, token budgets 2048/65536, lengths 4096/65536, TP=PP=1 and eager execution. The corpus selects the first 32 distinct EN/ZH targets within fixed 6–20 EN-word / 10–36 ZH-character bounds, then interleaves them.

The existing launcher is also published with the [routing profiling reproducer](https://github.com/EddieTzy/vllm-omni/blob/30aa5bc7d9007cbddbb9ca0cfbd7680526970f49/benchmarks/higgs_audio_routing_profile/scripts/run-server.sh). For baseline arm A:

```bash
HIGGS_CONTAINER_NAME=zyb-higgs-mask-rtx-perf-A \
HIGGS_SOURCE_DIR=/home/tzy/zyb-fish-pr/vllm-higgs-mask-base \
HIGGS_PORT=18844 HIGGS_DEPLOY_CONFIG=/data/higgs-routing/c32.yaml \
HIGGS_LOG_STATS=1 bash /home/tzy/zyb-fish-pr/higgs-routing/run-server.sh
```

After `/health` returns 200, run the checked-in client in the same prepared runtime. Place the archived client at `/home/tzy/zyb-fish-pr/higgs-logits-mask/benchmark_client.py` (container path below). It sends streaming 24 kHz signed-16-bit mono PCM requests, omits the speech seed, warms 32 requests, then measures 64 requests in two waves of 32:

```bash
sudo -n docker run --rm --network host --user "$(id -u):$(id -g)" \
  -v /home/tzy/zyb-fish-pr:/data \
  --entrypoint /data/venv-vllm/bin/python tts-runtime:20261117-pinned \
  /data/higgs-logits-mask/benchmark_client.py \
  --port 18844 --label higgs-mask-rtx-perf-A \
  --concurrency 32 --count 64 \
  --corpus /data/higgs-routing/public-corpus.json \
  --omit-request-seed --request-timeout 360
```

Stop the server before starting the next arm. Repeat B/B2 with `vllm-higgs-mask-main`, A2 with `vllm-higgs-mask-base`, using distinct names, labels and ports 18845/18846/18847. Retain all arms; aggregate per-boot means rather than selecting the fastest result. The client output directory must be new.

GPU kernel and MRv2 regression commands, from the candidate checkout in the compatible environment:

```bash
PYTHONPATH="$PWD" PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  -p pytest_mock -p xdist.plugin -p pytest_asyncio.plugin -p no:cacheprovider \
  -q tests/model_executor/models/test_higgs_audio_v2_mrv2.py \
  tests/model_executor/models/higgs_audio_v2/test_delay_pattern_mask.py
```

## Scope

TTFA is first nonempty PCM bytes, near completion in this tested non-incremental deployment. E2E includes queueing and both model stages. Request RTF is not the inverse of aggregate throughput. C32-only evidence does not establish a scaling curve. No full-model service trace, full ASR/SIM or listening suite, full repository CI, other GPU platform, multi-GPU or graph-mode qualification is claimed.

The first cold baseline setup exceeded the old 240s readiness budget during JIT/autotune. It generated no audio; its log was retained, and fresh r2 correctness arms used 900s. All timed serving arms completed on their first planned run.

Real-service stage attribution and CPU/CUDA traces: [RTX profile](profile_rtx_20261125/README.md).
