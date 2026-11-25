# Fish S2-Pro streaming voice clone on Model Runner V2

[Serving measurements](RESULTS.md) · [per-request results](request-metrics.json) · [copy counts and timings](profile.json) · [Perfetto traces](evidence.tar.gz) · [source, reference and archive hashes](hashes.json)

## What changed

Fish SlowAR builds a dense prompt embedding on the GPU. The baseline moves it to CPU memory, pins a copy and sends the scheduled slice back to the GPU. When the first prefill consumes the entire tensor, the cache is immediately discarded. The candidate uses the original tensor in this case. When only part of the prompt is scheduled, both implementations retain the pinned host cache and advance the offset across chunks. Sampling, DAC and audio streaming use the same code on both sides.

## Compared revisions and inputs

- The tested upstream `main` revision `95513b5322fc78fe3b47bd02b9244d30ba6f0e20` failed during voice-clone warmup with the Fish deferred MTP path on Model Runner V2. An earlier launch also failed before serving because a local driver named `queue.py` shadowed the Python standard library. Neither launch produced timing data.
- Both sides of the runnable comparison use Fish SlowAR migration [#9806](https://github.com/vllm-project/vllm-omni/pull/9806) at `95988e77ed8862edf93f8eeaf906e984d9a5e424`. Candidate `b4336014bb0f53b92b48b8d71f6681db3bef7cfc` adds the full-prefill change. These results apply to that migration revision.
- Single NVIDIA RTX PRO 6000 Blackwell Server Edition, 96 GB, driver 580.173.02. vLLM 0.34.0 and PyTorch 2.14.0+cu130. Local Docker image `tts-runtime:20261117-pinned`, ID `sha256:170f88230e131cfd71ffe25cc82438b773371f6076bd8b9dcb1a9cae92af22a4`.
- Fish S2-Pro model and codec from `fishaudio/s2-pro`. The cached config, codec and both weight shards identify HF revision `1de9996b6be38b745688de084d87a5633f714e4e`. SlowAR and DAC share GPU 0 using `vllm_omni/deploy/fish_qwen3_omni_high_concurrency_single_gpu.yaml`. Chunked prefill and prefix caching are disabled in this configuration.
- Each request supplied the same inline public WAV (`tests/assets/qwen3_tts/clone_2.wav`) and its transcript from the repository. The client cycled through 32 English and 32 Chinese targets from `zhaochenyang20/seed-tts-eval` revision `8f5e1aa2a35d42f42e940074c1983358b9491f89`. Exact targets and transcript are in `reproduction/`. Sampling used deployment defaults without a per-request seed override.

## Reading the measurements

Four fresh server processes ran in A/B/B/A order. Each received 32 warmup requests and two consecutive waves of 32 timed requests. Throughput is total returned mono 44.1 kHz PCM16 audio seconds divided by wall seconds. TTFA ends at the first nonempty response block. Request elapsed time ends at stream completion. RTF is the arithmetic mean of each request’s elapsed time divided by its own audio duration. Summary columns average the two process-level readings per arm. Streaming generation length can vary between runs. Timing contains loopback HTTP and client overhead, including saving PCM locally before the next wave. No ASR or perceptual quality score was computed.

Two separate diagnostic processes used the same warmup and workload. Their throughput is excluded from the A/B averages. Each JSON trace samples the region between building the embedding and creating the audio-code state. Open the archive’s `diag-a/` and `diag-b/` JSON files in [Perfetto](https://ui.perfetto.dev/) and search for `fish.first_prefill_embedding_transfer`. The probe hashes the prefill output after the marked region. `controls.json` records the CUDA fixture checks for whole and chunked prefill with BF16 and FP32 inputs, including cache, offsets, padded IDs and audio-code tensors. These are separate from the full service runs.

## Reproducing the run

The scripts use the benchmark host’s directory layout. On another host, map `/home/tzy/zyb-fish-pr` to a writable experiment root and mount it at `/data` inside the container. Provide the two pinned source trees and the same model, codec, runtime and deployment YAML. Set up `fish-prefill-mrv2-95988/{source,candidate,diagnostics}` below the experiment root. Copy `reproduction/*.py` into `fish-prefill-mrv2-95988/`, and copy `fish_probe.py` into its `diagnostics/` subdirectory. Place `public-corpus.json` at `higgs-routing/public-corpus.json`. The client writes its PCM and JSON under `results/`. Use an empty output directory for each new run.

From `fish-prefill-mrv2-95988/`, run `python3 trial.py base-a`, then `python3 run_trials.py`. The latter launches the remaining timed and profiled runs, followed by the CUDA controls and upstream tests. Use `python3 report_remote.py base-a` to aggregate the six runs. `manifest.json` records the measured commits and process order.
