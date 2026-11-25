# Reproduction

Use the same upstream model code as the archive base and a working vLLM-Omni GPU environment. Point `MODEL` at your prepared Fun-CosyVoice3-0.5B-2512 checkpoint and `REF_WAV` at a reference clip you have permission to use. Supply its transcript with `REF_TEXT`. Waveforms and the original reference are not distributed. Set the base_config in deploy.yaml to the absolute path to vllm_omni/deploy/cosyvoice3.yaml in your checkout if the installed resolver requires it. Use warm and measured waves in the order in README.md. Each client invocation submits four requests. Changes in hardware, reference or random output length can change the metrics.

```bash
vllm serve "$MODEL" --omni --served-model-name cv3 --deploy-config deploy.yaml --trust-remote-code
python client.py --url http://127.0.0.1:8000 --ref "$REF_WAV" --ref-text "$REF_TEXT" --seed 42
```

Export `COSYVOICE3_TRT=0`. In separate boots set Stage 1 hf_overrides.cosyvoice3_compile_dit false / true using --stage-overrides. Use three warmup waves, four measured waves and fresh A/B/B2/A2 processes. Shape probes in traces/ isolate two real estimator inputs. Quality comparisons in Bquality/ use the same input token/cache/RNG state per chunk.
