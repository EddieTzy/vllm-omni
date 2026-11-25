# Reproduction

Use the same upstream model code as the archive base and a working vLLM-Omni GPU environment. Point `MODEL` at your prepared Fun-CosyVoice3-0.5B-2512 checkpoint and `REF_WAV` at a reference clip you have permission to use. Supply its transcript with `REF_TEXT`. Waveforms and the original reference are not distributed. Set the base_config in deploy.yaml to the absolute path to vllm_omni/deploy/cosyvoice3.yaml in your checkout if the installed resolver requires it. Use warm and measured waves in the order in README.md. Each client invocation submits four requests. Changes in hardware, reference or random output length can change the metrics.

```bash
vllm serve "$MODEL" --omni --served-model-name cv3 --deploy-config deploy.yaml --trust-remote-code
python client.py --url http://127.0.0.1:8000 --ref "$REF_WAV" --ref-text "$REF_TEXT" --seed 42
```

Export `COSYVOICE3_TRT=0` and `COSYVOICE3_BATCH_FLOW_DEBUG=1`. Set profiler_config on both stages (profiler: torch, torch_profiler_dir: your output directory, torch_profiler_record_shapes: true). After warmup, POST {} to /start_profile, run one C4 wave, then POST {} to /stop_profile. The measured snapshot additionally marked _forward_mel and _stream_hift_from_feat with torch.profiler.record_function.
