# Reproduction

Use the same upstream model code as the archive base and a working vLLM-Omni GPU environment. Point `MODEL` at your prepared Fun-CosyVoice3-0.5B-2512 checkpoint and `REF_WAV` at a reference clip you have permission to use. Supply its transcript with `REF_TEXT`. Waveforms and the original reference are not distributed. Set the base_config in deploy.yaml to the absolute path to vllm_omni/deploy/cosyvoice3.yaml in your checkout if the installed resolver requires it. Use warm and measured waves in the order in README.md. Each client invocation submits four requests. Changes in hardware, reference or random output length can change the metrics.

```bash
vllm serve "$MODEL" --omni --served-model-name cv3 --deploy-config deploy.yaml --trust-remote-code
python client.py --url http://127.0.0.1:8000 --ref "$REF_WAV" --ref-text "$REF_TEXT" 
```

Enable `COSYVOICE3_TRT=1` with TRT 10.14 and set COSYVOICE3_ESTIMATOR_ONNX to the bundled flow.decoder.estimator.fp32.onnx. Disable padded DiT compilation. Verify the server logs "using TensorRT flow-decoder estimator (code2wav)". Repeat once with --seed 42 and once without --seed. One single-request smoke check and two warm C4 waves precede four measured waves. For the separate diagnostic boots enable torch profiler on both stages, enable COSYVOICE3_BATCH_FLOW_DEBUG, and capture one C4 wave after warmup.
