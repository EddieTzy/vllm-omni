# Reproduction

Use the same upstream model code as the archive base and a working vLLM-Omni GPU environment. Point `MODEL` at your prepared Fun-CosyVoice3-0.5B-2512 checkpoint and `REF_WAV` at a reference clip you have permission to use. Supply its transcript with `REF_TEXT`. Waveforms and the original reference are not distributed. Set the base_config in deploy.yaml to the absolute path to vllm_omni/deploy/cosyvoice3.yaml in your checkout if the installed resolver requires it. Use warm and measured waves in the order in README.md. Each client invocation submits four requests. Changes in hardware, reference or random output length can change the metrics.

```bash
vllm serve "$MODEL" --omni --served-model-name cv3 --deploy-config deploy.yaml --trust-remote-code
python client.py --url http://127.0.0.1:8000 --ref "$REF_WAV" --ref-text "$REF_TEXT" 
```

Compare the archive base TRT route with the candidate runtime on this branch. Enable TRT, use the same engine, leave packed and batch Flow disabled; omit request seeds. See the baseline TRT archive for engine activation and profiling. For the component screen set CV3_EXPERIMENT_DIR to a directory containing plans/flow_estimator_*.plan. Run python contexts.py in the GPU runtime. multi.py contains the measured dispatch method and verify.py the optional paired-check hook. Production rollout is not recommended from this experiment.
