import argparse
import json
import os
from pathlib import Path
import subprocess as sp
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument("label")
p.add_argument("--profile", action="store_true")
p.add_argument("--candidate", action="store_true")
args = p.parse_args()
root = Path("/home/tzy/zyb-fish-pr")
work = root / "fish-prefill-mrv2-95988"
d = work / args.label
d.mkdir()
mapped = "/data/fish-prefill-mrv2-95988/"
name = "s2-mrv2-prefill-" + args.label.lower()
src = work / ("candidate" if args.candidate else "source")
assert not sp.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(), "GPU in use"
port = 18939
img = "tts-runtime:20261117-pinned"
run = ["sudo","-n","docker","run","--name",name,"-d","--gpus","all","--network","host","--ipc","private","--shm-size","8g","--user",f"{os.getuid()}:{os.getgid()}","-v",f"{root}:/data","-v",f"{src}:/opt/vllm-omni:ro"]
codec = "vllm_omni/model_executor/models/fish_speech/fish_speech_slow_ar.py"
if args.profile:
    instrumented = d/"profiled_codec.py"
    source = (src/codec).read_text()
    start_text = "                " + ("total_prompt_len = int(prompt_embeds.shape[0])" if args.candidate else "prompt_embeds_buf = prompt_embeds.detach()")
    i = source.index(start_text)
    j = source.index("                zeros = torch.zeros(", i)
    block = source[i:j]
    annotated = "                from fish_probe import trace_transfer, record_output\n                with trace_transfer(prompt_embeds, span_len, info_dict):\n" + "".join("    "+line if line.strip() else line for line in block.splitlines(keepends=True)) + "                record_output(prompt_embeds)\n"
    instrumented.write_text(source[:i] + annotated + source[j:])
    run += ["-v",f"{instrumented}:/opt/vllm-omni/{codec}:ro","-e","HIGGS3_PROFILE=1"]
env = {"HOME":"/data","HF_HOME":"/data/cache/huggingface","HF_HUB_OFFLINE":"1","PYTHONPATH":"/opt/vllm-omni:"+mapped+"diagnostics","VLLM_WORKER_MULTIPROC_METHOD":"spawn","VLLM_USE_DEEP_GEMM":"0","VLLM_MOE_USE_DEEP_GEMM":"0","OMP_NUM_THREADS":"4","TRITON_CACHE_DIR":"/data/cache/triton-vllm","FISH_ARTIFACTS":mapped+args.label}
for k,v in env.items(): run += ["-e",k+"="+v]
run += ["--entrypoint","/data/venv-vllm/bin/vllm",img,"serve","/data/models/s2-pro","--omni","--deploy-config","/opt/vllm-omni/vllm_omni/deploy/fish_qwen3_omni_high_concurrency_single_gpu.yaml","--host","127.0.0.1","--port",str(port),"--trust-remote-code","--disable-log-stats"]
sp.run(run,check=True)
docker=["sudo","-n","docker"]
try:
    deadline=time.monotonic()+1200
    while time.monotonic()<deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",timeout=3) as res:
                if res.status==200: break
        except OSError: pass
        state=sp.check_output(docker+["inspect","-f","{{.State.Running}}",name],text=True).strip()
        if state!="true": raise RuntimeError("server exited")
        time.sleep(3)
    else: raise TimeoutError("readiness")
    client=docker+["exec",name,"/data/venv-vllm/bin/python",mapped+"client.py","--port",str(port),"--label",name,"--concurrency","32","--count","64","--corpus","/data/higgs-routing/public-corpus.json","--omit-request-seed","--request-timeout","360"]
    if args.profile: (d/"armed").touch()
    sp.run(client,check=True)
finally:
    with (d/"server.log").open("w") as f: sp.run(docker+["logs",name],stdout=f,stderr=sp.STDOUT)
    sp.run(docker+["stop","-t","7",name])
