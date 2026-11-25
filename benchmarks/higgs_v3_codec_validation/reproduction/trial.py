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
work = root / "higgs3-codec-95513"
d = work / args.label
d.mkdir()
mapped = "/data/higgs3-codec-95513/"
name = "higgs3-codec-" + args.label.lower()
src = work / ("candidate" if args.candidate else "source")
assert not sp.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip(), "GPU in use"
port = 18938
img = "tts-runtime:20261117-pinned"
run = ["sudo","-n","docker","run","--name",name,"-d","--gpus","all","--network","host","--ipc","private","--shm-size","8g","--user",f"{os.getuid()}:{os.getgid()}","-v",f"{root}:/data","-v",f"{src}:/opt/vllm-omni:ro"]
codec = "vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py"
if args.profile:
    instrumented = d/"profiled_codec.py"
    instrumented.write_text((src/codec).read_text()+'\nfrom profile_codec import instrument\ninstrument(HiggsAudioV3Code2Wav)\n')
    run += ["-v",f"{instrumented}:/opt/vllm-omni/{codec}:ro","-e","HIGGS3_PROFILE=1"]
env = {"HOME":"/data","HF_HOME":"/data/cache/huggingface","HF_HUB_OFFLINE":"1","PYTHONPATH":"/opt/vllm-omni:"+mapped.rstrip("/"),"VLLM_WORKER_MULTIPROC_METHOD":"spawn","VLLM_USE_DEEP_GEMM":"0","VLLM_MOE_USE_DEEP_GEMM":"0","OMP_NUM_THREADS":"4","TRITON_CACHE_DIR":"/data/cache/triton-vllm","HIGGS3_ARTIFACTS":mapped+args.label}
for k,v in env.items(): run += ["-e",k+"="+v]
run += ["--entrypoint","/data/venv-vllm/bin/vllm",img,"serve","/data/models/higgs-v3-tts-4b","--omni","--deploy-config","/opt/vllm-omni/vllm_omni/deploy/higgs_multimodal_qwen3_high_throughput.yaml","--host","127.0.0.1","--port",str(port),"--trust-remote-code","--disable-log-stats"]
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
