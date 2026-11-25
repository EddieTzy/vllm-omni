"""Run one instrumented C4 service using the existing unseeded TRT deployment."""

import base64
import concurrent.futures
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

work = Path(__file__).resolve().parent
root = work.parent
container_dir = "/data/" + work.name
baseline = root / "cv3-trt-c4/unseed"
site = "venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py"
runner = "vllm_omni/worker_v2/omni_generation_model_runner.py"
name = work.name
url = "http://127.0.0.1:18952"
docker = ["sudo", "-n", "docker"]
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
marker = '__import__("cv3_delivery_marks").mark'


def replace(source, target, edits):
    data = source.read_text()
    for old, new in edits:
        if data.count(old) != 1:
            raise ValueError(f"{source}: expected one occurrence of {old!r}, got {data.count(old)}")
        data = data.replace(old, new)
    compile(data, str(target), "exec")
    target.write_text(data)


def prepare():
    cmd = json.loads((baseline / "launch.json").read_text())
    cmd[cmd.index("--name") + 1] = name
    cmd[cmd.index("--port") + 1] = "18952"
    pp = next(i for i, s in enumerate(cmd) if s.startswith("PYTHONPATH="))
    cmd[pp] = cmd[pp].replace("PYTHONPATH=", "PYTHONPATH=" + container_dir + ":")
    cmd[pp + 1 : pp + 1] = ["-e", "CV3_DELIVERY_DIR=" + container_dir]

    pop = "        future, scheduler_output, exec_model_fut = batch_queue.pop()"
    replace(
        root / site,
        work / "core.py",
        [
            (
                pop,
                pop
                + f'\n        {marker}("queue_pop", scheduler_output.num_scheduled_tokens, '
                "depth=self.batch_queue_size, executor=type(self.model_executor).__name__)",
            )
        ],
    )
    ids = "self.model_runner_output.req_ids"
    record = "            self.copy_event.record(copy_stream)"
    sync = "        self.copy_event.synchronize()"
    payload = "        if self.finalize_output is not None:"
    replace(
        root / "vllm-higgs-mask-main" / runner,
        work / "runner.py",
        [
            (record, record + f'\n            {marker}("copy_enqueued", {ids})'),
            (
                sync,
                f'        {marker}("get_begin", {ids}, ready=self.copy_event.query())\n'
                + sync
                + f'\n        {marker}("copy_wait_end", {ids})',
            ),
            (payload, f'        {marker}("host_payload", {ids})\n' + payload),
        ],
    )
    at = "            if streaming_flow_items:"
    end = '            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})'
    group = '[item["req_id"] for item in streaming_flow_items]'
    replace(
        baseline / "cosyvoice3.py",
        work / "cosyvoice3.py",
        [
            (at, at + f'\n                {marker}("group_begin", {group})'),
            (
                end,
                "            if streaming_flow_items:\n"
                f'                {marker}("group_end", {group}, '
                'samples=[audios[item["index"]].numel() for item in streaming_flow_items])\n'
                + end,
            ),
        ],
    )
    send = "            await super().stream_response(send)"
    extract = "                audio_output = cast(dict, audio_output)\n                meta_collect"
    yielding = "            if arrival_s is None:\n                arrival_s = time.perf_counter() - stream_start_s"
    replace(
        baseline / "serving_speech.py",
        work / "serving_speech.py",
        [
            (
                send,
                '            await __import__("cv3_delivery_marks").instrument_send('
                "self, send, super().stream_response)",
            ),
            (extract, f'                {marker}("api_audio", [request_id])\n' + extract),
            (yielding, f'            {marker}("yield_audio", [request_id], size=len(audio_bytes))\n' + yielding),
        ],
    )
    for file in ("cosyvoice3.py", "serving_speech.py"):
        idx = next(i for i, value in enumerate(cmd) if value.startswith(str(baseline / file) + ":"))
        cmd[idx] = str(work / file) + ":" + cmd[idx].split(":", 1)[1]
    idx = cmd.index("--entrypoint")
    cmd[idx:idx] = ["-v", f"{work}/core.py:/data/{site}:ro", "-v", f"{work}/runner.py:/opt/vllm-omni/{runner}:ro"]
    return cmd


def request(pair):
    i, utterance = pair
    body = dict(
        model="cv3-profile",
        input=utterance,
        response_format="pcm",
        stream=True,
        stream_format="audio",
        ref_audio=reference,
        ref_text="and sometimes I pronounce the words wrong and people cannot get what I mean",
    )
    req = urllib.request.Request(
        url + "/v1/audio/speech", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    start = time.monotonic_ns()
    with http.open(req, timeout=100) as r:
        rid = r.headers["X-CV3-ID"]
        first_chunk = r.read1(65536)
        first = time.monotonic_ns()
        total = len(first_chunk) + len(r.read())
    if not rid or not first_chunk:
        raise ValueError("Missing request ID or empty first body")
    return dict(i=i, id=rid, start=start, first=first, end=time.monotonic_ns(), size=total)


if __name__ == "__main__":
    cmd = prepare()
    corpus = json.loads((root / "cv3-profile-20261125/corpus.json").read_text())
    assert len(corpus) == 4
    wav = Path(os.environ.get("CV3_REF_WAV", root.parent / "tts-results/references/F00A_eng_continuous.wav"))
    reference = "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode()
    users = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    memory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    if users.strip() or int(memory.strip()) > 100:
        raise SystemExit("GPU busy; skipping")
    subprocess.run(cmd, check=True)
    try:
        deadline = time.monotonic() + 600
        while True:
            try:
                with http.open(url + "/health", timeout=2):
                    break
            except OSError:
                state = subprocess.check_output(docker + ["inspect", "-f", "{{.State.Running}}", name], text=True)
                if state.strip() != "true" or time.monotonic() > deadline:
                    raise RuntimeError("Server failed readiness")
                time.sleep(2)
        print("Ready", flush=True)
        for wave in ("warm0", "warm1", "measure0", "measure1"):
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                rows = list(pool.map(request, enumerate(corpus)))
            (work / (wave + ".json")).write_text(json.dumps(rows, indent=2))
            print(wave, [r["size"] for r in rows], flush=True)
    finally:
        subprocess.run(docker + ["stop", "-t", "20", name], check=True)
        with (work / "server.log").open("w") as f:
            subprocess.run(docker + ["logs", name], stdout=f, stderr=subprocess.STDOUT)
