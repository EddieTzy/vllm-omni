"""Run A1 B1 B2 A2 against the existing unseeded, single-GPU TRT C4 launch."""

import base64
import concurrent.futures
import http.client
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path

work = Path(__file__).resolve().parent
root = work.parent
baseline = root / "cv3-trt-c4/unseed"
port = 18954
docker = ["sudo", "-n", "docker"]
site = "venv-vllm/lib/python3.12/site-packages/vllm/v1/engine/core.py"


def overlay(original, output, anchor, replacement):
    content = original.read_text()
    assert content.count(anchor) == 1, anchor
    updated = content.replace(anchor, replacement)
    compile(updated, str(output), "exec")
    output.write_text(updated)


def launch(arm, directory, name):
    cmd = json.loads((baseline / "launch.json").read_text())
    cmd[cmd.index("--name") + 1] = name
    cmd[cmd.index("--port") + 1] = str(port)
    switch = "true" if arm.startswith("A") else "false"
    cfg = (baseline / "deploy.yaml").read_text().rstrip() + f"\n- stage_id: 1\n  async_scheduling: {switch}\n"
    (directory / "deploy.yaml").write_text(cfg)
    cmd[cmd.index("--deploy-config") + 1] = f"/data/{work.name}/{arm}/deploy.yaml"
    anchor = "        self.async_scheduling = vllm_config.scheduler_config.async_scheduling"
    overlay(
        root / site,
        directory / "core.py",
        anchor,
        anchor + '\n        print("CV3AB_CONFIG", type(self.scheduler).__name__, self.async_scheduling, '
        "self.batch_queue_size, self.step_fn.__name__, flush=True)",
    )
    anchor = '            return OmniOutput(text_hidden_states=None, multimodal_outputs={"audio": audios, "sr": srs})'
    overlay(
        baseline / "cosyvoice3.py",
        directory / "model.py",
        anchor,
        "            if streaming_flow_items:\n"
        '                print("CV3AB_GROUP", __import__("time").monotonic_ns(), '
        '[(item["req_id"], audios[item["index"]].numel()) for item in streaming_flow_items], flush=True)\n' + anchor,
    )
    i = cmd.index("--entrypoint")
    cmd[i:i] = ["-v", f"{directory}/core.py:/data/{site}:ro"]
    i = next(i for i, val in enumerate(cmd) if val.startswith(str(baseline / "cosyvoice3.py") + ":"))
    cmd[i] = str(directory / "model.py") + ":" + cmd[i].split(":", 1)[1]
    return cmd


def request(item):
    i, payload = item
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=100)
    try:
        start = time.monotonic_ns()
        conn.request("POST", "/v1/audio/speech", body=payload, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read(500)!r}")
        assert response.chunked
        rid = response.getheader("X-CV3-ID")
        sample_rate = response.getheader("X-Audio-Sample-Rate")
        chunks = []
        # Uvicorn emits a chunk per ASGI body. Preserve HTTP boundaries across socket reads.
        while (size := response._get_chunk_left()) is not None:
            if size <= 0 or len(response.read(1)) != 1:
                raise ValueError("Invalid body chunk")
            first = time.monotonic_ns()
            if size > 1 and len(response.read(size - 1)) != size - 1:
                raise ValueError("Incomplete body chunk")
            chunks.append([first, size])
        end = time.monotonic_ns()
        assert rid and chunks and all(n % 2 == 0 for _, n in chunks)
        return dict(i=i, id=rid, start=start, end=end, chunks=chunks, sample_rate=sample_rate)
    finally:
        conn.close()


def run(arm, payloads):
    directory = work / arm
    directory.mkdir()
    name = f"{work.name}-{arm.lower()}"
    cmd = launch(arm, directory, name)
    users = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True)
    used = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True
    )
    if users.strip() or int(used.strip()) > 100:
        raise RuntimeError("GPU busy")
    subprocess.run(cmd, check=True)
    try:
        deadline = time.monotonic() + 480
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        while True:
            try:
                with opener.open(f"http://127.0.0.1:{port}/health", timeout=2):
                    break
            except OSError:
                state = subprocess.check_output(
                    docker + ["inspect", "-f", "{{.State.Running}}", name], text=True
                ).strip()
                if state != "true" or time.monotonic() > deadline:
                    raise RuntimeError("Service failed readiness")
                time.sleep(2)
        for j in range(8):
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                rows = list(pool.map(request, enumerate(payloads)))
            (directory / f"wave{j}.json").write_text(json.dumps(rows))
            print(arm, j, [round((r["chunks"][0][0] - r["start"]) / 1e6) for r in rows], flush=True)
    finally:
        subprocess.run(docker + ["stop", "-t", "15", name], check=True)
        with (directory / "server.log").open("w") as stream:
            subprocess.run(docker + ["logs", name], stdout=stream, stderr=subprocess.STDOUT)
    config = (directory / "server.log").read_text()
    modes = re.findall(r"CV3AB_CONFIG ([^\n]+)", config)
    print(arm, modes, flush=True)
    expected = "True 2 step_with_batch_queue" if arm.startswith("A") else "False 1 step"
    if not any("Generation" in mode and expected == mode.split(" ", 1)[1].strip() for mode in modes):
        raise RuntimeError("Unexpected generation-stage mode")


if __name__ == "__main__":
    wav = Path(os.environ.get("CV3_REF_WAV", root.parent / "tts-results/references/F00A_eng_continuous.wav"))
    reference = "data:audio/wav;base64," + base64.b64encode(wav.read_bytes()).decode()
    corpus = json.loads((root / "cv3-profile-20261125/corpus.json").read_text())
    assert len(corpus) == 4
    payloads = [
        json.dumps(
            dict(
                model="cv3-profile",
                input=s,
                stream=True,
                stream_format="audio",
                response_format="pcm",
                ref_audio=reference,
                ref_text="and sometimes I pronounce the words wrong and people cannot get what I mean",
            )
        ).encode()
        for s in corpus
    ]
    for arm in ("A1", "B1", "B2", "A2"):
        run(arm, payloads)
