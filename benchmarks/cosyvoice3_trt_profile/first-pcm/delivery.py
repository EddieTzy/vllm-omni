"""Align HTTP first reads with the stage-1 output copy and the next forward."""

import gzip
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent


def analyze(label, capture):
    folder = root / "inputs" / label
    requests = {r["id"]: r for r in json.loads((folder / "measure0.json").read_text())["rows"]}
    markers = (folder / "time-markers.txt").read_text()
    begins = sorted(
        (int(ts), rid, int(offset)) for rid, offset, ts in re.findall(r"CV3BEGIN (\S+) (\d+) \d+ (\d+)", markers)
    )
    flows = [(int(a), int(b)) for a, b in re.findall(r"CV3FLOW (\d+) (\d+) \d+", markers)]
    blocks = []
    for i, (ts, rid, offset) in enumerate(begins):
        request = next((k for k in requests if rid.startswith(k + "-") or k == rid), None)
        if not request:
            continue
        upper = begins[i + 1][0] if i + 1 < len(begins) else float("inf")
        matched = [(a, b) for a, b in flows if ts <= a <= b < upper]
        assert len(matched) == 1
        blocks.append((request, offset, *matched[0]))
    with gzip.open(root.parent / "traces" / capture) as f:
        events = json.load(f)["traceEvents"]
    complete = sorted((e for e in events if e.get("ph") == "X"), key=lambda e: e["ts"])
    scopes = [e for e in complete if e.get("cat") == "user_annotation"]
    mel = [e for e in scopes if e["name"] == "cv3_flow_mel"]
    groups = [e for e in scopes if e["name"].startswith("execute_context_")]
    end = lambda e: e["ts"] + e["dur"]
    # The CV3FLOW start/end clock reads happen inside the matching _forward_mel scope.
    assert len(mel) == len(blocks)
    low = max(b / 1000 - end(e) for (_, _, a, b), e in zip(blocks, mel))
    high = min(a / 1000 - e["ts"] for (_, _, a, b), e in zip(blocks, mel))
    assert low <= high and high - low < 1000, (low, high)
    clock_offset = (low + high) / 2
    copies = [
        e
        for e in complete
        if e.get("cat") == "gpu_memcpy" and "DtoH" in e["name"] and e.get("args", {}).get("stream") == 13
    ]
    assert len(copies) == len(blocks)
    rows = []
    for (rid, offset, a, b), f, copy in zip(blocks, mel, copies):
        if offset != 0:
            continue
        assert copy["args"]["bytes"] == 69120
        ix = next(i for i, g in enumerate(groups) if g["ts"] <= f["ts"] and end(f) <= end(g))
        own, following = groups[ix : ix + 2]
        r = requests[rid]
        pcm = (r["last"] + (r["first"] - r["e2e"]) * 1e9) / 1000 - clock_offset
        assert end(own) <= copy["ts"] and end(copy) - end(own) < 2000
        if label == "unseedtrace":
            assert end(copy) < following["ts"]
        rows.append(
            dict(
                request=rid[-6:],
                group=own["name"],
                next_group=following["name"],
                group_end_to_copy_end_ms=(end(copy) - end(own)) / 1000,
                copy_us=copy["dur"],
                copy_end_to_pcm_ms=(pcm - end(copy)) / 1000,
                next_group_duration_ms=following["dur"] / 1000,
                next_group_end_to_pcm_ms=(pcm - end(following)) / 1000,
            )
        )
    assert len(rows) == len(requests)
    return dict(clock_offset_bracket_us=round(high - low, 2), requests=rows)


result = {
    name: analyze(name, f"{prefix}-stage1.json.gz")
    for name, prefix in [("unseedtrace", "unseeded"), ("trace", "seeded")]
}


def without_profiler(arm):
    folder = root / "inputs" / arm
    starts = [
        (rid, int(ts))
        for rid, off, ts in re.findall(r"CV3BEGIN (\S+) (\d+) \d+ (\d+)", (folder / "time-markers.txt").read_text())
        if int(off) > 0
    ]
    waves = []
    for f in sorted(folder.glob("measure*.json")):
        rows = json.loads(f.read_text())["rows"]
        ids = [r["id"] for r in rows]
        next_start = min(ts for rid, ts in starts if any(rid.startswith(k + "-") for k in ids))
        deltas = sorted(((r["last"] - next_start) / 1e6 + (r["first"] - r["e2e"]) * 1000) for r in rows)
        waves.append(deltas)
    return waves


result["client_first_read_relative_to_first_followup_start_ms"] = {
    arm: without_profiler(arm) for arm in ["unseed", "unseed2"]
}
(root / "delivery.json").write_text(json.dumps(result, indent=2) + "\n")
