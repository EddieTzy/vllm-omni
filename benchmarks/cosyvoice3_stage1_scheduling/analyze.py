"""Build metrics and per-start Perfetto traces from four timestamp archives."""

import ast
import gzip
import itertools
import json
import math
import re
import statistics
import sys
from pathlib import Path

base = Path(__file__).resolve().parent
arms = ("A1", "B1", "B2", "A2")


def percentile(values, p):
    ordered = sorted(values)
    index = (len(ordered) - 1) * p
    fraction = index % 1
    return ordered[math.floor(index)] * (1 - fraction) + ordered[math.ceil(index)] * fraction


def import_capture(source):
    for arm in arms:
        folder = source / arm
        marked = folder / "marks.log"
        log = (marked if marked.exists() else folder / "server.log").read_text()
        modes = re.findall(r"CV3AB_CONFIG (\w+) (True|False) (\d+) (\w+)", log)
        assert len(modes) == 2
        waves = [json.loads((folder / f"wave{i}.json").read_text()) for i in range(2, 8)]
        ids = {r["id"] for wave in waves for r in wave}
        assert len(ids) == 24

        def rid(s):
            public, internal = s.rsplit("-", 1)
            assert re.fullmatch("[0-9a-f]+", internal)
            return public

        events = []
        for line in log.splitlines():
            begin = re.search(r"CV3BEGIN (speech-[\w-]+) (\d+) (\d+) (\d+)", line)
            if begin and rid(begin[1]) in ids:
                events.append(dict(name="begin", ns=int(begin[4]), ids=[rid(begin[1])], offset=int(begin[2])))
            end = re.search(r"CV3AB_GROUP (\d+) (\[.*\])", line)
            if end:
                samples = [(rid(i), n) for i, n in ast.literal_eval(end[2])]
                if any(i in ids for i, _ in samples):
                    assert all(i in ids for i, _ in samples)
                    events.append(
                        dict(
                            name="group_end",
                            ns=int(end[1]),
                            ids=[i for i, _ in samples],
                            samples=[n for _, n in samples],
                        )
                    )
        target = base / f"{arm}.json.gz"
        target.write_bytes(gzip.compress(json.dumps(dict(modes=modes, waves=waves, events=events)).encode(), mtime=0))


def analyze_arm(arm):
    data = json.loads(gzip.decompress((base / f"{arm}.json.gz").read_bytes()))
    earliest = min(r["start"] for wave in data["waves"] for r in wave)
    rows = []
    trace = []

    def span(lane, name, start, end, category, **args):
        trace.append(
            dict(
                ph="X",
                cat=category,
                pid=1,
                tid=lane,
                name=name,
                ts=(start - earliest) / 1000,
                dur=(end - start) / 1000,
                args=args,
            )
        )

    def lane_name(lane, name):
        trace.append(dict(ph="M", name="thread_name", pid=1, tid=lane, args=dict(name=name)))

    lane_name(8, "Stage 1 model")
    first_in_group = None
    group_ids = []
    for event in data["events"]:
        if event["name"] == "begin":
            if first_in_group is None:
                first_in_group = event["ns"]
            group_ids.extend(event["ids"])
        else:
            if first_in_group is None:
                # A concurrent process interleaved its stdout with one B2 begin line.
                # Keep the measured endpoint; never infer the missing span start.
                assert not group_ids
                trace.append(
                    dict(
                        ph="i",
                        s="t",
                        cat="model",
                        pid=1,
                        tid=8,
                        name="audio group end (missing start)",
                        ts=(event["ns"] - earliest) / 1000,
                        args=dict(ids=event["ids"]),
                    )
                )
                continue
            assert group_ids == event["ids"]
            span(8, f"audio group ({len(group_ids)})", first_in_group, event["ns"], "model", ids=group_ids)
            first_in_group = None
            group_ids = []
    assert not group_ids
    ends = [e for e in data["events"] if e["name"] == "group_end"]
    wave_times = []
    for j, wave in enumerate(data["waves"]):
        wave_times.append((max(r["end"] for r in wave) - min(r["start"] for r in wave)) / 1e9)
        for r in wave:
            chunks = r["chunks"]
            first = chunks[0][0]
            total_bytes = sum(s for _, s in chunks)
            assert int(r["sample_rate"]) == 24000 and total_bytes > 0
            audio = total_bytes / (2 * int(r["sample_rate"]))
            cost = (r["end"] - r["start"]) / 1e9
            rid = r["id"]
            group = next(e for e in ends if any(i == rid and n > 0 for i, n in zip(e["ids"], e["samples"])))
            assert r["start"] < group["ns"] < first < r["end"]
            extra = sum(group["ns"] < e["ns"] <= first for e in ends)
            gaps = [(b[0] - a[0]) / 1e6 for a, b in itertools.pairwise(chunks)]
            # Immediate playback on loopback, with each body chunk available at its first byte.
            # The minimum margin includes all audio received before the arriving chunk.
            buffered_ms = 0.0
            margins = []
            for (_, size), (arrival, _) in itertools.pairwise(chunks):
                buffered_ms += size / 48000 * 1000
                margins.append(buffered_ms - (arrival - first) / 1e6)
            rows.append(
                dict(
                    wave=j,
                    audio=audio,
                    cost=cost,
                    rtf=cost / audio,
                    ttfa=(first - r["start"]) / 1e6,
                    handoff=(first - group["ns"]) / 1e6,
                    other_groups=extra,
                    gaps=gaps,
                    min_margin_ms=min(margins, default=None),
                )
            )
            lane = r["i"] + 1
            if j == 0:
                lane_name(lane, f"client {r['i']}")
            span(lane, "request → first group return", r["start"], group["ns"], "delivery", id=rid)
            span(lane, "group return → first PCM", group["ns"], first, "delivery", id=rid)
            for k, (a, b) in enumerate(itertools.pairwise(chunks)):
                span(lane, "PCM inter-chunk gap", a[0], b[0], "http", id=rid, chunk=k)
    (base / f"{arm}.perfetto.json.gz").write_bytes(gzip.compress(json.dumps(dict(traceEvents=trace)).encode(), mtime=0))
    return rows, wave_times


def metrics(rows, times):
    gaps = [gap for r in rows for gap in r["gaps"]]

    first_gaps = [r["gaps"][0] for r in rows if r["gaps"]]
    later_gaps = [g for r in rows for g in r["gaps"][1:]]
    margins = [r["min_margin_ms"] for r in rows if r["min_margin_ms"] is not None]

    def q(field, p):
        return percentile([r[field] for r in rows], p)

    return dict(
        n=len(rows),
        audio_s=sum(r["audio"] for r in rows),
        wave_wall_s=sum(times),
        audio_s_per_s=sum(r["audio"] for r in rows) / sum(times),
        mean_rtf=statistics.mean(r["rtf"] for r in rows),
        mean_cost_s=statistics.mean(r["cost"] for r in rows),
        ttfa_p50_ms=q("ttfa", 0.5),
        ttfa_p95_ms=q("ttfa", 0.95),
        handoff_p50_ms=q("handoff", 0.5),
        handoff_p95_ms=q("handoff", 0.95),
        with_other_group=sum(r["other_groups"] > 0 for r in rows),
        gaps=len(gaps),
        first_gap_p95_ms=percentile(first_gaps, 0.95),
        later_gap_p95_ms=percentile(later_gaps, 0.95),
        projected_starved_requests=sum(m < 0 for m in margins),
        min_playback_margin_ms=min(margins),
        gap_p50_ms=percentile(gaps, 0.5),
        gap_p95_ms=percentile(gaps, 0.95),
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import_capture(Path(sys.argv[1]))
    by_arm = {arm: analyze_arm(arm) for arm in arms}
    result = {arm: metrics(*data) for arm, data in by_arm.items()}
    for label in ("A", "B"):
        rows = [r for arm, (sample, _) in by_arm.items() if arm.startswith(label) for r in sample]
        times = [t for arm, (_, sample) in by_arm.items() if arm.startswith(label) for t in sample]
        result[label] = metrics(rows, times)
    (base / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({label: result[label] for label in ("A", "B")}, indent=2))
