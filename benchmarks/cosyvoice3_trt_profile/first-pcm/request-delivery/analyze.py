"""Join native engine IDs to the public speech ID and construct a Perfetto timeline."""

import gzip
import json
import re
import sys
from pathlib import Path

base = Path(__file__).resolve().parent


def public_id(engine_id):
    match = re.fullmatch(r"(speech-[0-9a-f]+)(?:-[0-9a-f]+)?", engine_id or "")
    return match[1] if match else engine_id


def read_marks():
    with gzip.open(base / "marks.jsonl.gz", "rt") as f:
        return sorted((json.loads(line) for line in f), key=lambda x: x["ns"])


def summarize(marks, rows):
    output = []
    spans = []
    for row in rows:
        rid = row["id"]
        own = [e for e in marks if rid in map(public_id, e["ids"]) and row["start"] <= e["ns"] <= row["end"]]
        group = next(
            e
            for e in own
            if e["name"] == "group_end" and any(public_id(i) == rid and n > 0 for i, n in zip(e["ids"], e["samples"]))
        )
        pid = group["pid"]
        cursor = group["ns"]
        found = {"group_end": cursor}
        stages = (
            "copy_enqueued",
            "queue_pop",
            "get_begin",
            "copy_wait_end",
            "host_payload",
            "api_audio",
            "yield_audio",
            "asgi_send",
        )
        for name in stages:
            event = next(
                e
                for e in own
                if e["name"] == name
                and e["ns"] >= cursor
                and (name in ("api_audio", "yield_audio", "asgi_send") or e["pid"] == pid)
            )
            cursor = found[name] = event["ns"]
            if name == "get_begin":
                ready = event["ready"]
            if name == "queue_pop":
                depth, executor = event["depth"], event["executor"]
        found["first_read"] = row["first"]
        assert found["asgi_send"] <= row["first"]
        # FIFO futures: find the invocation owning this copy, even when consecutive batches share IDs.
        copies = [e for e in marks if e["pid"] == pid and e["name"] == "copy_enqueued"]
        gets = [e for e in marks if e["pid"] == pid and e["name"] == "get_begin"]
        ordinal = next(i for i, e in enumerate(copies) if e["ns"] == found["copy_enqueued"])
        assert gets[ordinal]["ns"] == found["get_begin"] and gets[ordinal]["ids"] == copies[ordinal]["ids"]
        siblings = [
            e
            for e in marks
            if e["pid"] == pid
            and found["copy_enqueued"] < e["ns"] < found["queue_pop"]
            and e["name"] in ("group_begin", "group_end")
        ]
        assert len(siblings) % 2 == 0
        next_ns = 0
        for begin, end in zip(siblings[::2], siblings[1::2]):
            assert begin["name"] == "group_begin" and end["name"] == "group_end" and begin["ids"] == end["ids"]
            next_ns += end["ns"] - begin["ns"]
        groups = len(siblings) // 2
        spans.append((row["wave"], rid, {"request_start": row["start"], **found}))

        def ms(a, b):
            return round((found[b] - found[a]) / 1e6, 3)

        output.append(
            dict(
                wave=row["wave"],
                i=row["i"],
                id=rid,
                executor=executor,
                queue_capacity=depth,
                ready_at_get=ready,
                ttfa_ms=round((row["first"] - row["start"]) / 1e6, 3),
                group_to_enqueue_ms=ms("group_end", "copy_enqueued"),
                enqueue_to_pop_ms=ms("copy_enqueued", "queue_pop"),
                intervening_groups=groups,
                intervening_model_ms=round(next_ns / 1e6, 3),
                pop_to_host_payload_ms=ms("queue_pop", "host_payload"),
                host_payload_to_api_ms=ms("host_payload", "api_audio"),
                api_to_send_ms=ms("api_audio", "asgi_send"),
                send_to_read_ms=ms("asgi_send", "first_read"),
                group_to_first_read_ms=ms("group_end", "first_read"),
            )
        )
    return output, spans


def trace(marks, spans):
    import itertools

    events = []
    zero = min(times["request_start"] for _, _, times in spans)
    for lane, (wave, rid, times) in enumerate(spans, 1):
        events.append(dict(ph="M", name="thread_name", pid=1, tid=lane, args=dict(name=wave + " " + rid)))
        for (start, t0), (end, t1) in itertools.pairwise(times.items()):
            events.append(
                dict(
                    ph="X",
                    name=start + " → " + end,
                    cat="request",
                    pid=1,
                    tid=lane,
                    ts=(t0 - zero) / 1000,
                    dur=(t1 - t0) / 1000,
                    args=dict(id=rid),
                )
            )
    last = max(times["first_read"] for _, _, times in spans)
    for e in marks:
        if zero <= e["ns"] <= last:
            events.append(
                dict(
                    ph="i",
                    s="t",
                    name=e["name"],
                    cat="host",
                    pid=2,
                    tid=e["pid"],
                    ts=(e["ns"] - zero) / 1000,
                    args={k: v for k, v in e.items() if k not in ("pid", "ns", "name")},
                )
            )
    (base / "delivery.perfetto.json.gz").write_bytes(
        gzip.compress(json.dumps({"traceEvents": events}).encode(), mtime=0)
    )


def archive(directory):
    rows = []
    for wave in ("measure0", "measure1"):
        wave_rows = json.loads((directory / (wave + ".json")).read_text())
        if len(wave_rows) != 4:
            raise ValueError(f"{wave} must contain four completed requests")
        rows.extend(dict(**row, wave=wave) for row in wave_rows)
    start = min(r["start"] for r in rows)
    end = max(r["end"] for r in rows)
    events = []
    for path in directory.glob("marks-*.jsonl"):
        with path.open() as stream:
            for line in stream:
                event = json.loads(line)
                if start <= event["ns"] <= end:
                    events.append(event)
    events.sort(key=lambda e: e["ns"])
    data = "".join(json.dumps(e) + "\n" for e in events).encode()
    (base / "marks.jsonl.gz").write_bytes(gzip.compress(data, mtime=0))
    (base / "requests.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        archive(Path(sys.argv[1]))
    marks = read_marks()
    requests = json.loads((base / "requests.json").read_text())
    results, spans = summarize(marks, requests)
    (base / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    trace(marks, spans)
    for row in results:
        print(row)
