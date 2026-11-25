"""Monotonic host annotations. No device synchronization or event polling loop."""

import json
import os
import time

_stream = None


def mark(name, ids, **fields):
    global _stream
    ts = time.monotonic_ns()
    if _stream is None:
        path = os.path.join(os.environ["CV3_DELIVERY_DIR"], f"marks-{os.getpid()}.jsonl")
        _stream = open(path, "a", buffering=1)
    record = dict(ns=ts, pid=os.getpid(), name=name, ids=list(ids), **fields)
    _stream.write(json.dumps(record, separators=(",", ":")) + "\n")


async def instrument_send(response, send, stream_response):
    first = True

    async def timed(message):
        nonlocal first
        if first and message["type"] == "http.response.body" and message.get("body"):
            first = False
            ids = [response.headers.get("X-CV3-ID")]
            mark("asgi_send", ids)
            await send(message)
            mark("asgi_sent", ids)
        else:
            await send(message)

    await stream_response(timed)
