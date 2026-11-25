"""Diagnostic-only overlay: trace streaming codec forwards on the active service."""
from collections import Counter
from contextlib import nullcontext
import json
import os
from pathlib import Path
import time
import torch


def instrument(cls):
    dest = Path(os.environ["HIGGS3_ARTIFACTS"])
    if not os.environ.get("HIGGS3_PROFILE"):
        return
    gate = dest / "armed"
    output = dest / "events.jsonl"
    current = None
    step = 0
    traced = set()
    original = cls.forward
    validate = cls._validate_codes
    chunk = cls.forward_chunk

    def forward(self, *args, **kwargs):
        nonlocal current, step
        if not gate.exists():
            return original(self, *args, **kwargs)
        step += 1
        current = {"step": step, "groups": []}
        trace = step in (3, 24, 80)
        manager = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) if trace else nullcontext()
        with manager as prof:
            begin = time.perf_counter_ns()
            result = original(self, *args, **kwargs)
            current["forward_us"] = (time.perf_counter_ns() - begin) / 1000
        if trace:
            prof.export_chrome_trace(str(dest / f"trace-{step}.json"))
        current["traced"] = trace
        with output.open("a") as f:
            f.write(json.dumps(current) + "\n")
        current = None
        return result

    def forward_chunk(self, codes, *args, **kwargs):
        if current is not None:
            current["groups"].append(dict(batch=codes.shape[0], frames=codes.shape[-1], left=kwargs.get("left_context_size"), right=kwargs.get("right_holdback_size"), gpu=codes.is_cuda, graph=(codes.shape[0],codes.shape[-1]) in self._decode_graphs and self._codec_graph_enabled))
        return chunk(self, codes, *args, **kwargs)

    def checked(self, codes):
        if current is None:
            return validate(self, codes)
        with torch.profiler.record_function("higgs3.codec.validate_codes"):
            begin = time.perf_counter_ns()
            result = validate(self, codes)
            current["groups"][-1]["validate_us"] = (time.perf_counter_ns() - begin) / 1000
            return result

    cls.forward, cls.forward_chunk, cls._validate_codes = forward, forward_chunk, checked
