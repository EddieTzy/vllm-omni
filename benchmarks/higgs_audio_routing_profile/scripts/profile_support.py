"""Private diagnostic instrumentation; never part of the proposed model patch."""
import atexit
import functools
import json
import os
from pathlib import Path
import time

import torch

ROOT = Path('/data/results/higgs-routing-profile') / os.environ['HIGGS_PROFILE_ARM']
ROOT.mkdir(parents=True, exist_ok=True)
LOG = (ROOT / f'events-{os.getpid()}.jsonl').open('a', buffering=1)
PENDING = []
COUNTS = {}
ACTIVE = False
TRACES = 0


def flush(final=False):
    if final and PENDING:
        torch.cuda.synchronize()
    while PENDING and (final or PENDING[0][2].query()):
        row, start, end = PENDING.pop(0)
        row['cuda_interval_ms'] = start.elapsed_time(end)
        LOG.write(json.dumps(row) + '\n')


atexit.register(lambda: flush(True))


def route_wrap(method, label):
    @functools.wraps(method)
    def wrapped(*args, **kwargs):
        if ACTIVE:
            with torch.profiler.record_function(label):
                return method(*args, **kwargs)
        return method(*args, **kwargs)
    return wrapped


def stage_wrap(method, label):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        global ACTIVE, TRACES
        counts = COUNTS.setdefault(label, {})
        # First tensor is input_ids for talker.forward and codes for codec.forward.
        tensor = next((x for x in args if isinstance(x, torch.Tensor)), None)
        if tensor is None:
            tensor = next((x for x in kwargs.values() if isinstance(x, torch.Tensor)), None)
        shape = list(tensor.shape) if tensor is not None else None
        key = str(shape)
        counts[key] = counts.get(key, 0) + 1
        step = sum(counts.values())
        # Capture two real 32-row decoder forwards, after first-use warmup.
        trace = (label == 'talker.forward' and shape == [32]
                 and counts[key] in (32, 64) and TRACES < 2)
        flush()
        profiler = None
        if trace:
            torch.cuda.synchronize()
            profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True, with_stack=False)
            profiler.__enter__()
            ACTIVE = True
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter_ns()
        try:
            result = method(self, *args, **kwargs)
        finally:
            after = time.perf_counter_ns()
            end.record()
            if profiler is not None:
                ACTIVE = False
                profiler.__exit__(None, None, None)
        row = dict(stage=label, step=step, shape=shape,
                   shape_occurrence=counts[key], host_ms=(after-before)/1e6,
                   profiled=trace, monotonic_ns=before)
        PENDING.append((row, start, end))
        if profiler is not None:
            TRACES += 1
            stem = ROOT / f'trace-{TRACES}'
            profiler.export_chrome_trace(str(stem) + '.json')
            entries = []
            for event in profiler.key_averages():
                entries.append(dict(name=event.key, count=event.count,
                                    cpu_total_us=event.cpu_time_total,
                                    self_cpu_us=event.self_cpu_time_total,
                                    device_total_us=event.device_time_total,
                                    self_device_us=event.self_device_time_total))
            (Path(str(stem) + '-ops.json')).write_text(json.dumps(entries, indent=2))
            metadata = dict(row, input_ids=tensor.detach().cpu().tolist())
            (Path(str(stem) + '-meta.json')).write_text(json.dumps(metadata, indent=2))
        return result
    return wrapped


def instrument_talker(talker, layer):
    for name in ('forward', 'compute_logits', 'sample'):
        setattr(talker, name, stage_wrap(getattr(talker, name), 'talker.' + name))
    for name in ('_routed_norm', '_routed_mlp'):
        setattr(layer, name, route_wrap(getattr(layer, name), 'routing.' + name))


def instrument_codec(codec):
    codec.forward = stage_wrap(codec.forward, 'codec.forward')
