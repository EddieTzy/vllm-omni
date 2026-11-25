"""Private identical diagnostic instrumentation, excluded from PR."""
import atexit
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import time
import torch

ROOT = Path('/data/results/higgs-followup-profile') / os.environ['HIGGS_PROFILE_ARM']
ROOT.mkdir(parents=True, exist_ok=True)
LOG = (ROOT / f'events-{os.getpid()}.jsonl').open('a', buffering=1)
PENDING = []
STEP = 0
ACTIVE = False
COUNT32 = 0
TRACES = 0
META = {}

def flush(final=False):
    if final and PENDING:
        torch.cuda.synchronize()
    while PENDING and (final or PENDING[0][2].query()):
        row, start, end = PENDING.pop(0)
        row['cuda_interval_ms'] = start.elapsed_time(end)
        LOG.write(json.dumps(row) + '\n')

atexit.register(lambda: flush(True))

@contextmanager
def phase(label, **metadata):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    before = time.perf_counter_ns()
    row = dict(stage=label, step=STEP, profiled=ACTIVE, monotonic_ns=before, **metadata)
    with torch.profiler.record_function(label) if ACTIVE else __import__('contextlib').nullcontext():
        try:
            yield
        finally:
            after = time.perf_counter_ns()
            end.record()
            row['host_ms'] = (after-before)/1e6
            PENDING.append((row, start, end))

def wrap(method, label):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        global STEP, COUNT32, ACTIVE, TRACES
        if label == 'talker.forward':
            STEP += 1
            flush()
        tensor = next((x for x in args if isinstance(x, torch.Tensor)), None)
        if tensor is None:
            tensor = next((x for x in kwargs.values() if isinstance(x, torch.Tensor)), None)
        shape = list(tensor.shape) if tensor is not None else None
        meta = dict(shape=shape)
        trace = False
        batch = None
        if label == 'talker.sample_step':
            batch = args[1] if len(args)>1 else kwargs['input_batch']
            meta['num_reqs'] = batch.num_reqs
            meta['output_rows'] = batch.logits_indices.numel()
            if shape and shape[0] == 32 and batch.num_reqs == 32 and meta['output_rows'] == 32:
                COUNT32 += 1
                trace = COUNT32 in (64,128)
            meta['c32_occurrence'] = COUNT32
        if label == 'talker.audio_mode_bias':
            tokens = kwargs.get('row_prev_tokens')
            if tokens is not None:
                allowed = (self.config.audio_token_id, getattr(self.config, 'audio_delay_token_id', -999))
                LOG.write(json.dumps(dict(kind='iteration', step=STEP, rows=len(tokens), all_audio=bool(tokens) and all(t in allowed for t in tokens)))+'\n')
        profiler = None
        if trace:
            torch.cuda.synchronize()
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], record_shapes=True, with_stack=False)
            profiler.__enter__()
            ACTIVE = True
        try:
            with phase(label, **meta):
                result = method(self, *args, **kwargs)
        finally:
            if profiler is not None:
                ACTIVE = False
                profiler.__exit__(None,None,None)
                TRACES += 1
                stem = ROOT / f'trace-{TRACES}'
                profiler.export_chrome_trace(str(stem)+'.json')
                trace_meta = dict(step=STEP, **meta, input_ids=batch.input_ids[batch.logits_indices].tolist())
                Path(str(stem)+'-meta.json').write_text(json.dumps(trace_meta,indent=2))
                Path(str(stem)+'-ops.json').write_text(json.dumps([dict(name=e.key,count=e.count,cpu_total_us=e.cpu_time_total,self_cpu_us=e.self_cpu_time_total,device_total_us=e.device_time_total,self_device_us=e.self_device_time_total) for e in profiler.key_averages()],indent=2))
        return result
    return wrapped

def instrument_talker(talker, sampler):
    for name, label in [('forward','forward'),('compute_logits','compute_logits'),('audio_codebook_logits','audio_head'),('_apply_audio_mode_bias','audio_mode_bias'),('_dispatch_audio_codes','dispatch_audio'),('_current_codes_by_request','deliver_codes')]:
        setattr(talker, name, wrap(getattr(talker,name), 'talker.'+label))
    sampler.sample_step = wrap(sampler.sample_step, 'talker.sample_step')

def instrument_codec(codec):
    codec.forward = wrap(codec.forward, 'codec.forward')
