import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
import time

import torch

dest = Path(os.environ["FISH_ARTIFACTS"])
count = 0
record = None

@contextmanager
def trace_transfer(embeds, span_len, info):
    global count, record
    if not (dest/'armed').exists():
        yield
        return
    count += 1
    tracing = count in (4,40,78)
    record = dict(call=count,span_len=span_len,shape=list(embeds.shape),embedding_bytes=embeds.numel()*embeds.element_size(),clone=bool(info.get('fish_structured_voice_clone')),traced=tracing)
    if tracing:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.profiler.record_function("fish.first_prefill_embedding_transfer"):
                yield
        prof.export_chrome_trace(str(dest/f"trace-{count}.json"))
    else:
        start=time.perf_counter_ns()
        yield
        record['host_us']=(time.perf_counter_ns()-start)/1000

def record_output(embeds):
    global record
    if record is None:
        return
    # Hash outside the measured region and only on the diagnostic service.
    record["embedding_sha256"]=hashlib.sha256(embeds.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    with (dest/'embeddings.jsonl').open('a') as f:
        f.write(json.dumps(record)+'\n')
    record = None
