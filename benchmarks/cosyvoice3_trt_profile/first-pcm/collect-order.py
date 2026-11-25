"""Locate the stage worker's CUDA event waits after first-chunk model invocations."""
import gzip
import json
from pathlib import Path

here = Path(__file__).resolve().parent
report = {}
for arm in ('seeded', 'unseeded'):
    with gzip.open(here.parent / 'traces' / f'{arm}-stage1.json.gz') as stream:
        events = sorted((e for e in json.load(stream)['traceEvents'] if e.get('ph') == 'X'), key=lambda e: e['ts'])
    groups = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith('execute_context_')]
    rows = []
    for index, group in enumerate(groups):
        if group['name'] not in {'execute_context_1(25)_generation_0(0)', 'execute_context_3(75)_generation_0(0)'}:
            continue
        following, later = groups[index + 1:index + 3]
        next_end = following['ts'] + following['dur']
        waits = [e for e in events if e.get('cat') == 'cuda_runtime' and e['name'] == 'cudaEventSynchronize' and e['tid'] == group['tid'] and next_end <= e['ts'] < later['ts']]
        rows.append(dict(first_group=group['name'], next_group=following['name'], next_forward_ms=following['dur']/1000,
                         waits_after_next=[dict(after_next_end_ms=(e['ts']-next_end)/1000, duration_us=e['dur']) for e in waits]))
    report[arm] = rows
(here/'collect-order.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['unseeded'],indent=2))
