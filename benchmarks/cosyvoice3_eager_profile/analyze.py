import json, gzip, sys, collections, bisect
from pathlib import Path

def union(intervals):
    end = -float('inf')
    total = 0
    for a, b in sorted(intervals):
        if b > end:
            total += b - max(a, end)
            end = b
    return total

def analyze(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        obj = json.load(f)
    events = [e for e in obj['traceEvents'] if e.get('ph') == 'X' and 'dur' in e]
    gpu = [e for e in events if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')]
    scopes = [e for e in events if e.get('cat') == 'user_annotation' and e['name'].startswith(('cv3_', 'cosyvoice3_'))]
    scope_totals = collections.defaultdict(lambda: dict(calls=0, cpu_inclusive_ms=0, gpu_sum_ms=0, gpu_activities=0))
    for e in scopes:
        s = scope_totals[e['name']]
        s['calls'] += 1
        s['cpu_inclusive_ms'] += e['dur']/1000
    # Correlate device activity to its CPU operator via External id, then
    # attribute that operator to every enclosing diagnostic scope on its thread.
    by_thread = collections.defaultdict(list)
    for e in scopes:
        by_thread[e['pid'], e['tid']].append(e)
    scope_index = collections.defaultdict(list)
    for key, values in by_thread.items():
        names=collections.defaultdict(list)
        for e in values:
            names[e['name']].append(e)
        for values in names.values():
            values.sort(key=lambda e:e['ts'])
            scope_index[key].append(([e['ts'] for e in values], values))
    cpu_ids = {}
    for e in events:
        if e.get('cat') == 'cpu_op' and 'External id' in e.get('args', {}):
            cpu_ids[e['args']['External id']] = e
    matched = 0
    kernel_totals = collections.defaultdict(lambda: [0,0])
    for e in gpu:
        k = kernel_totals[e['name']]; k[0] += 1; k[1] += e['dur']/1000
        cpu = cpu_ids.get(e.get('args', {}).get('External id'))
        if not cpu:
            continue
        matched += 1
        for starts, values in scope_index.get((cpu['pid'],cpu['tid']), []):
            index=bisect.bisect_right(starts,cpu['ts'])-1
            if index < 0:
                continue
            s=values[index]
            if s['ts'] <= cpu['ts'] and cpu['ts']+cpu['dur'] <= s['ts']+s['dur']+1:
                item = scope_totals[s['name']]
                item['gpu_sum_ms'] += e['dur']/1000
                item['gpu_activities'] += 1
    gpu_iv = [(e['ts'],e['ts']+e['dur']) for e in gpu]
    span = max((b for a,b in gpu_iv),default=0)-min((a for a,b in gpu_iv),default=0)
    sync = collections.defaultdict(lambda: [0,0])
    for e in events:
        if e.get('cat') == 'cuda_runtime' and ('Synchronize' in e['name'] or 'Memcpy' in e['name']):
            sync[e['name']][0] += 1
            sync[e['name']][1] += e['dur']/1000
    return dict(file=str(path),gpu_events=len(gpu),matched_gpu_events=matched,
                gpu_active_union_ms=union(gpu_iv)/1000,gpu_sum_ms=sum(b-a for a,b in gpu_iv)/1000,
                gpu_span_ms=span/1000,scope_totals=dict(scope_totals),sync=dict(sync),
                top_kernels=sorted([dict(name=n,calls=k[0],gpu_sum_ms=k[1]) for n,k in kernel_totals.items()],key=lambda x:-x['gpu_sum_ms'])[:25],
                categories=dict(collections.Counter(e.get('cat') for e in events)))

root=Path(sys.argv[1])
results=[]
for p in sorted((root/'traces').glob('*.json.gz')):
    print('ANALYZE',p,flush=True)
    results.append(analyze(p))
(root/'analysis.json').write_text(json.dumps(results,indent=2))
for item in results:
    print(json.dumps(item,indent=2))
