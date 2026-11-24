"""Summarize diagnostic stage events; Chrome traces remain the primary evidence."""
import collections
import json
from pathlib import Path
import statistics
import sys

root = Path(sys.argv[1])


def percentile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def stats(values):
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p05=percentile(values, .05), p95=percentile(values, .95))


report = {}
for arm in ('A', 'B', 'A2', 'B2'):
    rows = []
    for path in sorted((root / arm).glob('events-*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines())
    trace_steps = {row['step'] for row in rows if row['profiled']}
    excluded_forward = {step + offset for step in trace_steps for offset in (0, 1, 2)}
    forward = [row for row in rows if row['stage'] == 'talker.forward'
               and row['shape'] == [32] and row['shape_occurrence'] > 16
               and row['step'] not in excluded_forward]
    all_forwards = sorted((row for row in rows if row['stage'] == 'talker.forward'), key=lambda row: row['step'])
    selected_steps = {row['step'] for row in forward}
    cycles = []
    for current, following in zip(all_forwards, all_forwards[1:]):
        if current['step'] in selected_steps and following['step'] in selected_steps:
            elapsed = (following['monotonic_ns'] - current['monotonic_ns']) / 1e6
            cycles.append(dict(cycle_ms=elapsed, forward_host_ms=current['host_ms'],
                               outside_forward_ms=elapsed-current['host_ms']))
    stages = {}
    for label in sorted({row['stage'] for row in rows}):
        selected = [row for row in rows if row['stage'] == label and not row['profiled']]
        stages[label] = dict(calls=len(selected),
                            shapes=dict(collections.Counter(str(row['shape']) for row in selected)),
                            host_ms=stats([row['host_ms'] for row in selected]) if selected else None)
    traces = []
    for path in sorted((root / arm).glob('trace-*-ops.json')):
        ops = json.loads(path.read_text())
        by_name = {op['name']: op for op in ops}
        metadata = json.loads(path.with_name(path.name.replace('-ops.json', '-meta.json')).read_text())
        chrome = json.loads(path.with_name(path.name.replace('-ops.json', '.json')).read_text())['traceEvents']
        chrome_metrics = {}
        for category in ('user_annotation', 'kernel', 'gpu_memcpy', 'cuda_runtime'):
            selected = [event for event in chrome if event.get('cat') == category and event.get('ph') == 'X']
            totals = collections.defaultdict(list)
            for event in selected:
                totals[event['name']].append(event['dur'])
            chrome_metrics[category] = dict(count=len(selected), total_us=sum(event['dur'] for event in selected),
                                           by_name={name: dict(count=len(values), total_us=sum(values))
                                                    for name, values in totals.items()})
        traces.append(dict(metadata=metadata,
                           chrome=chrome_metrics,
                           operations={key: by_name[key] for key in (
                               'routing._routed_norm', 'routing._routed_mlp',
                               'aten::nonzero', 'aten::index', 'aten::_index_put_impl_',
                               'aten::any', 'aten::all', 'aten::item', 'aten::_local_scalar_dense',
                               'cudaStreamSynchronize', 'cudaDeviceSynchronize',
                               'cudaMemcpyAsync', 'cudaLaunchKernel') if key in by_name}))
    # Match exact codec input shape; do not infer equal workloads from client concurrency.
    codec = collections.defaultdict(list)
    for row in rows:
        if row['stage'] == 'codec.forward' and row['shape_occurrence'] > 2:
            codec[str(row['shape'])].append(row)
    report[arm] = dict(c32_forward_host_ms=stats([row['host_ms'] for row in forward]) if forward else None,
                      c32_forward_cuda_interval_ms=stats([row['cuda_interval_ms'] for row in forward]) if forward else None,
                      c32_cycles={key: stats([row[key] for row in cycles])
                                  for key in ('cycle_ms', 'forward_host_ms', 'outside_forward_ms')} if cycles else None,
                      stages=stages, traces=traces,
                      codec_by_shape={shape: dict(host_ms=stats([r['host_ms'] for r in group]),
                                                cuda_interval_ms=stats([r['cuda_interval_ms'] for r in group]))
                                      for shape, group in codec.items()})
report['limitations'] = [
    'Private instrumentation, not a new uninstrumented serving throughput benchmark.',
    'CUDA event interval includes launch gaps and same-stream waits; it is not summed kernel execution time.',
    'Stage intervals can overlap across processes; summing them does not reconstruct request critical paths.',
    'C32 forward summary conditions on 32 input tokens, drops first 16 occurrences and traced plus next two forwards.',
    'Same client workload does not imply identical stochastic output lengths or batch trajectories.',
    'Codec comparison must condition on matching input shape, rather than comparing all-call medians.',
]
(root / 'summary.json').write_text(json.dumps(report, indent=2))
print(json.dumps({arm: dict(host_ms=report[arm]['c32_forward_host_ms'],
                           cuda_interval_ms=report[arm]['c32_forward_cuda_interval_ms'],
                           traces=len(report[arm]['traces']),
                           stage_calls={k:v['calls'] for k,v in report[arm]['stages'].items()})
                  for arm in ('A', 'B', 'A2', 'B2')}, indent=2))
