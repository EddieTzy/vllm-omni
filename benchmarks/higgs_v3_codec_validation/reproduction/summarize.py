"""Read the streaming client and codec diagnostics without including prompts or audio."""
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

work = Path(__file__).resolve().parent
label = sys.argv[1]
d = work / label
data = json.loads((work.parent / 'results' / ('higgs3-codec-' + label) / 'result.json').read_text())
rows = data['requests']
summary = dict(data['summaries'][0], mean_e2e_s=statistics.mean(r['elapsed_seconds'] for r in rows),
               mean_rtf=statistics.mean(r['elapsed_seconds']/r['audio_seconds'] for r in rows),
               requests_with_multiple_http_blocks=sum(len(r['chunk_arrivals'])>1 for r in rows),
               max_audio_seconds=max(r['audio_seconds'] for r in rows))
if (d/'events.jsonl').exists():
    events = [json.loads(line) for line in (d/'events.jsonl').read_text().splitlines()]
    regular = [e for e in events if not e['traced'] and e['groups']]
    groups = [g for e in regular for g in e['groups']]
    if groups:
        summary.update(codec_forwards=len(regular), groups=len(groups), cuda_groups=sum(g['gpu'] for g in groups),
                       validate_median_us=statistics.median(g['validate_us'] for g in groups),
                       validate_aggregated_host_share=sum(g['validate_us'] for g in groups)/sum(e['forward_us'] for e in regular),
                       graph_hits=sum(g['graph'] for g in groups),
                       group_shapes=Counter(str((g['batch'],g['frames'])) for g in groups),
                       groups_after_grouping_by_frames=sum(len({g['frames'] for g in e['groups']}) for e in regular))
    traces=[]
    for f in sorted(d.glob('trace-*.json')):
        trace=json.loads(f.read_text())['traceEvents']
        ranges=[e for e in trace if e.get('name')=='higgs3.codec.validate_codes' and 'dur' in e]
        ops=Counter()
        for region in ranges:
            for e in trace:
                if e.get('tid')==region['tid'] and e.get('pid')==region['pid'] and e.get('ph')=='X' and region['ts']<=e['ts']<region['ts']+region['dur']:
                    ops[e['name']]+=1
        traces.append(dict(file=f.name,regions=len(ranges),region_ops=dict(ops)))
    summary['traces']=traces
(d/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k not in ['traces','group_shapes']},indent=2))
