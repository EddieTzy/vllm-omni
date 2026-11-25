import json
from pathlib import Path
import statistics
from collections import Counter
import sys
ROOT=Path(sys.argv[1])
def stats(xs):
    xs=sorted(xs)
    if not xs:return None
    return dict(n=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),p95=xs[min(len(xs)-1,int(.95*len(xs)))])
report={}
for arm in ('A','B','B2','A2'):
    path=ROOT/arm
    if not path.exists():continue
    stages={}; cycles=[]; traces=[]
    for log in path.glob('events-*.jsonl'):
        rows=[json.loads(l) for l in log.read_text().splitlines() if l.strip()]
        metadata={r['step']:r for r in rows if r.get('kind')=='iteration'}
        sampled={r['step']:r for r in rows if r.get('stage')=='talker.sample_step'}
        bad={r['step']+d for r in rows if r.get('profiled') for d in range(3)}
        valid={s for s,r in sampled.items() if s not in bad and r.get('num_reqs')==32 and r.get('output_rows')==32 and r.get('shape',[0])[0]==32 and r.get('c32_occurrence',0)>16 and metadata.get(s,{}).get('all_audio') and metadata[s]['rows']==32}
        mask_steps={r['step'] for r in rows if r.get('stage')=='mask.delay_pattern' and r.get('audio_rows')==32}
        valid &= mask_steps
        for row in rows:
            label=row.get('stage')
            if not label or row['step'] not in valid:continue
            if 'audio_rows' in row and row['audio_rows']!=32:continue
            stages.setdefault(label,[]).append(row)
        forwards={r['step']:r for r in rows if r.get('stage')=='talker.forward'}
        for step in valid:
            if step+1 in valid and step in forwards and step+1 in forwards:
                cycles.append((forwards[step+1]['monotonic_ns']-forwards[step]['monotonic_ns'])/1e6)
    for file in path.glob('trace-*-meta.json'):
        stem=str(file).replace('-meta.json','.json')
        data=json.loads(Path(stem).read_text()); meta=json.loads(file.read_text())
        events=data['traceEvents']; annotations={}
        for e in events:
            if e.get('cat')=='user_annotation' and e.get('ph')=='X':
                annotations.setdefault(e['name'],[]).append(e['dur']/1000)
        mask_ranges=[e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='mask.delay_pattern' and e.get('ph')=='X']
        mask_details=[]
        for bounds in mask_ranges:
            inside=[e for e in events if e.get('ph')=='X' and e.get('pid')==bounds['pid'] and e.get('tid')==bounds['tid'] and bounds['ts']<=e.get('ts',-1)<bounds['ts']+bounds['dur']]
            ids={e.get('args',{}).get('External id') for e in inside if e.get('cat') in ('cpu_op','user_annotation')}
            ids.discard(None)
            correlations={e.get('args',{}).get('correlation') for e in inside if e.get('cat') in ('cuda_runtime','cuda_driver')}
            correlations.discard(None)
            kernels=[e for e in events if e.get('cat')=='kernel' and (e.get('args',{}).get('External id') in ids or e.get('args',{}).get('correlation') in correlations)]
            copies=[e for e in events if e.get('cat')=='gpu_memcpy' and e.get('args',{}).get('External id') in ids]
            mask_details.append(dict(correlated_memcpy_count=len(copies),correlated_memcpy_busy_ms=sum(e.get('dur',0) for e in copies)/1000,cpu_ops=dict(Counter(e['name'] for e in inside if e.get('cat')=='cpu_op')),cuda_driver=dict(Counter(e['name'] for e in inside if e.get('cat')=='cuda_driver')),cuda_runtime=dict(Counter(e['name'] for e in inside if e.get('cat')=='cuda_runtime')),correlated_kernel_count=len(kernels),correlated_kernel_busy_ms=sum(e.get('dur',0) for e in kernels)/1000,kernels=dict(Counter(e['name'] for e in kernels))))
        traces.append(dict(mask_details=mask_details,metadata=meta,annotation_host_ms={k:stats(v) for k,v in annotations.items()},kernel_count=sum(e.get('cat')=='kernel' for e in events),cuda_runtime_calls=sum(e.get('cat')=='cuda_runtime' for e in events)))
    report[arm]=dict(stages={k:{metric:stats([r[metric] for r in v]) for metric in ('host_ms','cuda_interval_ms')} for k,v in stages.items()},decode_cycle_ms=stats(cycles),traces=traces)
output=dict(scope='Real-service C32 all-audio single-token steps; CPU wall and CUDA event intervals (including launch gaps), profiler steps and next 2 excluded. Event-timed runs diagnostic only. Codec excluded because input work is not matched.',arms=report)
(ROOT/'summary.json').write_text(json.dumps(output,indent=2))
for arm,value in report.items():
    print(arm,'cycle',value['decode_cycle_ms'])
    for key,value in value['stages'].items():print(key,value)
