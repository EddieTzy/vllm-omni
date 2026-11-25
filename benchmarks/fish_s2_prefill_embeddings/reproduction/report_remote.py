"""Collect only non-audio, non-prompt diagnostics and metrics for publication."""
import collections,hashlib,json,os,statistics,sys,tarfile
from pathlib import Path
w=Path(__file__).resolve().parent
base=sys.argv[1] if len(sys.argv)>1 else 'base-a'
prefix='s2-mrv2-prefill-'
src='vllm_omni/model_executor/models/fish_speech/fish_speech_slow_ar.py'
results={}
requests={}
for arm in [base,'cand-b','cand-b2','base-a2','diag-a','diag-b']:
    path=w.parent/'results'/(prefix+arm)/'result.json'
    d=json.loads(path.read_text())
    rows=d['requests'];assert len(rows)==64 and all(r['bytes']>0 for r in rows)
    assert len({r['request_number'] for r in rows})==64 and d['request_seed'] is None
    row_metrics=[]
    for r in rows:
        metric={k:r[k] for k in ('request_number','concurrency','bytes','audio_seconds','elapsed_seconds','ttfa_seconds')}
        metric['http_blocks']=len(r['chunk_arrivals'])
        row_metrics.append(metric)
    requests[arm]=row_metrics
    s=d['summaries'][0]
    assert s['concurrency']==32
    results[arm]=dict(audio_per_s=sum(r['audio_seconds'] for r in rows)/s['wall_seconds'],wall=s['wall_seconds'],audio_total=sum(r['audio_seconds'] for r in rows),p50_ms=1000*s['ttfa_p50_seconds'],p95_ms=1000*s['ttfa_p95_seconds'],e2e_s=statistics.mean(r['elapsed_seconds'] for r in rows),rtf=statistics.mean(r['elapsed_seconds']/r['audio_seconds'] for r in rows),multi_block=sum(r['http_blocks']>1 for r in row_metrics))
(w/'serving.json').write_text(json.dumps(results,indent=2))
(w/'request-metrics.json').write_text(json.dumps(requests,indent=2))
evidence=[]
profile={}
counts={}
for arm in ('diag-a','diag-b'):
    d=w/arm
    rows=[json.loads(line) for line in (d/'embeddings.jsonl').read_text().splitlines()]
    assert rows and all(r['clone'] for r in rows)
    counts[arm]=collections.Counter((r['embedding_sha256'],tuple(r['shape']),r['span_len']) for r in rows)
    regular=[r for r in rows if not r['traced']]
    stats=dict(calls=len(rows),eligible=sum(r['span_len']==r['shape'][0] for r in rows),median_host_us=statistics.median(r['host_us'] for r in regular),byte_range=[min(r['embedding_bytes'] for r in rows),max(r['embedding_bytes'] for r in rows)],traces=[])
    evidence.append((d/'embeddings.jsonl',arm+'/embeddings.jsonl'))
    for f in sorted(d.glob('trace-*.json')):
        events=json.loads(f.read_text())['traceEvents']
        root=[e for e in events if e['name']=='fish.first_prefill_embedding_transfer' and e.get('cat')=='user_annotation']
        assert len(root)==1
        copies=collections.Counter()
        copybytes=collections.Counter()
        for e in events:
            if e.get('cat')=='gpu_memcpy':
                copies[e['name']]+=1
                copybytes[e['name']]+=e.get('args',{}).get('bytes',0)
        ops=collections.Counter(e.get('name') for e in events if e.get('cat')=='cuda_runtime' and e.get('ph')=='X' and root[0]['ts']<=e['ts']<root[0]['ts']+root[0]['dur'])
        stats['traces'].append(dict(file=f.name,region_us=root[0]['dur'],cuda_runtime=ops,copies=copies,copy_bytes=copybytes))
        evidence.append((f,arm+'/'+f.name))
    assert len(stats['traces'])==3
    profile[arm]=stats
profile['matched_embedding_hash_multiset']=counts['diag-a']==counts['diag-b']
profile['matched_occurrences']=sum((counts['diag-a']&counts['diag-b']).values())
(w/'profile.json').write_text(json.dumps(profile,indent=2))
with tarfile.open(w/'evidence.tar.gz','w:gz') as out:
    for f,name in evidence: out.add(f,arcname=name,recursive=False)
info={'source_sha256':{s:hashlib.sha256((w/s/src).read_bytes()).hexdigest() for s in ['source','candidate']},'reference_sha256':hashlib.sha256((w/'source/tests/assets/qwen3_tts/clone_2.wav').read_bytes()).hexdigest(),'trace_archive_sha256':hashlib.sha256((w/'evidence.tar.gz').read_bytes()).hexdigest()}
(w/'hashes.json').write_text(json.dumps(info,indent=2))
