import re,json,gzip,bisect
from pathlib import Path
from collections import defaultdict,Counter
from statistics import mean,median
import sys
base=Path(__file__).parent
def distribution(seq):
    v=sorted(seq)
    if not v:return None
    t=(len(v)-1)*.95;i=int(t)
    return dict(count=len(v),mean=mean(v),median=median(v),p95=v[i]+(v[min(i+1,len(v)-1)]-v[i])*(t-i),max=max(v),min=min(v))
def perf(folder):
    log=re.sub(r'\x1b\[[0-9;]*m','',(folder/'server.log').read_text(errors='replace'))
    # Docker interleaved two worker prints; join only the unambiguous numeric suffix.
    log=re.sub(r'^(CV3READY \S+ \d+ )(CV3BEGIN \S+ \d+ \d+ \d+)\n(\d+ \d+)$',r'\1\3\n\2',log,flags=re.M)
    log=re.sub(r'^CV3END(CV3FLOW \d+ \d+ \d+)\n (speech-\S+ \d+)$',r'\1\nCV3END \2',log,flags=re.M)
    log=re.sub(r'^(CV3END \S+) CV3BEGIN(\d+) *\n(speech-\S+ \d+ \d+ \d+)$',r'\1 \2\nCV3BEGIN \3',log,flags=re.M)
    final={rid:int(ts) for rid,ts in re.findall(r'CV3END (\S+) (\d+)',log)}
    flows=[(int(a),int(b)) for a,b in re.findall(r'CV3FLOW (\d+) (\d+) \d+',log)]
    def read(key):
        return {(rid,int(off),int(n)):int(t) for rid,off,n,t in re.findall(key+r' (\S+) (\d+) (\d+) (\d+)',log)}
    ready=read('CV3READY');beg=read('CV3BEGIN')
    measures=[json.loads(p.read_text()) for p in sorted(folder.glob('measure*.json'))]
    out=defaultdict(list);audio=wall=0;matched_rows=0
    for w in measures:
        rows=w['rows'];audio+=sum(r['audio'] for r in rows);wall+=w['wall']
        end=max(r['last'] for r in rows);start=end-w['wall']*1e9
        flow_times=[(b-a)/1e6 for a,b in flows if start<=a<b<=end]
        out['flow_call_ms'].extend(flow_times)
        for r in rows:
            out['first_s'].append(r['first']);out['rtf'].append(r['e2e']/r['audio']);out['e2e_s'].append(r['e2e'])
            ids=[k for k in final if k==r['id'] or k.startswith(r['id']+'-')]
            assert len(ids)==1,ids
            key=ids[0];out['tail_s'].append((r['last']-final[key])/1e9)
            readyset={k[1:]:v for k,v in ready.items() if k[0]==key}; beginset={k[1:]:v for k,v in beg.items() if k[0] in {r['id'],key} or k[0].startswith(r['id']+'-')}
            if readyset and set(readyset)==set(beginset):
                matched_rows+=1
                firstblock=min(readyset)
                # Client retained last nonempty read and E2E; EOF follows the last read.
                request_start_est=r['last']-r['e2e']*1e9
                out['client_start_to_first_ready_approx_s'].append((readyset[firstblock]-request_start_est)/1e9)
                out['first_ready_to_start_s'].append((beginset[firstblock]-readyset[firstblock])/1e9)
                out['first_start_to_first_pcm_approx_s'].append((request_start_est+r['first']*1e9-beginset[firstblock])/1e9)
                out['chunk_ready_to_start_s'].extend((beginset[k]-readyset[k])/1e9 for k in readyset)
                for k,a in readyset.items():
                    b=beginset[k]
                    others={j[0] for j,t in ready.items() if j[0]!=key and t<=b<beg.get(j,-1)}
                    out['other_distinct_requests_ready_at_start'].append(len(others))
                    out['any_other_request_ready_at_start_fraction'].append(int(bool(others)))
                    out['other_flow_host_overlap_during_chunk_wait_s'].append(sum(max(0,min(b,f1)-max(a,f0)) for f0,f1 in flows)/1e9)
    return dict(boots=1,waves=len(measures),audio_s=audio,wall_s=wall,throughput=audio/wall,chunk_matched_requests=matched_rows,stages={k:distribution(v) for k,v in out.items()},activation=any('using TensorRT flow-decoder estimator' in l for l in log.splitlines()))
def union(iv): 
    end=-1;total=0
    for a,b in sorted(iv):
        if b>end:total+=b-max(a,end);end=b
    return total
def trace(path):
    with gzip.open(path,'rt') as f:events=[e for e in json.load(f)['traceEvents'] if e.get('ph')=='X']
    names=('cv3_flow_mel','cv3_hift_stream','cosyvoice3_cfm_forward_estimator','cosyvoice3_cfm_euler_10_steps')
    scopes=defaultdict(list)
    for e in events:
        if e['name'] in names and e.get('cat')=='user_annotation':scopes[e['name'],e['tid']].append(e)
    launches={}
    for e in events:
        if e.get('cat') in ('cuda_runtime','cuda_driver') and 'correlation' in e.get('args',{}):
            launches[e['args']['correlation']]=e
    device=[e for e in events if e.get('cat') in ('kernel','gpu_memcpy','gpu_memset')]
    gpu_scopes=defaultdict(list);samples=defaultdict(list)
    ordered_steps=[];kernels=defaultdict(lambda:[0,0.0]);matched_vars=[]
    for (name,tid),vals in scopes.items():
        vals.sort(key=lambda x:x['ts']);starts=[x['ts'] for x in vals];iv=defaultdict(list)
        for event in device:
            cpu=launches.get(event.get('args',{}).get('correlation'))
            if cpu is None or cpu['tid']!=tid:continue
            pos=bisect.bisect_right(starts,cpu['ts'])-1
            if pos>=0 and cpu['ts']<=starts[pos]+vals[pos]['dur']:
                span=(event['ts'],event['ts']+event['dur']);gpu_scopes[name].append(span);iv[pos].append(span)
                if name=='cosyvoice3_cfm_forward_estimator':
                    row=kernels[event['name']];row[0]+=1;row[1]+=event['dur']/1000
        for i,e in enumerate(vals):
            intervals=iv[i]
            if name=='cosyvoice3_cfm_forward_estimator' and intervals: ordered_steps.append((e['ts'],min(a for a,b in intervals),max(b for a,b in intervals)))
            samples[name].append(dict(cpu_ms=e['dur']/1000,gpu_ms=union(intervals)/1000,activities=len(intervals),span_ms=(max(b for a,b in intervals)-min(a for a,b in intervals))/1000 if intervals else 0))
    ordered_steps.sort()
    gaps=[]
    for (name,_),solves in scopes.items():
        if name!='cosyvoice3_cfm_euler_10_steps':continue
        for g in solves:
            steps=[(a,b) for ts,a,b in ordered_steps if g['ts']<=ts<g['ts']+g['dur']]
            gaps.extend((c-b)/1000 for (a,b),(c,d) in zip(steps,steps[1:]))
    return dict(estimator_top_kernels=sorted([[k,*v] for k,v in kernels.items()],key=lambda x:-x[2])[:10],interstep_device_gap_ms=distribution(gaps),file=path.name,device_events=len(device),correlated=sum(e.get('args',{}).get('correlation') in launches for e in device),stages={name:{col:distribution([r[col] for r in rows]) for col in rows[0]} for name,rows in samples.items()})
if __name__=='__main__':
    folder=base/sys.argv[1]
    result=perf(folder)
    for p in folder.glob('traces/*stage1*/*.gz'):result['trace']=trace(p)
    print(json.dumps(result,indent=2))
