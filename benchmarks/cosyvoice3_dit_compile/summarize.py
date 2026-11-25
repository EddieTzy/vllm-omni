from pathlib import Path
import json,re,statistics,sys,math
def mean(seq):return statistics.mean(seq)
def q(values,p):
    v=sorted(values);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def summarize(d):
    log=re.sub(r'\x1b\[[0-9;]*m','',(d/'server.log').read_text(errors='replace'))
    ends={rid:int(ts) for rid,ts in re.findall(r'CV3END\s+(\S+)\s+(\d+)',log)}
    flows=[(int(a),int(b),int(k)) for a,b,k in re.findall(r'CV3FLOW\s+(\d+)\s+(\d+)\s+(\d+)',log)]
    tails=[];fs=[];rows=[];walls=[];steps=[]
    for path in sorted(d.glob('measure*.json')):
        w=json.loads(path.read_text());batch=w['rows'];t1=max(r['last'] for r in batch);t0=t1-int(w['wall']*1e9)
        walls.append(w['wall']);rows.extend(batch)
        for row in batch:
            rid=row['id'];hits=[v for k,v in ends.items() if k==rid or k.startswith(rid+'-')]
            if len(hits)==1:tails.append((row['last']-hits[0])/1e9)
        for a,b,k in flows:
            if t0<=a<=b<=t1:fs.append((b-a)/1e6);steps.append(k)
    assert rows
    return dict(requests=len(rows),matched_tails=len(tails),flow_calls=len(fs),throughput=sum(r['audio'] for r in rows)/sum(walls),mean_e2e=mean(r['e2e'] for r in rows),mean_rtf=mean(r['e2e']/r['audio'] for r in rows),first_p95=q([r['first'] for r in rows],.95),mean_tail=mean(tails) if tails else None,mean_flow_host_ms=mean(fs) if fs else None,amortized_ms_per_step=sum(fs)/sum(steps) if steps else None,audio_s=sum(r['audio'] for r in rows),wave_throughputs=[sum(r['audio'] for r in rows[i*4:i*4+4])/w for i,w in enumerate(walls)])
if __name__=='__main__':
    base=Path(sys.argv[1])
    print(json.dumps({d.name:summarize(d) for d in sorted(base.iterdir()) if d.is_dir() and (d/'server.log').exists() and (d/'measure0.json').exists()},indent=2))
