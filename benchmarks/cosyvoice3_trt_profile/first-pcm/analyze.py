"""Join measured first blocks to existing TRT timelines, without a new GPU run."""
import gzip
import json
import re
from pathlib import Path
from statistics import mean

root = Path(__file__).resolve().parent / "inputs"
def analyze(label):
    folder = root / label
    log = (folder / "time-markers.txt").read_text(errors="replace")
    log = re.sub(r"\x1b\[[0-9;]*m","",log)
    log = re.sub(r"^(CV3READY \S+ \d+ )(CV3BEGIN \S+ \d+ \d+ \d+)\n(\d+ \d+)$",r"\1\3\n\2",log,flags=re.M)
    log = re.sub(r"^(CV3END \S+) CV3BEGIN(\d+) *\n(speech-\S+ \d+ \d+ \d+)$",r"\1 \2\nCV3BEGIN \3",log,flags=re.M)
    def times(marker):
        return sorted((int(t),rid,int(off),int(n)) for rid,off,n,t in re.findall(marker+r" (\S+) (\d+) (\d+) (\d+)",log))
    begins = times("CV3BEGIN")
    flows = sorted((int(a),int(b)) for a,b in re.findall(r"CV3FLOW (\d+) (\d+) \d+",log))
    rows = {r["id"]:r for file in sorted(folder.glob("measure*.json")) for r in json.loads(file.read_text())["rows"]}
    blocks = []
    for i,(ts,rid,off,n) in enumerate(begins):
        prefix = next((k for k in rows if rid==k or rid.startswith(k+"-")),None)
        if prefix is None: continue
        upper = begins[i+1][0] if i+1<len(begins) else float('inf')
        fs = [(a,b) for a,b in flows if ts<=a<b<upper]
        if len(fs)!=1: continue
        blocks.append(dict(id=prefix,offset=off,n=n,begin=ts,start=fs[0][0],end=fs[0][1]))
    if label=="unseedtrace":
        path=root.parent.parent / "traces/unseeded-stage1.json.gz"
        events=[e for e in json.load(gzip.open(path))["traceEvents"] if e.get("cat")=="user_annotation" and e.get("ph")=="X"]
        scope=lambda n: sorted((e for e in events if e["name"]==n), key=lambda e:e["ts"])
        flow, hift=scope("cv3_flow_mel"),scope("cv3_hift_stream")
        assert len(blocks)==len(flow)==len(hift)==19
        offsets=[b["begin"]/1000-e["ts"] for b,e in zip(blocks,flow)]
        assert max(offsets)-min(offsets)<2500
        for i,(block,f,h) in enumerate(zip(blocks,flow,hift)):
            assert f["ts"]+f["dur"]<=h["ts"] and (i+1==len(flow) or h["ts"]+h["dur"]<=flow[i+1]["ts"])
            block.update(flow_scope_ms=f["dur"]/1000,hift_scope_ms=h["dur"]/1000)
            batch=[e for e in events if e["name"].startswith("execute_context") and e["ts"]<=f["ts"] and h["ts"]+h["dur"]<=e["ts"]+e["dur"]]
            group=min(batch,key=lambda e:e["dur"])
            block["group"]=group["name"]
            block["group_tail_ms"]=(group["ts"]+group["dur"]-h["ts"]-h["dur"])/1000
            block["after_own_hift_est_ms"]=(h["ts"]+h["dur"]-f["ts"])/1000
    first=[]
    for block in blocks:
        if block["offset"]!=0: continue
        r=rows[block["id"]]; first_pcm=r["last"]+(r["first"]-r["e2e"])*1e9
        vals=dict(id=block["id"],span_start_to_pcm_ms=(first_pcm-block["begin"])/1e6,flow_host_ms=(block["end"]-block["start"])/1e6)
        if "group" in block:
            vals.update(flow_scope_ms=block["flow_scope_ms"],hift_scope_ms=block["hift_scope_ms"],group=block["group"],group_tail_ms=block["group_tail_ms"],post_hift_est_ms=vals["span_start_to_pcm_ms"]-block["after_own_hift_est_ms"])
        first.append(vals)
    return dict(count=len(first),first=first,means={key:mean(v[key] for v in first) for key,value in first[0].items() if isinstance(value,(float,int))},followup_trace_means={key:mean(b[key] for b in blocks if b["offset"]>0) for key in ["flow_scope_ms","hift_scope_ms"]} if label=="unseedtrace" else {})
results={run:analyze(run) for run in ["unseed","unseed2","unseedtrace"]}
Path(__file__).with_name("first-pcm.json").write_text(json.dumps(results,indent=2))
for k,v in results.items(): print(k,v["count"],v["means"],v["followup_trace_means"])
print(results["unseedtrace"]["first"])
