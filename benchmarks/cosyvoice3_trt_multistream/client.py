from pathlib import Path
import argparse, base64, concurrent.futures, json, time, urllib.request
p=argparse.ArgumentParser()
p.add_argument("--url",required=True);p.add_argument("--ref",type=Path,required=True);p.add_argument("--ref-text",required=True);p.add_argument("--seed",type=int)
a=p.parse_args()
texts=json.loads(Path(__file__).with_name("corpus.json").read_text())
ref="data:audio/wav;base64,"+base64.b64encode(a.ref.read_bytes()).decode()
def request(text):
    body=dict(model="cv3",input=text,ref_audio=ref,ref_text=a.ref_text,response_format="pcm",stream=True,stream_format="audio")
    if a.seed is not None: body["seed"]=a.seed
    start=time.perf_counter(); first=None; size=0
    req=urllib.request.Request(a.url.rstrip("/")+"/v1/audio/speech",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=1200) as response:
        while chunk:=response.read1(65536):
            if first is None: first=time.perf_counter()-start
            size+=len(chunk)
    if not size: raise RuntimeError("empty PCM")
    elapsed=time.perf_counter()-start
    return dict(first_s=first,elapsed_s=elapsed,audio_s=size/48000,rtf=elapsed/(size/48000))
start=time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(4) as pool: rows=list(pool.map(request,texts))
print(json.dumps(dict(rows=rows,throughput=sum(r["audio_s"] for r in rows)/(time.perf_counter()-start)),indent=2))
