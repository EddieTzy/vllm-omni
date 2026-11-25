"""Exercise the exact baseline/candidate validation methods on identical CUDA codes."""
import ast
import json
from pathlib import Path
import statistics
import time
import torch

root=Path('/data/higgs3-codec-95513')
rel='vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py'
def validator(src):
    cls=next(c for c in ast.parse((src/rel).read_text()).body if isinstance(c,ast.ClassDef) and c.name=='HiggsAudioV3Code2Wav')
    fn=next(f for f in cls.body if isinstance(f,ast.FunctionDef) and f.name=='_validate_codes')
    env={'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(src/rel),'exec'),env)
    return env[fn.name]
a,b=[validator(root/p) for p in ('source','candidate')]
class Codec:
    num_codebooks=8
    num_real_codes=1024
codec=Codec()
g=torch.Generator(device='cuda').manual_seed(318)
results=[]
for batch,frames in [(1,5),(2,5),(8,5),(16,54),(3,29),(1,0)]:
    raw=torch.randint(0,1024,(batch,8,frames*2),dtype=torch.long,device='cuda',generator=g)
    cases=[raw[:,:,::2],raw[:,:,::2].contiguous()]
    for codes in cases:
        for fun in [a,b]:
            assert fun(codec,codes) is codes
        if frames:
            for low,high in [(-1,1023),(0,1024),(-7,1034)]:
                bad=codes.clone()
                bad[0,0,0]=low
                bad[0,1,0]=high
                errors=[]
                for fun in [a,b]:
                    try: fun(codec,bad)
                    except ValueError as exc: errors.append(str(exc))
                assert len(errors)==2 and errors[0]==errors[1]
    if not frames: continue
    codes=cases[-1]
    times={'A':[],'B':[]}
    for phase,fun in [('A',a),('B',b),('B',b),('A',a)]:
        for i in range(30): fun(codec,codes)
        for i in range(160):
            start=time.perf_counter_ns()
            fun(codec,codes)
            times[phase].append((time.perf_counter_ns()-start)/1000)
    results.append(dict(batch=batch,frames=frames,median_us={k:statistics.median(t) for k,t in times.items()}))
for codes in [torch.zeros((8,5),device='cuda'),torch.zeros((1,7,5),device='cuda')]:
    errors=[]
    for fun in [a,b]:
        try: fun(codec,codes)
        except ValueError as exc: errors.append(str(exc))
    assert len(errors)==2 and errors[0]==errors[1]
(root/'control.json').write_text(json.dumps({'component_only':True,'shapes':results,'valid_invalid_controls':'passed'},indent=2))
print(results)
