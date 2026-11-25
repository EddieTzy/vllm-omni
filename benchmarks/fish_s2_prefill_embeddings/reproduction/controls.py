"""Run the original and candidate methods on the same deterministic GPU inputs."""
import ast
import json
from pathlib import Path
from typing import Any

import torch

root = Path('/data/fish-prefill-mrv2-95988')
relative = Path('vllm_omni/model_executor/models/fish_speech/fish_speech_slow_ar.py')
methods = []
for side in ('source','candidate'):
    cls = next(node for node in ast.parse((root/side/relative).read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'FishSpeechSlowARForConditionalGeneration')
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'preprocess')
    ns = {'torch':torch,'Any':Any}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(relative),'exec'),ns)
    methods.append(ns['preprocess'])

class Fixture:
    _audio_pad_token_id = 9
    _num_codebooks = 10
    def __init__(self, length, dtype):
        self.embedding = torch.arange(length*40, device='cuda').reshape(length,40).to(dtype)
    def _build_structured_voice_clone_prefill_embeds(self, info):
        return self.embedding
    def _build_prefill_embeds(self,ids,info):
        return self.embedding
    def embed_input_ids(self,ids):
        return torch.zeros((*ids.shape,40), device=ids.device, dtype=self.embedding.dtype)

def same(a,b):
    if isinstance(a,torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and a.device == b.device and torch.equal(a,b)
        if a.device.type=='cpu': assert a.is_pinned()==b.is_pinned()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a: same(a[k],b[k])
    else: assert a==b

checks=[]
for dtype in (torch.bfloat16,torch.float32):
    for clone in (False,True):
        for spans in [(17,), (5,7,5)]:
            obj=Fixture(sum(spans),dtype)
            infos=[{},{}]
            for span in spans:
                ids=torch.ones(span,dtype=torch.long,device='cuda')
                old,new=[fn(obj,ids,None,fish_structured_voice_clone=clone,**info) for fn,info in zip(methods,infos)]
                for x,y in zip(old,new): same(x,y)
                infos=[old[2],new[2]]
            checks.append(dict(dtype=str(dtype),clone=clone,spans=spans))
(root/'controls.json').write_text(json.dumps(checks,indent=2))
print(len(checks),'matched')
