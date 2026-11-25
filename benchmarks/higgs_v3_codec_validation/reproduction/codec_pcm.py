"""Compare real codec PCM on identical codes, graph hits, misses, trims and invalid peers."""
import hashlib
import json
from pathlib import Path
import types

import torch
from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import HiggsAudioV3Code2Wav
from vllm_omni.transformers_utils.configs.higgs_audio_v3 import HiggsAudioV3Config
from importlib.machinery import SourceFileLoader

root=Path('/data/higgs3-codec-95513')
path='/data/models/higgs-v3-tts-4b'
baseline=SourceFileLoader('higgs3_baseline_codec',str(root/'source/vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_code2wav.py')).load_module()
cfg=HiggsAudioV3Config.from_pretrained(path)
cfg.codec_cuda_graph=True
cfg.codec_graph_single_max_frames=5
cfg.codec_graph_batch_sizes=[2]
cfg.codec_graph_frame_sizes=[5]
model=HiggsAudioV3Code2Wav(config=cfg).eval()
model._model_path=path
model._ensure_codec_loaded()
model.capture_auxiliary_graphs()
assert (2,5) in model._decode_graphs
g=torch.Generator(device='cuda').manual_seed(88)
records=[]
with torch.inference_mode():
    for batch,frames,invalid in [(1,5,False),(2,5,False),(3,29,False),(2,5,True)]:
        codes=torch.randint(0,1024,(batch,8,frames),generator=g,device='cuda')
        if invalid: codes[0,0,0]=1024
        group=[dict(meta=dict(left_context_size=1,right_holdback_size=1)) for _ in range(batch)]
        wavs=[]
        for validate in [baseline.HiggsAudioV3Code2Wav._validate_codes,HiggsAudioV3Code2Wav._validate_codes]:
            model._validate_codes=types.MethodType(validate,model)
            result=model(input_ids=codes.flatten(),seq_token_counts=[frames*8]*batch,runtime_additional_information=group)
            wavs.append([v.detach().cpu().contiguous() for v in result.multimodal_outputs['model_outputs']])
        for row,(a,b) in enumerate(zip(*wavs)):
            assert torch.equal(a,b)
            assert a.numel()==(0 if invalid and row==0 else (frames-2)*960)
            records.append(dict(batch=batch,frames=frames,row=row,invalid=invalid and row==0,samples=a.numel(),sha256=hashlib.sha256(a.numpy().tobytes()).hexdigest()))
(root/'pcm-control.json').write_text(json.dumps(records,indent=2))
print('PCM controls passed',len(records))
