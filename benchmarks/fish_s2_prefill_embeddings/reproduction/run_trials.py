from pathlib import Path
import subprocess as sp
import sys,time,os
w=Path(__file__).resolve().parent
os.chdir(w)
r=w.parent
base=os.environ.get('FISH_BASE','base-a')
deadline=time.monotonic()+1500
while not (w/base/'server.log').exists() or sp.check_output(['sudo','-n','docker','inspect','-f','{{.State.Running}}','s2-mrv2-prefill-'+base],text=True).strip()=='true':
    if time.monotonic()>deadline: raise RuntimeError('baseline timeout')
    time.sleep(3)
assert (r/'results'/('s2-mrv2-prefill-'+base)/'result.json').exists(), 'baseline failed'
for arm,options in [('cand-b',['--candidate']),('cand-b2',['--candidate']),('base-a2',[]),('diag-a',['--profile']),('diag-b',['--profile','--candidate'])]:
    print('START',arm,flush=True)
    sp.run([sys.executable,'-u','trial.py',arm,*options],check=True)
assert not sp.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
runner=['sudo','-n','docker','run','--rm','--gpus','all','--network','none','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/data','-e','HF_HUB_OFFLINE=1','-e','PYTHONPATH=/opt/vllm-omni','-e','OMP_NUM_THREADS=4','-v',f'{r}:/data','-v',f'{w}/candidate:/opt/vllm-omni:ro','-w','/opt/vllm-omni','--entrypoint','/data/venv-vllm/bin/python','tts-runtime:20261117-pinned']
sp.run(runner+['/data/fish-prefill-mrv2-95988/controls.py'],check=True)
sp.run(runner+['-m','pytest','-q','-p','no:cacheprovider','tests/model_executor/models/fish_speech/test_fish_speech_prompt_utils.py','tests/model_executor/models/fish_speech/test_fish_speech_regressions.py'],check=True)
print('COMPLETE',flush=True)
