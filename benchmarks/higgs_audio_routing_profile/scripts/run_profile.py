"""Run private C32 A/B/A/B stage timing and sampled routing traces."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/home/tzy/zyb-fish-pr')
WORK = ROOT / 'higgs-routing-profile'
OUT = ROOT / 'results/higgs-routing-profile'
OUT.mkdir(exist_ok=False)
REL = Path('vllm_omni/model_executor/models/higgs_audio_v2')
TALKER = REL / 'higgs_audio_v2_talker.py'
CODEC = REL / 'higgs_audio_v2_code2wav.py'
PATCH = '''        # Resolve uniform audio routing once instead of indexing in every layer.
        if audio_mask is not None and audio_mask.all().item():
            audio_mask = None
'''
original = (ROOT / 'vllm-source' / TALKER).read_text()
candidate = (ROOT / 'vllm-higgs-candidate' / TALKER).read_text()
assert candidate.count(PATCH) == 1 and candidate.replace(PATCH, '') == original
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(scope='Private instrumentation; diagnostic runs, not replacement serving benchmark',
                order=['A', 'B', 'A2', 'B2'],
                source_baseline='4dafb464442956996be9633ed036cd24ebfef1d0',
                baseline_talker_sha256=sha(ROOT / 'vllm-source' / TALKER),
                candidate_talker_sha256=sha(ROOT / 'vllm-higgs-candidate' / TALKER),
                instrumentation_sha256=sha(WORK / 'profile_support.py'),
                config_sha256=sha(ROOT / 'higgs-routing/c32.yaml'),
                corpus_sha256=sha(ROOT / 'higgs-routing/public-corpus.json'),
                concurrency=32, warmup_requests=32, measured_requests=32)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
for label, source in [('original', 'vllm-source'), ('patched', 'vllm-higgs-candidate')]:
    target = ROOT / ('vllm-routing-profile-' + label)
    assert not target.exists()
    subprocess.run(['cp', '-a', '--reflink=auto', str(ROOT / source), str(target)], check=True)
    for rel, suffix in [(TALKER, '\ninstrument_talker(HiggsAudioV2TalkerForConditionalGeneration, HiggsAudioV2DecoderLayer)\n'),
                        (CODEC, '\ninstrument_codec(HiggsAudioV2Code2Wav)\n')]:
        with (target / rel).open('a') as stream:
            stream.write('\n# Private profiling instrumentation, identical in both arms.\n'
                         'import sys as _profile_sys\n'
                         '_profile_sys.path.insert(0, "/data/higgs-routing-profile")\n'
                         'from profile_support import instrument_talker, instrument_codec\n' + suffix)
launcher = (ROOT / 'higgs-routing/run-server.sh').read_text()
assert launcher.count('-e OMP_NUM_THREADS=4') == 1
launcher = launcher.replace('-e OMP_NUM_THREADS=4', '-e HIGGS_PROFILE_ARM="$HIGGS_PROFILE_ARM" -e OMP_NUM_THREADS=4')
(WORK / 'run-server.sh').write_text(launcher)
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for index, arm in enumerate(manifest['order']):
    # Refuse to share the GPU with another task.
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not occupied, 'GPU busy: ' + occupied
    name = 'zyb-routing-profile-' + arm.lower()
    source = 'original' if arm.startswith('A') else 'patched'
    port = 18840 + index
    env = dict(os.environ, HIGGS_CONTAINER_NAME=name,
               HIGGS_SOURCE_DIR=str(ROOT / ('vllm-routing-profile-' + source)),
               HIGGS_PORT=str(port), HIGGS_DEPLOY_CONFIG='/data/higgs-routing/c32.yaml',
               HIGGS_PROFILE_ARM=arm)
    started = False
    try:
        print('BOOT', arm, flush=True)
        with (OUT / f'{arm}-launch.log').open('w') as log:
            subprocess.run(['bash', str(WORK / 'run-server.sh')], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        started = True
        deadline = time.monotonic() + 300
        while True:
            try:
                with http.open(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            state = subprocess.check_output(['sudo', '-n', 'docker', 'inspect', '-f', '{{.State.Running}}', name], text=True).strip()
            if state != 'true' or time.monotonic() > deadline:
                raise RuntimeError('Server unhealthy: ' + state)
            time.sleep(1)
        print('PROFILE', arm, flush=True)
        with (OUT / f'{arm}-client.log').open('w') as log:
            subprocess.run(['sudo', '-n', 'docker', 'exec', 'zyb-fish-pr-prepare',
                            '/data/venv-vllm/bin/python', '/data/higgs-routing/screen.py',
                            '--port', str(port), '--label', 'higgs-routing-profile-client-' + arm,
                            '--concurrency', '32', '--count', '32',
                            '--corpus', '/data/higgs-routing/public-corpus.json',
                            '--omit-request-seed', '--request-timeout', '360'],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        print('DONE', arm, flush=True)
    finally:
        if started:
            subprocess.run(['sudo', '-n', 'docker', 'stop', '--time', '30', name], check=True)
            with (OUT / f'{arm}-server.log').open('w') as log:
                subprocess.run(['sudo', '-n', 'docker', 'logs', name], stdout=log, stderr=subprocess.STDOUT, check=True)
print('PROFILE_COMPLETE', flush=True)
