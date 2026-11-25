"""Run private C32 A/B/A/B stage timing and sampled routing traces."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/home/tzy/zyb-fish-pr')
WORK = ROOT / 'higgs-mask-profile'
OUT = ROOT / 'results/higgs-mask-profile'
OUT.mkdir(exist_ok=False)
REL = Path('vllm_omni/model_executor/models/higgs_audio_v2')
TALKER = REL / 'higgs_audio_v2_talker.py'
CODEC = REL / 'higgs_audio_v2_code2wav.py'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = dict(scope='Private instrumentation; diagnostic runs, not replacement serving benchmark',
                order=['A', 'B', 'B2', 'A2'],
                source_baseline='649e7b3b9a5b93f9a33f753942df149ba939e523',
                baseline_talker_sha256=sha(ROOT / 'vllm-higgs-mask-base' / TALKER),
                candidate_talker_sha256=sha(ROOT / 'vllm-higgs-mask-main' / TALKER),
                instrumentation_sha256=sha(WORK / 'profile_support.py'),
                config_sha256=sha(ROOT / 'higgs-routing/c32.yaml'),
                corpus_sha256=sha(ROOT / 'higgs-routing/public-corpus.json'),
                concurrency=32, warmup_requests=32, measured_requests=32)
(OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2))
for label, source in [('original', 'vllm-higgs-mask-base'), ('patched', 'vllm-higgs-mask-main')]:
    target = ROOT / ('vllm-mask-profile-' + label)
    assert not target.exists()
    subprocess.run(['cp', '-a', '--reflink=auto', str(ROOT / source), str(target)], check=True)
    talker_path = target / TALKER
    text = talker_path.read_text()
    markers = [
        ('        # Emulate upstream', '        # Sample per-codebook', 'mask.delay_pattern'),
        ('        # Sample per-codebook', '        # (PyTorch align): Repetition-Aware', 'audio.sample_codes'),
        ('        # (PyTorch align): Repetition-Aware', '        # Apply the upstream delay pattern', 'audio.ras'),
        ('        # Apply the upstream delay pattern', '        # Reassemble the per-position', 'audio.state_update'),
        ('        # Reassemble the per-position', '\n    @staticmethod\n    def _sample_audio_codes_upstream', 'audio.assemble_output'),
    ]
    for start, stop, label in markers:
        assert text.count(start) == text.count(stop) == 1, (label, start, stop)
        a = text.index(start)
        b = text.index(stop, a)
        block = text[a:b]
        text = text[:a] + '        with _profile_phase("' + label + '", audio_rows=cb_logits.shape[0]):\n' + ''.join('    '+line if line.strip() else line for line in block.splitlines(keepends=True)) + text[b:]
    ast.parse(text)
    talker_path.write_text(text)
    for rel, suffix in [(TALKER, '\ninstrument_talker(HiggsAudioV2TalkerForConditionalGeneration, HiggsAudioV2Sampler)\n'),
                        (CODEC, '\ninstrument_codec(HiggsAudioV2Code2Wav)\n')]:
        with (target / rel).open('a') as stream:
            stream.write('\n# Private profiling instrumentation, identical in both arms.\n'
                         'import sys as _profile_sys\n'
                         '_profile_sys.path.insert(0, "/data/higgs-mask-profile")\n'
                         'from profile_support import instrument_talker, instrument_codec, phase as _profile_phase\n' + suffix)
launcher = (ROOT / 'higgs-routing/run-server.sh').read_text()
assert launcher.count('-e OMP_NUM_THREADS=4') == 1
launcher = launcher.replace('-e OMP_NUM_THREADS=4', '-e HIGGS_PROFILE_ARM="$HIGGS_PROFILE_ARM" -e OMP_NUM_THREADS=4')
(WORK / 'run-server.sh').write_text(launcher)
http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
for index, arm in enumerate(manifest['order']):
    # Refuse to share the GPU with another task.
    occupied = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    assert not occupied, 'GPU busy: ' + occupied
    name = 'zyb-mask-profile-' + arm.lower()
    source = 'original' if arm.startswith('A') else 'patched'
    port = 18860 + index
    env = dict(os.environ, HIGGS_CONTAINER_NAME=name,
               HIGGS_SOURCE_DIR=str(ROOT / ('vllm-mask-profile-' + source)),
               HIGGS_PORT=str(port), HIGGS_DEPLOY_CONFIG='/data/higgs-routing/c32.yaml',
               HIGGS_PROFILE_ARM=arm)
    started = False
    try:
        print('BOOT', arm, flush=True)
        with (OUT / f'{arm}-launch.log').open('w') as log:
            subprocess.run(['bash', str(WORK / 'run-server.sh')], env=env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        started = True
        deadline = time.monotonic() + 900
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
            subprocess.run(['sudo', '-n', 'docker', 'run', '--rm', '--network', 'host', '--user', f'{os.getuid()}:{os.getgid()}', '-v', str(ROOT)+':/data', '--entrypoint', '/data/venv-vllm/bin/python', 'tts-runtime:20261117-pinned', '/data/higgs-routing/screen.py',
                            '--port', str(port), '--label', 'higgs-mask-profile-client-' + arm,
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
