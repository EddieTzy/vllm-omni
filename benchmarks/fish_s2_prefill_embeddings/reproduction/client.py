"""Bounded streaming screening, not a full quality or acceptance evaluation."""
import argparse
import base64
import asyncio
import json
import pathlib
import time

import httpx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--port', type=int, required=True)
parser.add_argument('--label', required=True)
parser.add_argument('--concurrency',type=int)
parser.add_argument('--count',type=int)
parser.add_argument('--corpus',type=pathlib.Path)
parser.add_argument('--omit-request-seed',action='store_true')
parser.add_argument('--request-timeout',type=float,default=180)
arguments = parser.parse_args()
output = pathlib.Path('/data/results') / arguments.label
output.mkdir(exist_ok=False)
texts = [
    'The library opens at nine in the morning. Please bring your membership card.',
    'We can discuss the delivery schedule tomorrow after the team meeting.',
    'A gentle breeze moved through the garden while the children played nearby.',
    '图书馆每天上午九点开放，请记得携带你的借阅卡。',
    '我们可以在明天的团队会议结束后，讨论具体的送货时间。',
    '微风轻轻吹过花园，孩子们正在旁边开心地玩耍。',
]

if arguments.corpus:
    texts=json.loads(arguments.corpus.read_text())['texts']
cases=[(arguments.concurrency,arguments.count)] if arguments.concurrency else [(1,8),(8,16)]
assert all(count and count%concurrency==0 for concurrency,count in cases)


ref = pathlib.Path('/opt/vllm-omni/tests/assets/qwen3_tts/clone_2.wav').read_bytes()
assert ref.startswith(b'RIFF')
ref_data = 'data:audio/wav;base64,' + base64.b64encode(ref).decode()
ref_text = 'Okay. Yeah. I resent you. I love you. I respect you. But you know what? You blew it! And thanks to you.'

async def request_audio(client, request_number, warmup, concurrency):
    payload = dict(model='/data/models/s2-pro', ref_audio=ref_data, ref_text=ref_text,
                   input=texts[request_number % len(texts)], seed=42,
                   response_format='pcm', stream=True,stream_format='audio')
    if arguments.omit_request_seed:
        payload.pop('seed')
    start = time.perf_counter()
    first_audio_seconds = None
    blocks = []
    chunk_arrivals = []
    async with client.stream('POST', f'http://127.0.0.1:{arguments.port}/v1/audio/speech',
                             json=payload) as response:
        if response.is_error:
            raise RuntimeError((await response.aread())[:1024])
        assert response.headers.get("content-type", "").startswith("audio/"), response.headers
        async for block in response.aiter_bytes():
            if block and first_audio_seconds is None:
                first_audio_seconds = time.perf_counter() - start
            blocks.append(block)
            if block:
                chunk_arrivals.append(dict(seconds=time.perf_counter()-start,bytes=len(block)))
    seconds = time.perf_counter() - start
    pcm = b''.join(blocks)
    assert pcm and len(pcm) % 2 == 0
    if not warmup:
        (output / f'c{concurrency}-{request_number:03d}.pcm').write_bytes(pcm)
    return dict(request_number=request_number, text=payload['input'], bytes=len(pcm),
                audio_seconds=len(pcm) / 2 / 44100, elapsed_seconds=seconds,
                ttfa_seconds=first_audio_seconds,chunk_arrivals=chunk_arrivals)


async def main():
    rows = []
    summaries = []
    async with httpx.AsyncClient(timeout=arguments.request_timeout, trust_env=False) as client:
        for concurrency, count in cases:
            await asyncio.gather(*(request_audio(client, index, True, concurrency) for index in range(concurrency)))
            start = time.perf_counter()
            measured = []
            for offset in range(0, count, concurrency):
                measured.extend(await asyncio.gather(*(request_audio(client, index, False, concurrency)
                    for index in range(offset, offset + concurrency))))
            wall_seconds = time.perf_counter() - start
            for row in measured:
                row['concurrency'] = concurrency
            rows.extend(measured)
            summaries.append(dict(concurrency=concurrency, requests=count,
                wall_seconds=wall_seconds,
                audio_seconds_per_second=sum(row['audio_seconds'] for row in measured)/wall_seconds,
                ttfa_p50_seconds=float(np.median([row['ttfa_seconds'] for row in measured])),
                ttfa_p95_seconds=float(np.quantile([row['ttfa_seconds'] for row in measured], .95))))
    result = dict(label=arguments.label, summaries=summaries, requests=rows, request_seed=None if arguments.omit_request_seed else 42,
                  limitation='Fixed cohort per boot; no formal quality/noninferiority claim.')
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(dict(label=arguments.label, summaries=summaries)))


asyncio.run(main())
