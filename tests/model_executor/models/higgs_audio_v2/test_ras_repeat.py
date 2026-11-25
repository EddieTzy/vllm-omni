# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.higgs_audio_v2.ras_repeat import repeated_codebooks

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1),
]


def _resample(codes, histories, logits, batched):
    rows = repeated_codebooks(codes, histories, 7, 2) if batched else None
    for i, history in enumerate(histories):
        if batched:
            qs = dict(rows).get(i)
            if not qs:
                continue
            indices = torch.tensor(qs, dtype=torch.long, device=codes.device)
        else:
            if not isinstance(history, torch.Tensor) or history.shape[-1] < 7:
                continue
            counts = (history[:, -7:] == codes[i].unsqueeze(-1)).sum(-1)
            indices = torch.nonzero(counts >= 2, as_tuple=False).reshape(-1)
            if indices.numel() == 0:
                continue
        probs = logits[i, indices].float().softmax(-1)
        codes[i, indices] = torch.multinomial(probs, 1).squeeze(-1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n", [0, 1, 32])
@pytest.mark.parametrize("mode", ["none", "short", "mixed", "repeat", "no-repeat"])
def test_repeat_rows_preserve_resampling_and_rng(device, n, mode):
    if device == "cuda" and not current_platform.is_cuda():
        pytest.skip("CUDA required")
    torch.manual_seed(11)
    codes = torch.randint(0, 1024, (n, 8), device=device)
    histories: list[torch.Tensor | None] = []
    expected = []
    for i in range(n):
        if mode == "none" or (mode == "mixed" and i % 3 == 0):
            histories.append(None)
            continue
        length = 6 if mode == "short" or (mode == "mixed" and i % 3 == 1) else 70 + i
        history = torch.randint(0, 1024, (8, length), device=device)
        if mode == "repeat":
            history[::2, -2:] = codes[i, ::2, None]
        if mode == "no-repeat":
            history[:, -7:] = (codes[i, :, None] + 1) % 1024
        histories.append(history)
        if length >= 7:
            counts = (history[:, -7:] == codes[i, :, None]).sum(-1)
            qs = torch.nonzero(counts >= 2).flatten().tolist()
            if qs:
                expected.append((i, qs))
    assert repeated_codebooks(codes, histories, 7, 2) == expected
    logits = torch.randn(n, 8, 1026, device=device, dtype=torch.bfloat16)
    a, b = codes.clone(), codes.clone()
    torch.manual_seed(2026)
    _resample(a, histories, logits, False)
    rng_a = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
    torch.manual_seed(2026)
    _resample(b, histories, logits, True)
    rng_b = torch.get_rng_state() if device == "cpu" else torch.cuda.get_rng_state()
    assert torch.equal(a, b)
    assert torch.equal(rng_a, rng_b)
