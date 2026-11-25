# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.higgs_audio_v2.delay_state import update_audio_codes

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA state kernel"),
]


def _reference(codes, states, bos, eos):
    next_states = []
    codes = codes.clone()
    q = codes.shape[1]
    for i, (delay, remaining) in enumerate(states):
        if delay + 1 < q:
            codes[i, delay + 1 :] = bos
            delay += 1
        if remaining >= 0:
            codes[i, : q - remaining] = eos
            remaining -= 1
            terminate = remaining <= 0
        else:
            positions = torch.nonzero(codes[i] == eos).flatten()
            if positions.numel():
                last = int(positions[-1])
                codes[i, : last + 1] = eos
                remaining = q - last - 1
            terminate = remaining == 0
        if terminate:
            delay, remaining = 0, -1
        next_states.append([delay, remaining, int(terminate)])
    return codes, next_states


@pytest.mark.parametrize("q", [1, 2, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_fused_transition_all_delay_and_eos_boundaries(q, strided):
    bos, eos = 1024, 1025
    states = [(d, r) for d in range(q + 1) for r in range(-1, q + 1) for _ in range(q + 2)]
    raw = torch.zeros(len(states), q * (2 if strided else 1), device="cuda", dtype=torch.long)
    codes = raw[:, ::2] if strided else raw
    for i in range(len(states)):
        pattern = i % (q + 2)
        if pattern < q:
            codes[i, pattern] = eos
        elif pattern == q:
            codes[i] = eos
    saved = codes.clone()
    expected, next_expected = _reference(codes.cpu(), states, bos, eos)
    result, next_result = update_audio_codes(codes, states, bos, eos)
    torch.testing.assert_close(result.cpu(), expected, atol=0, rtol=0)
    assert next_result == next_expected
    torch.testing.assert_close(codes, saved, atol=0, rtol=0)


def test_empty_state_rows():
    codes, states = update_audio_codes(torch.empty(0, 8, device="cuda", dtype=torch.long), [], 1024, 1025)
    assert codes.shape == (0, 8)
    assert states == []
