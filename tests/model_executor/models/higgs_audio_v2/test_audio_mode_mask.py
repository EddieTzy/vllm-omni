# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy

import pytest
import torch
from vllm.platforms import current_platform

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.test_higgs_audio_v2_mrv2 import _Config
from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
    HiggsAudioV2TalkerForConditionalGeneration,
)

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": ["L4", "H100", "B200"]}, num_cards=1),
    pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA audio-mode mask"),
]


def _model(config, states):
    model = object.__new__(HiggsAudioV2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = config
    model._audio_state = copy.deepcopy(states)
    return model


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("vocab", [0, 8, 1026])
@pytest.mark.parametrize("invalid", [False, True])
def test_mrv2_audio_bias_matches_legacy(dtype, strided, vocab, invalid):
    config = _Config(audio_bos_token_id=3, audio_token_id=-3 if invalid else 4, audio_eos_token_id=-4 if invalid else 5)
    previous = [config.audio_token_id, 3, 1, 9, 3, config.audio_token_id]
    request_ids = ["a", "a", "b", "c", "d", "e"]
    states = {key: {"should_terminate": key in ("a", "d")} for key in request_ids}
    legacy = _model(config, states)
    candidate = _model(config, states)
    raw = torch.randn((len(previous), vocab * (2 if strided else 1)), dtype=dtype, device="cuda")
    if vocab:
        raw[0, 0] = float("nan")
        raw[-1, -1] = float("inf")
    logits = raw[:, ::2] if strided else raw
    expected = logits.cpu()
    padding = raw[:, 1::2].cpu() if strided else None
    legacy._apply_audio_mode_bias(expected, None, request_ids=request_ids, row_prev_tokens=previous)
    pointer = logits.data_ptr()
    candidate._apply_audio_mode_bias(logits, None, request_ids=request_ids, row_prev_tokens=previous)
    torch.testing.assert_close(logits.cpu(), expected, atol=0, rtol=0, equal_nan=True)
    torch.testing.assert_close(candidate._last_first_audio_after_bos.cpu(), legacy._last_first_audio_after_bos)
    assert candidate._audio_state == legacy._audio_state
    assert logits.data_ptr() == pointer
    if padding is not None:
        torch.testing.assert_close(raw[:, 1::2].cpu(), padding, atol=0, rtol=0, equal_nan=True)


def test_empty_rows():
    model = _model(_Config(audio_bos_token_id=3, audio_token_id=4, audio_eos_token_id=5), {})
    model._apply_audio_mode_bias(torch.empty(0, 8, device="cuda"), None, request_ids=[], row_prev_tokens=[])
    assert model._last_first_audio_after_bos.numel() == 0
