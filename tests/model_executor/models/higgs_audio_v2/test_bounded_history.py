# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v2.higgs_audio_v2_talker import (
    HiggsAudioV2TalkerForConditionalGeneration,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("frames", [6, 7, 200])
def test_dispatch_retains_exact_ras_window_and_emits_every_frame(frames):
    model = object.__new__(HiggsAudioV2TalkerForConditionalGeneration)
    model.config = SimpleNamespace(
        audio_token_id=4,
        audio_delay_token_id=9,
        audio_stream_bos_id=14,
        audio_stream_eos_id=15,
        num_codebooks=2,
        codebook_size=16,
    )
    model.__dict__["audio_codebook_logits"] = lambda hidden, mask: torch.zeros((int(mask.sum()), 2, 16))
    hidden = torch.ones(1, 2)
    emitted = []
    for step in range(frames):
        value = step % 13
        model.__dict__["_sample_audio_codes_upstream"] = lambda logits, **_: torch.tensor([value, value])
        output = SimpleNamespace(sampled_token_ids=torch.tensor([[4]]), num_sampled=None)
        model._dispatch_audio_codes(output, hidden, None, request_ids=["live"])
        emitted.append(model._last_audio_codes[0].clone())
        history = model._audio_state["live"]["audio_out_ids"]
        expected = torch.stack(emitted[-7:], dim=1)
        torch.testing.assert_close(history, expected, atol=0, rtol=0)
        assert history.shape[-1] <= 7
        assert history.untyped_storage().nbytes() <= 2 * 7 * history.element_size()
    assert len(emitted) == frames
    model.on_requests_finished({"live"})
    assert model._audio_state == {}
