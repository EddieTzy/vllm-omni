# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""In-place LM audio-mode forcing with a CPU-owned per-row decision table."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _force_audio_mode(
    logits, decisions, first, v: tl.constexpr, s0: tl.constexpr, s1: tl.constexpr, block: tl.constexpr
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    target = tl.load(decisions + row * 2)
    if block == 0:
        tl.store(first + row, tl.load(decisions + row * 2 + 1) != 0)
    if target != -1:
        keep = tl.load(logits + row * s0 + target * s1, mask=(target >= 0) & (target < v), other=-float("inf"))
        indices = block * block + tl.arange(0, block)
        values = tl.where(indices == target, keep, -float("inf"))
        tl.store(logits + row * s0 + indices * s1, values, mask=indices < v)


def mask_audio_mode_(logits: torch.Tensor, decisions: list[tuple[int, bool]]) -> torch.Tensor:
    """Target -1 leaves a row unchanged; -2 forces all logits to -inf."""
    first = torch.empty(logits.shape[0], dtype=torch.bool, device=logits.device)
    if not decisions:
        return first
    table = torch.tensor(decisions, dtype=torch.int32, device=logits.device)
    _force_audio_mode[(logits.shape[0], max(1, triton.cdiv(logits.shape[1], 1024)))](
        logits, table, first, logits.shape[1], *logits.stride(), 1024
    )
    return first
