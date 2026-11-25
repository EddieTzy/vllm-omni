# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Fuse per-frame delay/eos transitions while retaining CPU request ownership."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _update(
    codes,
    states,
    output,
    next_states,
    num_codebooks: tl.constexpr,
    s0: tl.constexpr,
    s1: tl.constexpr,
    o0: tl.constexpr,
    o1: tl.constexpr,
    bos: tl.constexpr,
    eos: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    q = tl.arange(0, block)
    delay = tl.load(states + row * 2)
    remaining = tl.load(states + row * 2 + 1)
    values = tl.load(codes + row * s0 + q * s1, mask=q < num_codebooks, other=-1)
    values = tl.where((delay + 1 < num_codebooks) & (q >= delay + 1), bos, values)
    next_delay = tl.where(delay + 1 < num_codebooks, delay + 1, delay)
    last_eos = tl.max(tl.where((q < num_codebooks) & (values == eos), q, -1), 0)
    ramp = remaining >= 0
    values = tl.where(ramp & (q < num_codebooks - remaining), eos, values)
    values = tl.where(~ramp & (last_eos >= 0) & (q <= last_eos), eos, values)
    next_remaining = tl.where(ramp, remaining - 1, tl.where(last_eos >= 0, num_codebooks - last_eos - 1, -1))
    terminate = tl.where(ramp, remaining - 1 <= 0, (last_eos >= 0) & (num_codebooks - last_eos - 1 <= 0))
    next_delay = tl.where(terminate, 0, next_delay)
    next_remaining = tl.where(terminate, -1, next_remaining)
    tl.store(output + row * o0 + q * o1, values, mask=q < num_codebooks)
    tl.store(next_states + row * 3, next_delay)
    tl.store(next_states + row * 3 + 1, next_remaining)
    tl.store(next_states + row * 3 + 2, terminate.to(tl.int32))


def update_audio_codes(
    codes: torch.Tensor, states: list[tuple[int, int]], bos: int, eos: int
) -> tuple[torch.Tensor, list[list[int]]]:
    output = torch.empty_like(codes)
    if not states:
        return output, []
    table = torch.tensor(states, dtype=torch.int32, device=codes.device)
    next_states = torch.empty((codes.shape[0], 3), dtype=torch.int32, device=codes.device)
    _update[(codes.shape[0],)](
        codes,
        table,
        output,
        next_states,
        codes.shape[1],
        *codes.stride(),
        *output.stride(),
        bos,
        eos,
        triton.next_power_of_2(codes.shape[1]),
    )
    return output, next_states.tolist()
