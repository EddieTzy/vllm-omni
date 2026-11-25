# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Batch RAS repeat detection without changing request-wise resampling."""

import torch


def repeated_codebooks(
    codes: torch.Tensor,
    histories: list[torch.Tensor | None],
    window_size: int,
    repeat_threshold: int,
) -> list[tuple[int, list[int]]]:
    eligible = []
    windows = []
    for i, history in enumerate(histories):
        if isinstance(history, torch.Tensor) and history.shape[-1] >= window_size:
            eligible.append(i)
            windows.append(history[:, -window_size:].to(codes.device))
    if not eligible:
        return []
    stacked = torch.stack(windows)
    current = codes if len(eligible) == codes.shape[0] else codes[eligible]
    repeated = ((stacked == current.unsqueeze(-1)).sum(dim=-1) >= repeat_threshold).tolist()
    result = []
    for i, flags in zip(eligible, repeated, strict=True):
        codebooks = [q for q, flag in enumerate(flags) if flag]
        if codebooks:
            result.append((i, codebooks))
    return result
