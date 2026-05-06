# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trigonometric scoring for TriAttention KV cache compression.

Scores cached keys by predicted future attention importance using a trigonometric
series derived from pre-RoPE Q/K concentration. See arXiv:2604.04921.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .rope_utils import to_complex_pairs

__all__ = [
    "HeadFrequencyStats",
    "compute_frequency_statistics_from_means",
    "score_keys_for_round",
]


@dataclass
class HeadFrequencyStats:
    """Per-head calibration statistics in frequency domain."""

    q_mean_complex: torch.Tensor  # (freq_count,) complex64
    q_abs_mean: torch.Tensor  # (freq_count,) float32


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
    disable_mlr: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute amplitude, phase, and MLR extra term from Q/K frequency statistics.

    Args:
        q_mean_complex: Mean of Q in complex frequency domain, shape (freq_count,).
        q_abs_mean: Mean of |Q| in frequency domain, shape (freq_count,).
        k_unrot: Unrotated key vectors, shape (num_keys, head_dim).
        style: RoPE pairing style.
        disable_mlr: If True, use q_abs_mean directly instead of (q_abs_mean - |q_mean|).

    Returns:
        amp: Amplitude, shape (num_keys, freq_count).
        phi: Phase, shape (num_keys, freq_count).
        extra: MLR extra term, shape (num_keys, freq_count).
    """
    k_complex = to_complex_pairs(k_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    if disable_mlr:
        extra = q_abs_mean.unsqueeze(0) * k_abs
    else:
        extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_trig: bool = False,
) -> torch.Tensor:
    """Score cached keys for a single pruning round.

    Evaluates the trigonometric importance formula over multiple future offsets
    and aggregates scores.

    Args:
        key_indices: Position indices of cached keys, shape (num_keys,).
        round_start: Current generation position.
        amp: Amplitude per key per frequency, shape (num_keys, freq_count).
        phi: Phase per key per frequency, shape (num_keys, freq_count).
        omega: RoPE frequencies (inv_freq), shape (freq_count,).
        extra: MLR extra term, shape (num_keys, freq_count).
        offsets: Geometric offsets for future distance sampling, shape (num_offsets,).
        aggregation: 'mean' or 'max' over offsets.
        freq_scale_sq: Per-frequency scaling weights, shape (freq_count,).
        disable_trig: If True, use only the additive (MLR) term.

    Returns:
        Importance scores, shape (num_keys,). Higher = more important.
    """
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)  # (num_keys, num_offsets)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)

    cos_phase = torch.cos(phase)
    scale = freq_scale_sq.view(1, 1, -1)
    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)

    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = additive if disable_trig else (base_scores + additive)

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values
