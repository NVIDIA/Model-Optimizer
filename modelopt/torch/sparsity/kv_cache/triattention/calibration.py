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

"""Calibration for TriAttention: compute per-head Q/K frequency statistics.

Hooks into attention layers during a forward pass, captures pre-RoPE Q vectors,
inverts RoPE, converts to frequency domain, and computes per-head mean statistics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .rope_utils import to_complex_pairs
from .scoring import HeadFrequencyStats

__all__ = [
    "CalibrationData",
    "compute_head_stats_from_q",
]


@dataclass
class CalibrationData:
    """Container for TriAttention calibration output.

    Stores per-head frequency statistics computed during calibration, along with
    model metadata needed for scoring at inference time.
    """

    head_stats: dict[tuple[int, int], HeadFrequencyStats]  # (layer, head) -> stats
    head_dim: int
    rope_style: str
    num_layers: int
    num_kv_heads: int

    def state_dict(self) -> dict[str, Any]:
        """Serialize to state dict for checkpoint embedding."""
        stats_serialized = {}
        for (layer, head), hs in self.head_stats.items():
            key = f"layer{layer:02d}_head{head:02d}"
            stats_serialized[key] = {
                "q_mean_real": hs.q_mean_complex.real.cpu(),
                "q_mean_imag": hs.q_mean_complex.imag.cpu(),
                "q_abs_mean": hs.q_abs_mean.cpu(),
            }
        return {
            "metadata": {
                "head_dim": self.head_dim,
                "rope_style": self.rope_style,
                "num_layers": self.num_layers,
                "num_kv_heads": self.num_kv_heads,
                "sampled_heads": [[layer, head] for layer, head in self.head_stats],
            },
            "stats": stats_serialized,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> CalibrationData:
        """Deserialize from state dict."""
        metadata = state["metadata"]
        stats_raw = state["stats"]
        sampled_heads = [tuple(pair) for pair in metadata["sampled_heads"]]
        head_stats: dict[tuple[int, int], HeadFrequencyStats] = {}
        for layer, head in sampled_heads:
            key = f"layer{layer:02d}_head{head:02d}"
            entry = stats_raw[key]
            q_mean_complex = torch.complex(
                entry["q_mean_real"].to(torch.float32),
                entry["q_mean_imag"].to(torch.float32),
            )
            q_abs_mean = entry["q_abs_mean"].to(torch.float32)
            head_stats[(int(layer), int(head))] = HeadFrequencyStats(
                q_mean_complex=q_mean_complex,
                q_abs_mean=q_abs_mean,
            )
        return cls(
            head_stats=head_stats,
            head_dim=metadata["head_dim"],
            rope_style=metadata["rope_style"],
            num_layers=metadata["num_layers"],
            num_kv_heads=metadata["num_kv_heads"],
        )


def compute_head_stats_from_q(
    q_pre_rope: torch.Tensor,
    style: str = "half",
) -> HeadFrequencyStats:
    """Compute frequency statistics for a single head from pre-RoPE Q vectors.

    Args:
        q_pre_rope: Pre-RoPE query vectors for one head, shape (seq_len, head_dim).
        style: RoPE pairing style.

    Returns:
        HeadFrequencyStats with q_mean_complex and q_abs_mean.
    """
    q_complex = to_complex_pairs(q_pre_rope, style=style)  # (seq_len, freq_count)
    q_mean_complex = q_complex.mean(dim=0)  # (freq_count,)
    q_abs_mean = q_complex.abs().mean(dim=0)  # (freq_count,)
    return HeadFrequencyStats(
        q_mean_complex=q_mean_complex,
        q_abs_mean=q_abs_mean,
    )
