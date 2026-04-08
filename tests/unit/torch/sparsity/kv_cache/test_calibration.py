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

"""Tests for TriAttention calibration."""

import torch

from modelopt.torch.sparsity.kv_cache.triattention.calibration import (
    CalibrationData,
    compute_head_stats_from_q,
)
from modelopt.torch.sparsity.kv_cache.triattention.scoring import HeadFrequencyStats


def test_compute_head_stats_shapes():
    """Head stats computed from Q tensor have correct shapes."""
    seq_len = 64
    head_dim = 16
    freq_count = head_dim // 2

    q_pre_rope = torch.randn(seq_len, head_dim)
    stats = compute_head_stats_from_q(q_pre_rope)

    assert stats.q_mean_complex.shape == (freq_count,)
    assert stats.q_mean_complex.dtype == torch.complex64
    assert stats.q_abs_mean.shape == (freq_count,)
    assert stats.q_abs_mean.dtype == torch.float32


def test_compute_head_stats_mean_abs_ge_abs_mean():
    """Mean of absolute values >= absolute value of mean (triangle inequality)."""
    q_pre_rope = torch.randn(128, 32)
    stats = compute_head_stats_from_q(q_pre_rope)

    abs_of_mean = torch.abs(stats.q_mean_complex)
    assert (stats.q_abs_mean >= abs_of_mean - 1e-6).all()


def test_compute_head_stats_single_token():
    """Single-token input: mean equals the single value."""
    head_dim = 8
    q_pre_rope = torch.randn(1, head_dim)
    stats = compute_head_stats_from_q(q_pre_rope)

    # For single token, mean_complex == the single complex value
    # and abs_mean == |single complex value|
    torch.testing.assert_close(stats.q_abs_mean, torch.abs(stats.q_mean_complex))


def test_calibration_data_state_dict_roundtrip():
    """CalibrationData can be serialized to and restored from state dict."""
    stats = {
        (0, 0): HeadFrequencyStats(
            q_mean_complex=torch.randn(8, dtype=torch.complex64),
            q_abs_mean=torch.rand(8),
        ),
        (0, 1): HeadFrequencyStats(
            q_mean_complex=torch.randn(8, dtype=torch.complex64),
            q_abs_mean=torch.rand(8),
        ),
        (1, 0): HeadFrequencyStats(
            q_mean_complex=torch.randn(8, dtype=torch.complex64),
            q_abs_mean=torch.rand(8),
        ),
    }
    calib = CalibrationData(
        head_stats=stats,
        head_dim=16,
        rope_style="half",
        num_layers=2,
        num_kv_heads=2,
    )

    state = calib.state_dict()
    restored = CalibrationData.from_state_dict(state)

    assert restored.head_dim == 16
    assert restored.rope_style == "half"
    assert restored.num_layers == 2
    assert restored.num_kv_heads == 2
    assert len(restored.head_stats) == 3

    for key in stats:
        torch.testing.assert_close(
            restored.head_stats[key].q_abs_mean,
            calib.head_stats[key].q_abs_mean,
        )
        torch.testing.assert_close(
            restored.head_stats[key].q_mean_complex,
            calib.head_stats[key].q_mean_complex,
        )


def test_calibration_data_state_dict_keys():
    """State dict has expected structure."""
    stats = {
        (2, 3): HeadFrequencyStats(
            q_mean_complex=torch.randn(4, dtype=torch.complex64),
            q_abs_mean=torch.rand(4),
        ),
    }
    calib = CalibrationData(
        head_stats=stats,
        head_dim=8,
        rope_style="half",
        num_layers=4,
        num_kv_heads=8,
    )

    state = calib.state_dict()
    assert "metadata" in state
    assert "stats" in state
    assert "layer02_head03" in state["stats"]
    assert state["metadata"]["head_dim"] == 8
    assert state["metadata"]["sampled_heads"] == [[2, 3]]
