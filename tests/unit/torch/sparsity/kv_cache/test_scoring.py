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

"""Tests for TriAttention trigonometric scoring."""

import torch

from modelopt.torch.sparsity.kv_cache.triattention.rope_utils import build_geometric_offsets
from modelopt.torch.sparsity.kv_cache.triattention.scoring import (
    HeadFrequencyStats,
    compute_frequency_statistics_from_means,
    score_keys_for_round,
)


def test_compute_frequency_statistics_shapes():
    """Frequency statistics have correct shapes."""
    freq_count = 8
    seq_len = 16
    head_dim = freq_count * 2

    q_mean_complex = torch.randn(freq_count, dtype=torch.complex64)
    q_abs_mean = torch.rand(freq_count)
    k_unrot = torch.randn(seq_len, head_dim)

    amp, phi, extra = compute_frequency_statistics_from_means(q_mean_complex, q_abs_mean, k_unrot)
    assert amp.shape == (seq_len, freq_count)
    assert phi.shape == (seq_len, freq_count)
    assert extra.shape == (seq_len, freq_count)


def test_compute_frequency_statistics_amplitude_positive():
    """Amplitude is non-negative (product of absolute values)."""
    freq_count = 4
    q_mean_complex = torch.randn(freq_count, dtype=torch.complex64)
    q_abs_mean = torch.rand(freq_count).abs() + 0.1
    k_unrot = torch.randn(8, freq_count * 2)

    amp, _, _ = compute_frequency_statistics_from_means(q_mean_complex, q_abs_mean, k_unrot)
    assert (amp >= 0).all()


def test_compute_frequency_statistics_disable_mlr():
    """With disable_mlr=True, extra uses q_abs_mean directly."""
    freq_count = 4
    q_mean_complex = torch.randn(freq_count, dtype=torch.complex64)
    q_abs_mean = torch.rand(freq_count) + 1.0
    k_unrot = torch.randn(8, freq_count * 2)

    _, _, extra_normal = compute_frequency_statistics_from_means(
        q_mean_complex, q_abs_mean, k_unrot, disable_mlr=False
    )
    _, _, extra_disabled = compute_frequency_statistics_from_means(
        q_mean_complex, q_abs_mean, k_unrot, disable_mlr=True
    )
    assert not torch.allclose(extra_normal, extra_disabled)


def test_score_keys_for_round_shape():
    """Score output matches number of keys."""
    num_keys = 32
    freq_count = 8
    key_indices = torch.arange(num_keys)
    amp = torch.rand(num_keys, freq_count)
    phi = torch.rand(num_keys, freq_count)
    omega = torch.rand(freq_count, dtype=torch.float64)
    extra = torch.rand(num_keys, freq_count)
    offsets = build_geometric_offsets(16, torch.device("cpu"))
    freq_scale_sq = torch.ones(freq_count)

    scores = score_keys_for_round(
        key_indices,
        round_start=64,
        amp=amp,
        phi=phi,
        omega=omega,
        extra=extra,
        offsets=offsets,
        aggregation="mean",
        freq_scale_sq=freq_scale_sq,
    )
    assert scores.shape == (num_keys,)


def test_score_keys_empty():
    """Empty key set returns empty scores."""
    scores = score_keys_for_round(
        key_indices=torch.tensor([], dtype=torch.long),
        round_start=100,
        amp=torch.empty(0, 4),
        phi=torch.empty(0, 4),
        omega=torch.rand(4, dtype=torch.float64),
        extra=torch.empty(0, 4),
        offsets=build_geometric_offsets(16, torch.device("cpu")),
        aggregation="mean",
        freq_scale_sq=torch.ones(4),
    )
    assert scores.numel() == 0


def test_score_aggregation_mean_vs_max():
    """Mean and max aggregation produce different results."""
    num_keys = 10
    freq_count = 4
    key_indices = torch.arange(num_keys)
    amp = torch.rand(num_keys, freq_count)
    phi = torch.rand(num_keys, freq_count)
    omega = torch.rand(freq_count, dtype=torch.float64)
    extra = torch.rand(num_keys, freq_count)
    offsets = build_geometric_offsets(16, torch.device("cpu"))
    freq_scale_sq = torch.ones(freq_count)

    scores_mean = score_keys_for_round(
        key_indices,
        50,
        amp,
        phi,
        omega,
        extra,
        offsets,
        "mean",
        freq_scale_sq,
    )
    scores_max = score_keys_for_round(
        key_indices,
        50,
        amp,
        phi,
        omega,
        extra,
        offsets,
        "max",
        freq_scale_sq,
    )
    assert not torch.allclose(scores_mean, scores_max)


def test_score_keys_disable_trig():
    """With disable_trig=True, scores are position-independent (additive only)."""
    freq_count = 4
    # Two keys at very different positions
    key_indices = torch.tensor([0, 99])
    # Large amplitude so trig term dominates when enabled
    amp = torch.ones(2, freq_count) * 10.0
    phi = torch.zeros(2, freq_count)
    omega = torch.tensor([0.1, 0.5, 1.0, 2.0], dtype=torch.float64)
    extra = torch.ones(2, freq_count)
    offsets = build_geometric_offsets(16, torch.device("cpu"))
    freq_scale_sq = torch.ones(freq_count)

    scores_no_trig = score_keys_for_round(
        key_indices,
        100,
        amp,
        phi,
        omega,
        extra,
        offsets,
        "mean",
        freq_scale_sq,
        disable_trig=True,
    )
    # Without trig, both keys get the same score (additive term is position-independent)
    torch.testing.assert_close(scores_no_trig[0], scores_no_trig[1])


def test_head_frequency_stats_dataclass():
    """HeadFrequencyStats holds correct fields."""
    stats = HeadFrequencyStats(
        q_mean_complex=torch.randn(8, dtype=torch.complex64),
        q_abs_mean=torch.rand(8),
    )
    assert stats.q_mean_complex.shape == (8,)
    assert stats.q_abs_mean.shape == (8,)
