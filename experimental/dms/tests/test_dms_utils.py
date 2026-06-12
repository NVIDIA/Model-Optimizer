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

"""Tests for DMS utility functions."""

import torch

from experimental.dms.tests.utils import add_dms_to_path

try:
    from dms.core import get_gating_with_noise
except ImportError:
    add_dms_to_path()
    from dms.core import get_gating_with_noise


class TestGetGatingWithNoise:
    """Tests for the get_gating_with_noise function."""

    def test_output_shapes(self):
        """Gating outputs should match the shape of the input weights."""
        batch, heads, seq_len = 2, 4, 16
        gating_weights = torch.randn(batch, heads, seq_len)
        noise = torch.randn(batch, heads, seq_len)

        probs, decisions, logits = get_gating_with_noise(gating_weights, noise, tau=1.0)

        assert probs.shape == (batch, heads, seq_len)
        assert decisions.shape == (batch, heads, seq_len)
        assert logits.shape == (batch, heads, seq_len)

    def test_decisions_are_binary(self):
        """Discretized decisions should contain only 0s and 1s (in forward pass values)."""
        gating_weights = torch.randn(2, 4, 16)
        noise = torch.randn(2, 4, 16)

        _probs, decisions, _logits = get_gating_with_noise(gating_weights, noise, tau=1.0)

        # The straight-through estimator means forward values are binary
        unique_vals = set(decisions.detach().unique().tolist())
        assert unique_vals.issubset({0.0, 1.0})

    def test_probs_in_unit_interval(self):
        """Probabilities should be in the [0, 1] range (sigmoid output)."""
        gating_weights = torch.randn(2, 4, 16)
        noise = torch.randn(2, 4, 16)

        probs, _decisions, _logits = get_gating_with_noise(gating_weights, noise, tau=1.0)

        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_temperature_effect(self):
        """Higher temperature should push probabilities closer to 0.5."""
        gating_weights = torch.tensor([2.0, -2.0])
        noise = torch.zeros(2)

        probs_low_tau, _, _ = get_gating_with_noise(gating_weights, noise, tau=0.1)
        probs_high_tau, _, _ = get_gating_with_noise(gating_weights, noise, tau=10.0)

        # With high tau, probs should be closer to 0.5 than with low tau
        dist_low = (probs_low_tau - 0.5).abs()
        dist_high = (probs_high_tau - 0.5).abs()
        assert (dist_high < dist_low).all()

    def test_gradient_flows_through_decisions(self):
        """Straight-through estimator: gradients should flow through decisions."""
        gating_weights = torch.randn(2, 4, 16, requires_grad=True)
        noise = torch.randn(2, 4, 16)

        _probs, decisions, _logits = get_gating_with_noise(gating_weights, noise, tau=1.0)
        loss = decisions.sum()
        loss.backward()

        assert gating_weights.grad is not None
        assert gating_weights.grad.shape == gating_weights.shape
