# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for bypass-distillation loss and loss-log formatting behavior."""

import pytest
import torch

from modelopt.torch.puzzletron.sewing_kit.utils import (
    batched_normalized_mse_loss,
    vectorwise_normalized_mse_loss,
)
from modelopt.torch.puzzletron.utils.parsing import format_stitched_losses


def test_vectorwise_normalized_mse_loss_matches_batched_last_dim():
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    target = input_ + 1.0

    vectorwise = vectorwise_normalized_mse_loss(input_, target)
    batched = batched_normalized_mse_loss(input_, target, batch_dims=(0, 1))

    torch.testing.assert_close(vectorwise, batched)


def test_batched_normalized_mse_loss_matches_manual_relative_l2():
    input_ = torch.tensor([[[1.0, 2.0], [3.0, 5.0]], [[2.0, 4.0], [6.0, 8.0]]])
    target = torch.tensor([[[1.0, 1.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]])

    loss = batched_normalized_mse_loss(input_, target, epsilon=1e-6, batch_dims=(0, 1))
    expected = (((input_ - target) ** 2).sum(dim=2) / ((target**2).sum(dim=2) + 1e-6)).mean()

    torch.testing.assert_close(loss, expected)


def test_batched_normalized_mse_loss_handles_zero_targets():
    """All-zero target slice must not produce NaN/Inf.

    With the relative-L2 formula ``sum((x-t)^2) / (sum(t^2) + eps)``, an all-zero
    target reduces the denominator to exactly ``eps`` — finite, no division by
    zero — so the loss equals ``||input||^2 / eps``. The numeric value is large
    by construction (that's what zero-magnitude targets mean), but the test
    pins the property we actually care about: finiteness, not magnitude.
    """
    loss = batched_normalized_mse_loss(torch.full((1, 8), 1.0), torch.zeros(1, 8))
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)

    zero_loss = batched_normalized_mse_loss(torch.zeros(2, 4), torch.zeros(2, 4))
    torch.testing.assert_close(zero_loss, torch.tensor(0.0))


def test_batched_normalized_mse_loss_rejects_invalid_inputs():
    input_ = torch.randn(2, 3)
    target = torch.randn(2, 3)

    for args, kwargs, match in [
        ((input_, torch.randn(2, 1)), {}, "input and target shapes must match exactly"),
        ((input_, target), {"batch_dims": (2,)}, "batch_dims contains invalid dimension"),
        ((input_, target), {"epsilon": 0.0}, "epsilon must be strictly positive"),
    ]:
        with pytest.raises(ValueError, match=match):
            batched_normalized_mse_loss(*args, **kwargs)


def test_format_stitched_losses_reports_expected_summary_states():
    non_finite_out = format_stitched_losses(
        {"block_0": float("nan"), "block_1": 1.0},
        initial_values_dict={"block_0": 0.5, "block_1": 2.0},
        not_trainable_names={"block_2"},
        step_number=3,
    )
    assert "nan" in non_finite_out
    assert "non-finite" in non_finite_out
    assert "Skipped=1" in non_finite_out
    assert "No trainable blocks found" not in non_finite_out

    empty_out = format_stitched_losses({}, not_trainable_names={"block_0", "block_1"})
    assert empty_out == "No trainable losses found; skipped 2 non-trainable blocks"

    delta_out = format_stitched_losses(
        {"block_0": 1.0, "block_1": 3.0},
        best_steps_dict={"block_0": 5, "block_9": 99},
        best_values_dict={"block_0": 0.5, "block_9": 9.0},
        initial_values_dict={"block_0": 2.0, "block_1": 3.0, "block_9": 9.0},
        not_trainable_names={"block_2"},
        step_number=8,
    )
    assert "↓ -1.0e+00 (-50%)" in delta_out
    assert "↔ 0.0e+00" in delta_out
    assert "Step 5" in delta_out
    assert "Step 99" not in delta_out
    assert "Skipped=1" in delta_out
    assert "Avg=2.00e+00" in delta_out
