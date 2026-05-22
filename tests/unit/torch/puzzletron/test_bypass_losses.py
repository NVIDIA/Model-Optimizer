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

"""Unit tests for normalized MSE loss functions in sewing_kit/utils.py."""

import pytest
import torch

from modelopt.torch.puzzletron.sewing_kit.utils import (
    batched_normalized_mse_loss,
    normalized_mse_loss,
    vectorwise_normalized_mse_loss,
)
from modelopt.torch.puzzletron.utils.parsing import format_stitched_losses

# ---------------------------------------------------------------------------
# normalized_mse_loss
# ---------------------------------------------------------------------------


def test_normalized_mse_loss_identical_tensors():
    """Identical input and target should produce a loss of approximately 0."""
    torch.manual_seed(42)
    x = torch.randn(4, 8)
    loss = normalized_mse_loss(x, x)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_normalized_mse_loss_basic():
    """Loss should be positive and finite for random, non-identical tensors."""
    torch.manual_seed(42)
    input_ = torch.randn(4, 8)
    target = torch.randn(4, 8)
    loss = normalized_mse_loss(input_, target)
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_normalized_mse_loss_reduction_none():
    """With reduction='none' the output shape should match the input shape."""
    torch.manual_seed(42)
    input_ = torch.randn(4, 8)
    target = torch.randn(4, 8)
    loss = normalized_mse_loss(input_, target, reduction="none")
    assert loss.shape == input_.shape


def test_normalized_mse_loss_reduction_sum():
    """With reduction='sum' the output should be a scalar tensor."""
    torch.manual_seed(42)
    input_ = torch.randn(4, 8)
    target = torch.randn(4, 8)
    loss = normalized_mse_loss(input_, target, reduction="sum")
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


# ---------------------------------------------------------------------------
# vectorwise_normalized_mse_loss
# ---------------------------------------------------------------------------


def test_vectorwise_normalized_mse_loss_shape():
    """vectorwise_normalized_mse_loss should return a scalar for any 2-D input."""
    torch.manual_seed(42)
    input_ = torch.randn(4, 16)
    target = torch.randn(4, 16)
    loss = vectorwise_normalized_mse_loss(input_, target)
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)


def test_vectorwise_normalized_mse_loss_identical():
    """Identical input and target should give a loss of approximately 0."""
    torch.manual_seed(42)
    x = torch.randn(4, 16)
    loss = vectorwise_normalized_mse_loss(x, x)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


# ---------------------------------------------------------------------------
# batched_normalized_mse_loss
# ---------------------------------------------------------------------------


def test_batched_normalized_mse_loss_basic():
    """Should return a scalar with a positive, finite value for random tensors."""
    torch.manual_seed(42)
    input_ = torch.randn(4, 8)
    target = torch.randn(4, 8)
    loss = batched_normalized_mse_loss(input_, target)
    assert loss.ndim == 0  # scalar
    assert loss.item() > 0.0
    assert torch.isfinite(loss)


def test_batched_normalized_mse_loss_custom_dims():
    """Custom batch_dims=(0, 1) on a 3-D tensor should still return a scalar."""
    torch.manual_seed(42)
    input_ = torch.randn(2, 3, 8)
    target = torch.randn(2, 3, 8)
    loss = batched_normalized_mse_loss(input_, target, batch_dims=(0, 1))
    assert loss.ndim == 0  # scalar
    assert torch.isfinite(loss)
    assert loss.item() > 0.0


def test_batched_normalized_mse_loss_negative_batch_dims_match_positive_dims():
    input_ = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    target = input_ + 1.0

    positive_dims = batched_normalized_mse_loss(input_, target, batch_dims=(0, 1))
    negative_dims = batched_normalized_mse_loss(input_, target, batch_dims=(0, -2))

    torch.testing.assert_close(negative_dims, positive_dims)


def test_batched_normalized_mse_loss_zero_target_is_finite():
    """All-zero target slice must not produce NaN/Inf.

    With the relative-L2 formula ``sum((x-t)^2) / (sum(t^2) + eps)``, an all-zero
    target reduces the denominator to exactly ``eps`` — finite, no division by
    zero — so the loss equals ``||input||^2 / eps``. The numeric value is large
    by construction (that's what zero-magnitude targets mean), but the test
    pins the property we actually care about: finiteness, not magnitude.
    """
    input_ = torch.full((1, 8), 1.0)
    target = torch.zeros(1, 8)
    loss = batched_normalized_mse_loss(input_, target)
    assert torch.isfinite(loss)
    assert not torch.isnan(loss)


def test_batched_normalized_mse_loss_zero_input_and_target():
    """Both zero should give exactly 0.0 — numerator is zero, denominator is eps."""
    input_ = torch.zeros(2, 4)
    target = torch.zeros(2, 4)
    loss = batched_normalized_mse_loss(input_, target)
    assert loss.item() == 0.0


def test_batched_normalized_mse_loss_scale_invariance():
    """Scaling both input and target by the same constant must leave the loss
    unchanged for non-tiny targets — the defining property of relative-L2."""
    torch.manual_seed(0)
    input_ = torch.randn(4, 8)
    target = torch.randn(4, 8)
    baseline = batched_normalized_mse_loss(input_, target)
    scaled = batched_normalized_mse_loss(10.0 * input_, 10.0 * target)
    assert torch.allclose(baseline, scaled, rtol=1e-4, atol=1e-6)


def test_batched_normalized_mse_loss_rejects_shape_mismatch():
    input_ = torch.randn(2, 3)
    target = torch.randn(2, 1)

    with pytest.raises(ValueError, match="input and target shapes must match exactly"):
        batched_normalized_mse_loss(input_, target)


def test_batched_normalized_mse_loss_rejects_invalid_batch_dim():
    input_ = torch.randn(2, 3)
    target = torch.randn(2, 3)

    with pytest.raises(ValueError, match="batch_dims contains invalid dimension"):
        batched_normalized_mse_loss(input_, target, batch_dims=(2,))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"epsilon": 0.0}, "epsilon must be strictly positive"),
        ({"batch_dims": (0, -2)}, "duplicate dimensions"),
        ({"batch_dims": ("0",)}, "iterable of integer dimensions"),
    ],
)
def test_batched_normalized_mse_loss_rejects_invalid_options(kwargs, message):
    input_ = torch.randn(2, 3)
    target = torch.randn(2, 3)

    with pytest.raises(ValueError, match=message):
        batched_normalized_mse_loss(input_, target, **kwargs)


def test_format_stitched_losses_keeps_trainable_nan_visible():
    out = format_stitched_losses(
        {"block_0": float("nan"), "block_1": 1.0},
        initial_values_dict={"block_0": 0.5, "block_1": 2.0},
        not_trainable_names={"block_2"},
        step_number=3,
    )

    assert "nan" in out
    assert "non-finite" in out
    assert "Skipped=1" in out
    assert "No trainable blocks found" not in out


def test_format_stitched_losses_empty_trainable_reports_skipped_blocks():
    out = format_stitched_losses({}, not_trainable_names={"block_0", "block_1"})

    assert out == "No trainable losses found; skipped 2 non-trainable blocks"


def test_format_stitched_losses_reports_delta_from_initial_and_filters_stale_history():
    out = format_stitched_losses(
        {"block_0": 1.0, "block_1": 3.0},
        best_steps_dict={"block_0": 5, "block_9": 99},
        best_values_dict={"block_0": 0.5, "block_9": 9.0},
        initial_values_dict={"block_0": 2.0, "block_1": 3.0, "block_9": 9.0},
        not_trainable_names={"block_2"},
        step_number=8,
    )

    assert "↓ -1.0e+00 (-50%)" in out
    assert "↔ 0.0e+00" in out
    assert "Step 5" in out
    assert "Step 99" not in out
    assert "Skipped=1" in out
    assert "Avg=2.00e+00" in out
