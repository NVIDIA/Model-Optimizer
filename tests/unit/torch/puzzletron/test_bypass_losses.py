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

import torch

from modelopt.torch.puzzletron.sewing_kit.utils import (
    batched_normalized_mse_loss,
    normalized_mse_loss,
    vectorwise_normalized_mse_loss,
)

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
