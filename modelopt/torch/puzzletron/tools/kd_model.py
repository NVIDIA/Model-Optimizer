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

"""
Knowledge distillation loss functions.

Provides normalized_mse_loss and cosine_embedding_loss_batched for comparing
model outputs. Used by validation.py.
"""
# mypy: ignore-errors

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = ["normalized_mse_loss", "cosine_embedding_loss_batched"]


def normalized_mse_loss(
    input: Tensor,
    target: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
    epsilon: float = 1e-6,
    *,
    chunk_numel: int = 8 * 1024 * 1024,
) -> Tensor:
    if reduction == "none" or input.shape != target.shape:
        return F.mse_loss(input, target, reduction=reduction) / F.mse_loss(
            target, torch.zeros_like(target) + epsilon, reduction=reduction
        )
    if chunk_numel <= 0:
        raise ValueError(f"chunk_numel must be positive, got {chunk_numel}")

    flat_input = input.reshape(-1)
    flat_target = target.reshape(-1)
    numerator = input.new_zeros((), dtype=torch.float32)
    denominator = input.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_input.numel(), chunk_numel):
        stop = start + chunk_numel
        input_chunk = flat_input[start:stop]
        target_chunk = flat_target[start:stop]
        numerator = numerator + F.mse_loss(input_chunk, target_chunk, reduction="sum").float()
        denominator = denominator + F.mse_loss(
            target_chunk,
            torch.full_like(target_chunk, epsilon),
            reduction="sum",
        ).float()
    return numerator / denominator


def cosine_embedding_loss_batched(input: Tensor, target: Tensor) -> Tensor:
    # inputs are of shape (B,T,H)
    batch_size = input.size(0)
    input = input.view(batch_size, -1)
    target = target.view(batch_size, -1)
    target_tensor = input.new(input.size(0)).fill_(1)
    loss = F.cosine_embedding_loss(
        input1=input, input2=target, target=target_tensor, reduction="none"
    )
    return loss
