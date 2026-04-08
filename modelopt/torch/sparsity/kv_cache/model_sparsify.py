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

"""Entry points for KV cache sparsity: sparsify() and calibrate()."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from modelopt.torch.opt.conversion import apply_mode

from .config import TriAttentionConfig
from .mode import KVCacheSparsityRegistry

if TYPE_CHECKING:
    import torch.nn as nn

    from modelopt.torch.opt.searcher import ForwardLoop

__all__ = ["calibrate", "sparsify"]


def sparsify(
    model: nn.Module,
    config: dict[str, Any] | TriAttentionConfig,
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Apply KV cache sparsity optimization to a model.

    Registers the TriAttention mode on the model. Call ``calibrate()`` afterwards
    to compute frequency statistics from calibration data.

    Args:
        model: The model to optimize.
        config: TriAttentionConfig or dict with config values.
        forward_loop: Optional forward loop for integrated calibration.

    Returns:
        The model with TriAttention mode applied (in-place).
    """
    if isinstance(config, dict):
        config = TriAttentionConfig(**config)

    model = apply_mode(
        model,
        mode=[("triattention", config.model_dump())],
        registry=KVCacheSparsityRegistry,
    )

    if forward_loop is not None:
        model = calibrate(model, config, forward_loop=forward_loop)

    return model


def calibrate(
    model: nn.Module,
    config: dict[str, Any] | TriAttentionConfig | None = None,
    forward_loop: ForwardLoop | None = None,
) -> nn.Module:
    """Calibrate TriAttention frequency statistics.

    Runs a forward pass with hooks to capture pre-RoPE Q vectors, inverts RoPE,
    and computes per-head frequency centers. Results are stored in the model's
    modelopt_state metadata.

    Args:
        model: Model with TriAttention mode applied.
        config: Optional config override.
        forward_loop: Callable that runs forward passes on calibration data.

    Returns:
        The model with calibration data stored in metadata.
    """
    # Full GPU calibration with model forward passes will be implemented
    # when engine integration is designed. For now, calibrate is a no-op
    # that can be called safely after sparsify().
    return model
