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

"""Eager attention wrapper for LTX-2 (ltx_core) skip-softmax sparse attention.

Patches ``Attention`` modules from ``ltx_core`` so that when the skip-softmax
thread-local flag is active, attention is computed eagerly with an explicit
``F.softmax`` call that the softmax-patching mechanism can intercept.

Used during **calibration only** — inference uses the Triton FA kernel via
the diffusers Triton backend.
"""

import math

import torch
import torch.nn.functional as F
from ltx_core.model.transformer.attention import Attention

from . import get_skip_softmax_context


def _ltx_eager_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Eager attention on LTX-2 layout ``[B, T, H*D]``.

    Mirrors the ``PytorchAttention`` class in ltx_core but uses an explicit
    ``F.softmax`` instead of ``scaled_dot_product_attention``.
    """
    b, _, dim_total = q.shape
    dim_head = dim_total // heads

    # Reshape to [B, T, H, D] then transpose to [B, H, T, D]
    q = q.view(b, -1, heads, dim_head).transpose(1, 2)
    k = k.view(b, -1, heads, dim_head).transpose(1, 2)
    v = v.view(b, -1, heads, dim_head).transpose(1, 2)

    scale = 1.0 / math.sqrt(dim_head)

    # Q @ K^T * scale
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Apply mask if provided
    if mask is not None:
        # Expand mask dimensions to match scores [B, H, Sq, Sk]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        scores = scores + mask

    # F.softmax — intercepted by skip-softmax patch
    scores = F.softmax(scores, dim=-1)

    # scores @ V
    out = torch.matmul(scores, v)

    # [B, H, T, D] → [B, T, H*D]
    out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
    return out


class _SkipSoftmaxLTXAttentionWrapper:
    """Wraps an ``attention_function`` callable from ltx_core.

    When the thread-local skip-softmax flag is active, routes to the eager
    attention path.  Otherwise calls the original function.
    """

    def __init__(self, original_fn):
        self._original_fn = original_fn

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_skip_softmax_context():
            return _ltx_eager_attention(q, k, v, heads, mask)
        return self._original_fn(q, k, v, heads, mask)


def register_ltx_eager_attention(model: torch.nn.Module) -> None:
    """Walk *model* and patch all ``ltx_core.model.transformer.attention.Attention`` modules.

    Patches modules so their ``attention_function`` is routed through the eager wrapper.
    Safe to call multiple times on the same model — already-wrapped modules are
    skipped.
    """
    for module in model.modules():
        if isinstance(module, Attention):
            fn = module.attention_function
            if not isinstance(fn, _SkipSoftmaxLTXAttentionWrapper):
                module.attention_function = _SkipSoftmaxLTXAttentionWrapper(fn)
