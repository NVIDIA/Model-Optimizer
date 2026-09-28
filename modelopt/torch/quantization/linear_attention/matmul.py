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

"""Operand QDQ and explicit arithmetic schedules for linear-attention matmuls."""

from typing import get_args

import torch
from torch import nn

from .config import LinearAttentionMatmulConfig, _PrefillSite

__all__ = ["LinearAttentionMatmulSites"]


class _MatmulSite(nn.Module):
    def __init__(self, *, legacy_w=False):
        # QuantizeConfig imports this package before TensorQuantizer is defined.
        from modelopt.torch.quantization.config import QuantizerAttributeConfig
        from modelopt.torch.quantization.nn import TensorQuantizer

        super().__init__()
        if not legacy_w:
            self.lhs_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))
        self.rhs_quantizer = TensorQuantizer(QuantizerAttributeConfig(enable=False))

    def forward(self, lhs, rhs, policy, *, lhs_quantizer=None):
        # The emulation working dtype must not inherit an outer training autocast.
        with torch.autocast(device_type=lhs.device.type, enabled=False):
            # Operands are [sequence-or-chunk, head, row, reduction] before transpose.
            quantizer = lhs_quantizer if lhs_quantizer is not None else self.lhs_quantizer
            lhs, rhs = quantizer(lhs), self.rhs_quantizer(rhs)
            if policy.accumulator_dtype is None:
                return lhs @ rhs.transpose(-1, -2)
            acc = lhs.new_zeros(*lhs.shape[:-1], rhs.shape[-2])
            for lo in range(0, lhs.shape[-1], policy.reduction_block):
                hi = lo + policy.reduction_block
                acc = acc + lhs[..., lo:hi] @ rhs[..., lo:hi].transpose(-1, -2)
                rounded = acc.to(getattr(torch, policy.accumulator_dtype)).to(lhs.dtype)
                acc = acc + (rounded - acc).detach()
            return acc


class LinearAttentionMatmulSites(nn.ModuleDict):
    """The eight prefill matmuls, with independent QDQ on their actual operands.

    Inputs to each handle are four-dimensional, with reduction on the last axis.
    The state-read LHS uses the parent's ``gdn_w_quantizer`` or ``kda_w_quantizer``;
    no second W quantizer is registered here. New handles start disabled.
    """

    def __init__(self):
        """Create disabled operand handles for the eight prefill sites."""
        super().__init__(
            {name: _MatmulSite(legacy_w=name == "state_read") for name in get_args(_PrefillSite)}
        )

    @property
    def is_enabled(self):
        """Whether any of the additional operand handles is enabled."""
        return any(q.is_enabled for site in self.values() for q in site.children())

    def matmul(self, name, lhs, rhs, policy, *, w_quantizer=None):
        """Multiply LHS by transposed RHS using the named numerical site."""
        return self[name](
            lhs,
            rhs,
            policy.matmul.get(name, LinearAttentionMatmulConfig()),
            lhs_quantizer=w_quantizer if name == "state_read" else None,
        )

    def validate(self):
        """Reject formats whose scale grouping has not been qualified at these sites."""
        for name, site in self.items():
            for handle, quantizer in site.named_children():
                _validate_operand_quantizer(quantizer, f"linear_attn_sites.{name}.{handle}")


def _validate_operand_quantizer(quantizer, name):
    # Deferred because QuantizeConfig imports this package before TensorQuantizer exists.
    from modelopt.torch.quantization.nn import TensorQuantizer

    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError(f"{name} requires a single TensorQuantizer")
    if not quantizer.is_enabled:
        return
    fp8 = (
        quantizer.num_bits == (4, 3)
        and quantizer.axis == (0, 1, 2)
        and quantizer.block_sizes is None
    )
    nvfp4 = (
        quantizer.num_bits == (2, 1)
        and quantizer.axis is None
        and quantizer.block_sizes == {-1: 16, "type": "dynamic", "scale_bits": (4, 3)}
    )
    if not (
        (fp8 or nvfp4)
        and quantizer._dynamic
        and quantizer.fake_quant
        and quantizer._pass_through_bwd
        and not quantizer.rotate_is_enabled
        and quantizer.pre_quant_scale is None
        and quantizer.backend is None
        and not quantizer._bias
        and not quantizer._use_constant_amax
    ):
        raise ValueError(
            f"{name} requires dynamic per-row E4M3 or block-16 NVFP4 "
            "with dynamic tensor amax, identity STE, and no extra transformations"
        )
