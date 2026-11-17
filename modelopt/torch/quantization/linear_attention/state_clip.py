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

"""Experimental per-head learned clipping with true NVFP4 state scales."""

import torch
from torch import nn

from .utils import forward_value

__all__ = ["NVFP4StateClip"]


def _round_e2m1(value):
    magnitude = value.abs()
    rounded = torch.where(magnitude <= 5, 4.0, 6.0)
    for threshold, level, inclusive in (
        (3.5, 3.0, False),
        (2.5, 2.0, True),
        (1.75, 1.5, False),
        (1.25, 1.0, True),
        (0.75, 0.5, False),
        (0.25, 0.0, True),
    ):
        below = magnitude <= threshold if inclusive else magnitude < threshold
        rounded = torch.where(below, level, rounded)
    return rounded.copysign(value)


def _round_e4m3(value):
    # Express the E4M3 grid explicitly so graph compilation also works on SM86.
    exponent = torch.frexp(value)[1].clamp(min=-5) - 4
    step = torch.ldexp(torch.ones_like(value), exponent)
    return (value / step).round() * step


def _capped_nvfp4(state, cap):
    """STE for E2M1 and E4M3 rounding; stop gradients through observed maxima/outer scale."""
    blocks = state.float().unflatten(-1, (-1, 16))
    amax = blocks.detach().abs().amax(-1, keepdim=True)
    cap = cap.reshape(*([1] * (blocks.ndim - 4)), -1, 1, 1, 1)
    # Smooth minimum (p=32) lets caps move before a hard clipping threshold is crossed.
    safe_amax = amax.clamp_min(torch.finfo(torch.float32).tiny)
    capped = torch.exp(-torch.logaddexp(-32 * safe_amax.log(), -32 * cap.log()) / 32)
    capped = torch.where(amax > 0, capped, 0.0)
    outer = (capped.detach().amax((-3, -2, -1), keepdim=True) / (6 * 448)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    scale_units = (capped / (6 * outer)).clamp(0, 448)
    scale_units = forward_value(scale_units, _round_e4m3(scale_units))
    scale = scale_units * outer
    safe_scale = torch.where(scale > 0, scale, 1.0)
    normalized = (blocks / safe_scale).clamp(-6, 6)
    rounded = forward_value(normalized, _round_e2m1(normalized))
    result = torch.where(scale > 0, rounded * scale, 0.0)
    return result.flatten(-2).to(state.dtype)


class NVFP4StateClip(nn.Module):
    """One calibrated, learned positive cap per local state head; blocks remain V16.

    ``calibrating=True`` observes unquantized states. Training uses the same forward
    as serving. Gradients use STE rounding and the saturation derivative, not the
    mathematical derivative of discrete quantization. Runtime storage remains QDQ.
    """

    format = "nvfp4_v16_e4m3_per_request_head"

    def __init__(self, initial_cap, *, calibrating=False, learning_rate_multiplier=1000.0):
        """Initialize FP32 per-head caps from a separate calibration pass."""
        super().__init__()
        initial_cap = torch.as_tensor(initial_cap, dtype=torch.float32)
        if (
            initial_cap.ndim != 1
            or not torch.isfinite(initial_cap).all()
            or (initial_cap <= 0).any()
        ):
            raise ValueError("State clipping requires finite positive caps, one per head")
        self.register_buffer("initial_cap", initial_cap.clone())
        self.register_buffer("observed_amax", torch.zeros_like(initial_cap))
        self.phi = nn.Parameter(torch.zeros_like(initial_cap), requires_grad=not calibrating)
        self.calibrating = calibrating
        self.learning_rate_multiplier = learning_rate_multiplier
        for tensor in (self.initial_cap, self.observed_amax, self.phi):
            tensor.tensor_model_parallel = True
            tensor.partition_dim = 0
            tensor.partition_stride = 1

    def _apply(self, fn, recurse=True):
        # Preserve FP32 values, not merely the dtype, through Megatron's BF16 wrapper.
        saved = {
            name: getattr(self, name).detach().clone()
            for name in ("phi", "initial_cap", "observed_amax")
        }
        super()._apply(fn, recurse)
        for name, value in saved.items():
            current = getattr(self, name)
            current.data = value.to(device=current.device, dtype=torch.float32)
            current.tensor_model_parallel = True
            current.partition_dim = 0
            current.partition_stride = 1
        return self

    def caps(self):
        """Return positive per-head thresholds in original state units."""
        return self.initial_cap * (self.learning_rate_multiplier * self.phi.float()).exp()

    def forward(self, state=None):
        """Quantize states, or return caps through module hooks for a compiled recurrence."""
        if state is None:
            if self.calibrating:
                raise ValueError("Calibration requires a state to observe")
            return self.caps()
        if state.ndim not in (3, 4) or state.shape[-3] != self.phi.numel() or state.shape[-1] % 16:
            raise ValueError("State clipping expects [H,K,V] or [B,H,K,V], with V divisible by16")
        if self.calibrating:
            current = state.detach().abs().flatten(-2).amax(-1)
            if state.ndim == 4:
                current = current.amax(0)
            self.observed_amax.copy_(torch.maximum(self.observed_amax, current))
            return state
        return _capped_nvfp4(state, self.caps())

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        """Shard head parameters and calibration buffers along the TP head axis."""
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        values = self.state_dict(keep_vars=True)
        return make_sharded_tensors_for_checkpoint(
            values, prefix, dict.fromkeys(values, 0), sharded_offsets
        )


class ReferenceK128StateClip(NVFP4StateClip):
    """Igor reference: K128 E2M1 with FP32 scales, identity state STE and LSQ cap gradient.

    This research profile is not NVIDIA NVFP4. Match the peer's BF16 parameter,
    bounded exponential parameterization and gradient normalization explicitly.
    """

    format = "e2m1_k128_fp32_reference"

    def __init__(self, initial_cap):
        super().__init__(initial_cap)
        self.phi.register_hook(lambda grad: grad / self.learning_rate_multiplier)

    def _apply(self, fn, recurse=True):
        target_dtype = fn(self.phi.detach()).dtype
        super()._apply(fn, recurse)
        self.phi.data = self.phi.data.to(target_dtype)
        return self

    def caps(self):
        psi = (self.learning_rate_multiplier * self.phi.float()).clamp(-3.0, 0.5)
        return self.initial_cap * psi.exp()

    def forward(self, state=None):
        if state is None:
            return self.caps()
        from ...kernels.quantization.linear_attention.reference_k128 import reference_k128_qdq

        if state.shape[-3] != self.phi.numel():
            raise ValueError("State and cap head counts differ")
        return reference_k128_qdq(state, self.caps())
