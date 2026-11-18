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

"""Differentiable IQ native scale fields; all code and additive bits stay fixed."""

import torch
from torch import nn

from .ggml.common import pin_packed_weight, pinned_importance
from .ggml.registry import IQ_FORMAT_REGISTRY
from .ggml.scale_fields import decode_scale_fields, replace_scale_fields

__all__: list[str] = []


def _ste(value, projected):
    return projected.detach() + (value - value.detach())


def _relative_scale(base, delta):
    # Let a zero scale recover while keeping the initial native value exactly zero.
    delta = delta.clamp(-20, 20)
    floor = torch.where(base == 0, torch.full_like(base, 2**-14), 0)
    scale = base * delta.exp() + floor * delta.expm1()
    # Preserve the recovery derivative at zero across PyTorch clamp conventions.
    return torch.where(scale < 0, 0, scale)


class _ReplayScales(torch.autograd.Function):
    @staticmethod
    def forward(ctx, levels, steps, dtype):
        ctx.save_for_backward(levels)
        shape, width = levels.shape, levels.shape[-1]
        levels, steps = levels.reshape(-1, width), steps.reshape(-1, 1)
        result = torch.empty_like(levels, dtype=dtype)
        rows = max(1, 262144 // width)
        for start in range(0, len(levels), rows):
            part = slice(start, start + rows)
            torch.mul(levels[part], steps[part], out=result[part])
        return result.reshape(shape)

    @staticmethod
    def backward(ctx, gradient):
        (levels,) = ctx.saved_tensors
        shape, width = levels.shape, levels.shape[-1]
        levels, gradient = levels.reshape(-1, width), gradient.reshape(-1, width)
        result = torch.empty(levels.shape[0], device=levels.device, dtype=torch.float32)
        rows = max(1, 262144 // width)
        for start in range(0, len(levels), rows):
            part = slice(start, start + rows)
            result[part] = (gradient[part].float() * levels[part].float()).sum(-1)
        return None, result.reshape(shape[:-1]), None


class IQScaleState(nn.Module):
    """FP32 native scale masters at an existing valid IQ PTQ boundary."""

    def __init__(self, quantizer, processed_weight, weight_shape, field_prefix):
        """Capture the PTQ payload, including the quantizer's padded coordinates."""
        super().__init__()
        self.name = quantizer.num_bits
        self.format = IQ_FORMAT_REGISTRY[self.name]
        self.weight_shape = tuple(weight_shape)
        self.codec_shape = (
            tuple(quantizer._original_shape)
            if quantizer.is_static_block_quant
            else self.weight_shape
        )
        self.field_prefix = field_prefix
        packed = (
            self.format.pack(processed_weight, quantizer)
            .detach()
            .reshape(
                *self.codec_shape[:-1],
                self.codec_shape[-1] // self.format.block_size,
                self.format.block_bytes,
            )
        )
        global_scale, local, self.local_max = decode_scale_fields(packed, self.format)
        self.register_buffer("packed", packed)
        self.register_buffer("global_base", global_scale.abs())
        self.register_buffer("global_sign", torch.where(torch.signbit(global_scale), -1.0, 1.0))
        self.global_delta = nn.Parameter(torch.zeros_like(global_scale))
        spec = self.format.scale_spec
        if spec is None:
            raise ValueError(f"{self.name} has no native scale descriptor")
        self.local_min = spec.local_min
        self.local_multiplier = spec.local_multiplier
        self.local_bias = spec.local_bias
        self.local_divisor = spec.divisor
        basis = None if local is None else (self.local_multiplier * local + self.local_bias).float()
        self.register_buffer("local_base", basis)
        self.local_delta = None if local is None else nn.Parameter(torch.zeros_like(local.float()))

    def values(self):
        """Return canonical IDs and FP32 masters for ownership and checkpointing."""
        return {f"{self.field_prefix}.{name}": p for name, p in self.named_parameters()}

    def scales(self):
        """Project global and local fields with straight-through native rounding."""
        magnitude = _relative_scale(self.global_base, self.global_delta)
        global_scale = _ste(magnitude, magnitude.clamp_max(65504).half().float()) * self.global_sign
        if self.local_delta is None:
            return global_scale, None
        basis = self.local_base * self.local_delta.clamp(-20, 20).exp()
        shadow = (basis - self.local_bias) / self.local_multiplier
        return global_scale, _ste(shadow, shadow.round().clamp(self.local_min, self.local_max))

    def projected_steps(self):
        """Compose the codec's native hierarchical scale geometry."""
        global_scale, local = self.scales()
        steps = global_scale[:, None]
        return (
            steps
            if local is None
            else steps * (self.local_multiplier * local + self.local_bias) / self.local_divisor
        )

    def forward(self, inputs):
        """Replay native fields; inputs supplies only metadata such as shape and dtype."""
        with torch.no_grad():
            levels, _ = self.format.decode_int8(self.packed, torch.tensor(self.codec_shape))
        steps = self.projected_steps()
        levels = levels.reshape(-1, steps.shape[-1], self.format.int8_group_size)
        return _ReplayScales.apply(levels, steps, inputs.dtype).reshape(inputs.shape)

    @torch.no_grad()
    def project_(self):
        """Bound log masters before native projection."""
        self.global_delta.clamp_(-20, 20)
        self.global_delta.masked_fill_((self.global_base == 0) & (self.global_delta < 0), 0)
        if self.local_delta is not None:
            self.local_delta.clamp_(-20, 20)

    @torch.no_grad()
    def native_payload(self):
        """Write only scale bits into the original native payload."""
        return replace_scale_fields(self.packed, self.format, *self.scales())

    @torch.no_grad()
    def finalize_(self, quantizer, weight, *, native=None):
        """Bake processed coordinates once and persist the exact native codes/scales."""
        native = self.native_payload() if native is None else native
        values = self.format.dequantize(native, torch.tensor(self.codec_shape), dtype=weight.dtype)
        padded = self.codec_shape != self.weight_shape
        if padded:
            values = quantizer._reset_to_original_shape(values)
        if values.numel():
            minimum, maximum = (
                torch.aminmax(values) if values.is_contiguous() else (values.amin(), values.amax())
            )
            if not (torch.isfinite(minimum) & torch.isfinite(maximum)):
                raise ValueError(f"Decoded {self.name} weights must be finite in {weight.dtype}")
        quantizer._enable_pre_quant_scale = False
        quantizer.disable_rotate()
        weight.copy_(values.reshape(weight.shape))
        del values
        # Future forwards recreate zero padding, even when the payload's tail decodes nonzero.
        source = quantizer._process_for_blockquant(weight) if padded else weight
        pin_packed_weight(
            quantizer,
            self.name,
            native,
            pinned_importance(quantizer, self.name),
            source=source,
        )
