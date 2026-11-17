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


def _ste(value, projected):
    return projected.detach() + (value - value.detach())


def _relative_scale(base, delta):
    # Let a zero scale recover while keeping the initial native value exactly zero.
    floor = torch.where(base == 0, torch.full_like(base, 2**-14), 0)
    return (base * delta.exp() + floor * delta.expm1()).clamp_min(0)


def _native_scale_kernels(tensor):
    if not tensor.is_cuda:
        return None
    try:
        from modelopt.torch.kernels.quantization import repair
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("triton"):
            raise
        return None
    return repair


def _packed_projection_qualified(inputs, name, shape):
    # GB300 forward/backward qualification: packed IQ is ~1.5x faster and
    # saves 31-36 MiB here. At 32K tokens, materialized GEMM is 15-80x faster.
    return (
        inputs.is_cuda
        and inputs.dtype == torch.bfloat16
        and name in ("iq1_m", "iq2_xxs")
        and tuple(shape) == (2048, 4096)
        and 0 < inputs.numel() // shape[-1] <= 8
        and torch.cuda.get_device_capability(inputs.device) == (10, 3)
    )


def repair_linear(quantizer, inputs, bias=None):
    """Try the prepared representation without bypassing quantizer post-transforms."""
    state = getattr(quantizer, "_local_repair_state", None)
    if not isinstance(state, IQScaleState) or getattr(quantizer, "rotate_back_is_enabled", False):
        return None
    return state.linear(inputs, bias)


def repair_grouped_linear(quantizers, inputs, offsets, bias=None, *, is_transposed=False):
    """Try compatible native expert banks; heterogeneous/frozen banks keep their forward."""
    from torch.distributed.tensor import DTensor

    states: list[IQScaleState] = []
    for quantizer in quantizers:
        state = getattr(quantizer, "_local_repair_state", None)
        if not isinstance(state, IQScaleState):
            return None
        states.append(state)
    if is_transposed or isinstance(inputs, DTensor) or len(states) != 4:
        return None
    first = states[0]
    if not _packed_projection_qualified(inputs, first.name, first.weight_shape) or any(
        state.name != first.name
        or state.weight_shape != first.weight_shape
        or getattr(q, "rotate_back_is_enabled", False)
        for state, q in zip(states, quantizers, strict=True)
    ):
        return None
    from modelopt.torch.kernels.quantization.iq_grouped_gemm import iq_grouped_linear

    packed = [state.packed for state in states]
    key = tuple((p.data_ptr(), p._version) for p in packed)
    cached = getattr(first, "_grouped_pointers", None)
    if cached is None or cached[0] != key:
        pointers = torch.tensor(
            [p.data_ptr() for p in packed], device=inputs.device, dtype=torch.int64
        )
        # One immutable pointer table per owning representation, never projected fields/weights.
        first._grouped_pointers = key, pointers
    else:
        pointers = cached[1]
    first.projection_counts["packed_grouped"] = first.projection_counts.get("packed_grouped", 0) + 1
    return iq_grouped_linear(
        inputs,
        packed,
        torch.stack([state.projected_steps() for state in states]),
        offsets,
        first.name,
        first.weight_shape,
        bias,
        packed_pointers=pointers,
    )


class IQScaleState(nn.Module):
    """FP32 log-scale masters projected onto legal native FP16/integer fields."""

    def __init__(self, quantizer, processed_weight, weight_shape, field_prefix):
        """Capture the PTQ payload in logical projection coordinates."""
        super().__init__()
        self.name = quantizer.num_bits
        self.format = IQ_FORMAT_REGISTRY[self.name]
        self.weight_shape = tuple(weight_shape)
        self.field_prefix = field_prefix
        self.projection_counts = {}
        packed = (
            self.format.pack(processed_weight, quantizer)
            .detach()
            .reshape(
                *self.weight_shape[:-1],
                self.weight_shape[-1] // self.format.block_size,
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
        self.register_buffer(
            "local_sign", None if basis is None else torch.where(basis < 0, -1.0, 1.0)
        )
        self.register_buffer("local_base", None if basis is None else basis.abs())
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
        basis = _relative_scale(self.local_base, self.local_delta) * self.local_sign
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
        """Decode only this replay's weight and release its expanded levels afterward."""
        self.projection_counts["materialized"] = self.projection_counts.get("materialized", 0) + 1
        kernels = _native_scale_kernels(inputs)
        if kernels is not None and self.name not in kernels.SUPPORTED_FORMATS:
            kernels = None
        with torch.no_grad():
            if kernels is not None:
                levels = kernels.decode_iq_levels(self.packed, self.name, self.weight_shape)
            else:
                levels, _ = self.format.decode_int8(self.packed, torch.tensor(self.weight_shape))
        steps = self.projected_steps()
        if kernels is not None:
            return kernels.apply_block_scales(levels, steps, inputs.dtype).reshape(inputs.shape)
        values = levels.reshape(
            -1, steps.shape[-1], self.format.int8_group_size
        ).float() * steps.unsqueeze(-1)
        return values.reshape(inputs.shape).to(inputs.dtype)

    def linear(self, inputs, bias=None):
        """Use packed GEMM only in a numerically and empirically qualified regime."""
        from torch.distributed.tensor import DTensor

        if isinstance(inputs, DTensor) or not _packed_projection_qualified(
            inputs, self.name, self.weight_shape
        ):
            return None
        from modelopt.torch.kernels.quantization.iq_gemm import iq_linear

        self.projection_counts["packed_dense"] = self.projection_counts.get("packed_dense", 0) + 1
        return iq_linear(
            inputs, self.packed, self.projected_steps(), self.name, self.weight_shape, bias
        )

    @torch.no_grad()
    def project_(self):
        """Bound log masters before native projection."""
        self.global_delta.clamp_(-20, 20)
        if self.local_delta is not None:
            self.local_delta.clamp_(-20, 20)

    @torch.no_grad()
    def native_payload(self):
        """Write only scale bits into the original native payload."""
        return replace_scale_fields(self.packed, self.format, *self.scales())

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Bake processed coordinates once and persist the exact native codes/scales."""
        native = self.native_payload()
        values = self.format.dequantize(native, torch.tensor(self.weight_shape), dtype=weight.dtype)
        quantizer._enable_pre_quant_scale = False
        quantizer.disable_rotate()
        weight.copy_(values.reshape(weight.shape))
        pin_packed_weight(
            quantizer,
            self.name,
            native,
            pinned_importance(quantizer, self.name),
            source=weight if self.format.payload_scales is not None else None,
        )
