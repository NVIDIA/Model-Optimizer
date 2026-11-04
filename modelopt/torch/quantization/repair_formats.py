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

"""Differentiable native scale fields for local quantization repair."""

import copy
import math
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

from .ggml.common import pin_packed_weight
from .ggml.registry import GGML_FORMAT_REGISTRY
from .ggml.scale_fields import decode_scale_fields, replace_scale_fields
from .utils.native_weight import pin_native_weight

_FUSED_BLOCK_SCALES = ContextVar("modelopt_fused_repair_scales", default=False)


@contextmanager
def native_scale_execution():
    """Use native fused scale arithmetic in a validated runtime replay scope."""
    token = _FUSED_BLOCK_SCALES.set(True)
    try:
        yield
    finally:
        _FUSED_BLOCK_SCALES.reset(token)


def _native_scale_kernels(packed):
    if not packed.is_cuda or not _FUSED_BLOCK_SCALES.get():
        return None
    try:
        # Triton is optional; CPU and installations without it keep Torch codecs.
        from modelopt.torch.kernels.quantization import repair
    except ModuleNotFoundError as exc:
        if not (exc.name or "").startswith("triton"):
            raise
        return None
    return repair


def _decode_repair_levels(packed, fmt, shape, kernels):
    if kernels is not None and fmt.name in ("iq1_s", "iq1_m", "iq2_s", "iq2_xs"):
        return kernels.decode_iq_levels(packed, fmt.name, shape)
    return fmt.decode_int8(packed, torch.tensor(shape, dtype=torch.int64))[0]


def _ste(value, projected):
    return projected.detach() + (value - value.detach())


def _relative_scale(base, delta):
    # A zero native scale remains exactly zero at initialization but can recover.
    floor = torch.where(base == 0, torch.full_like(base, 6.103515625e-5), 0)
    return (base * delta.exp() + floor * delta.expm1()).clamp_min(0)


def _write_prepared_weight(quantizer, weight, values):
    # Prepared coordinates already include weight-side transforms. Bake once;
    # inverting and reapplying would round again and can change native codes.
    quantizer._enable_pre_quant_scale = False
    quantizer.disable_rotate()
    if quantizer.is_static_block_quant:
        values = quantizer._reset_to_original_shape(values)
    weight.copy_(values.reshape(weight.shape))


class NativeWeightShadow(nn.Module):
    """Optimize weights through the original format, retaining calibration policy."""

    def __init__(self, quantizer, processed_weight):
        """Keep an FP32 shadow and the original format conversion."""
        super().__init__()
        self.shadow = nn.Parameter(processed_weight.detach().float().clone())
        self.quantize = quantizer._fake_quantize

    def forward(self, inputs):
        """Apply the original quantizer with an identity weight STE."""
        value = self.shadow.to(inputs.dtype)
        # The native forward owns its clipping, scales and code selection. A
        # single identity STE also covers codecs with non-differentiable packing.
        with torch.no_grad():
            native = self.quantize(value)
        return _ste(value, native)

    @torch.no_grad()
    def project_(self):
        """Let the original native kernel project weights during forward."""

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Persist the shadow in the original quantizer's prepared coordinates."""
        _write_prepared_weight(quantizer, weight, self.shadow.to(weight.dtype))


class StaticActivationScale(nn.Module):
    """Learn a calibrated integer threshold while retaining data-dependent codes."""

    def __init__(self, quantizer):
        """Retain calibration and create a positive threshold master."""
        super().__init__()
        self.register_buffer("base_amax", quantizer._amax.detach().float().clone())
        self.delta = nn.Parameter(torch.zeros_like(self.base_amax))
        self.native_dtype = quantizer._amax.dtype

    def amax(self):
        """Return the native-dtype threshold with a rounding STE."""
        raw = _relative_scale(self.base_amax, self.delta)
        return _ste(
            raw, raw.clamp_max(torch.finfo(self.native_dtype).max).to(self.native_dtype).float()
        )

    def forward(self, inputs, quantize):
        """Use the calibrated native kernel at the projected threshold."""
        base = self.base_amax.clamp_min(2**-24)
        ratio = (self.amax() / base).clamp_min(2**-24)
        output = quantize((inputs.float() / ratio).to(inputs.dtype))
        surrogate = (output.float() * ratio).to(inputs.dtype)
        # Use the actual kernel at the projected threshold; the rescaling above
        # supplies scale gradients without introducing deployment rounding drift.
        quantizer = quantize.__self__
        original = quantizer._amax
        try:
            quantizer._amax = self.amax().detach().to(self.native_dtype)
            with torch.no_grad():
                native = quantize(inputs)
        finally:
            quantizer._amax = original
        return _ste(surrogate, native)

    @torch.no_grad()
    def project_(self):
        """Bound the log-threshold master without collecting calibration."""
        self.delta.clamp_(-20, 20)

    @torch.no_grad()
    def finalize_(self, quantizer):
        """Persist the projected activation threshold in its calibrated dtype."""
        quantizer._amax.copy_(self.amax().to(quantizer._amax.dtype))


class StaticIntegerScale(nn.Module):
    """Frozen scalar integer codes with the quantizer's original amax layout."""

    def __init__(self, quantizer, processed_weight):
        """Capture original integer codes and native calibrated thresholds."""
        super().__init__()
        amax = quantizer._get_amax(processed_weight).detach().float().clone()
        bound = float(2 ** (quantizer.num_bits - 1 + int(quantizer.unsigned)) - 1)
        max_bound = amax.new_tensor(bound)
        divisor = torch.where(amax > 2**-24, max_bound / amax, torch.ones_like(amax))
        minimum = 0 if quantizer.unsigned else -bound - int(not quantizer.narrow_range)
        bias = quantizer._get_bias(processed_weight)
        self.register_buffer("bias_value", bias.detach().clone() if bias is not None else None)
        centered = processed_weight if bias is None else processed_weight - bias
        codes = (centered.float() * divisor).round().clamp(minimum, bound)
        codes = torch.where(amax > 2**-24, codes, 0)
        code_dtype = torch.int8
        if quantizer.num_bits <= 8 and quantizer.unsigned:
            code_dtype = torch.uint8
        elif quantizer.num_bits > 8:
            code_dtype = (
                torch.int32 if quantizer.unsigned and quantizer.num_bits == 16 else torch.int16
            )
        self.register_buffer("codes", codes.to(code_dtype))
        self.register_buffer("base_amax", amax)
        self.register_buffer("max_bound", max_bound)
        self.delta = nn.Parameter(torch.zeros_like(amax))
        self.bound = bound
        native_amax = getattr(quantizer, "_amax", None)
        self.native_dtype = native_amax.dtype if native_amax is not None else processed_weight.dtype
        self.minimum = minimum
        self.shadow = None

    def configure(self, config, processed_weight):
        """Select frozen codes, code refresh, or a trainable weight shadow."""
        if config.mode in ("weight_only", "joint"):
            self.shadow = nn.Parameter(processed_weight.detach().float().clone())
        elif config.requantize:
            self.register_buffer("requantize_source", processed_weight.detach().float().clone())
        self.delta.requires_grad_(config.mode != "weight_only")

    def amax(self):
        """Return positive native-dtype thresholds with a rounding STE."""
        value = _relative_scale(self.base_amax, self.delta.reshape_as(self.base_amax))
        projected = (
            value.clamp_max(torch.finfo(self.native_dtype).max).to(self.native_dtype).float()
        )
        return _ste(value, projected)

    def forward(self, inputs):
        """Decode original codes or explicitly reproject the selected shadow."""
        amax = self.amax()
        divisor = torch.where(amax > 2**-24, self.max_bound / amax, torch.ones_like(amax))
        codes = self.codes.float()
        source = (
            self.shadow if self.shadow is not None else getattr(self, "requantize_source", None)
        )
        if source is not None:
            centered = source if self.bias_value is None else source - self.bias_value
            raw = centered * divisor
            codes = _ste(raw, raw.round().clamp(self.minimum, self.bound))
            codes = torch.where(amax > 2**-24, codes, 0)
        values = (codes / divisor).to(inputs.dtype)
        return values if self.bias_value is None else values + self.bias_value

    @torch.no_grad()
    def project_(self):
        """Bound log-scale masters before native threshold projection."""
        self.delta.clamp_(-20, 20)

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Store the native threshold and weights without refreshing frozen codes."""
        threshold = self.amax().to(self.native_dtype)
        if getattr(quantizer, "_amax", None) is None:
            # Persist learned thresholds instead of deriving them from rounded DQ.
            quantizer._set_buffer("_amax", threshold.detach().clone())
            quantizer._dynamic = False
        else:
            quantizer._amax.copy_(threshold)
        values = self(torch.empty_like(self.codes, dtype=weight.dtype))
        _write_prepared_weight(quantizer, weight, values)


class GGMLScale(nn.Module):
    """FP16 super-block and legal discrete local scales, with immutable grid codes."""

    def __init__(self, quantizer, processed_weight):
        """Capture packed PTQ codes and legal hierarchical IQ fields."""
        super().__init__()
        self.name = quantizer.num_bits
        self.format = GGML_FORMAT_REGISTRY[self.name]
        # Reuse the actual payload selected by GPTQ/other PTQ, including its pin.
        quantizer._fake_quantize(processed_weight)
        cache = quantizer._quantizer_cache
        if cache is None:
            raise ValueError("GGML repair requires a materialized native payload.")
        packed = cache.packed_weights.detach()
        self.weight_shape = tuple(cache.weight_shape.tolist())
        global_scale, local, self.local_max = decode_scale_fields(packed, self.format)
        self.register_buffer("packed", packed)
        self.register_buffer("global_base", global_scale.abs())
        self.register_buffer("global_sign", torch.where(global_scale < 0, -1.0, 1.0))
        self.global_delta = nn.Parameter(torch.zeros_like(global_scale))
        self.register_buffer("local_base", None if local is None else (2 * local + 1).float())
        self.local_delta = None if local is None else nn.Parameter(torch.zeros_like(local.float()))

    def scales(self):
        """Return projected FP16 block and legal local scale fields."""
        magnitude = _relative_scale(self.global_base, self.global_delta)
        magnitude = _ste(magnitude, magnitude.clamp_max(65504).half().float())
        global_scale = magnitude * self.global_sign
        if self.local_delta is None:
            return global_scale, None
        shadow = (self.local_base * self.local_delta.exp() - 1) / 2
        local = _ste(shadow, shadow.round().clamp(0, self.local_max))
        return global_scale, local

    def forward(self, inputs):
        """Decode immutable IQ levels with differentiable native fields."""
        # Only the current replay needs expanded levels. Keeping int8 levels
        # for every expert/layer would erase much of IQ's storage advantage.
        kernels = _native_scale_kernels(self.packed)
        with torch.no_grad():
            levels = _decode_repair_levels(self.packed, self.format, self.weight_shape, kernels)
        global_scale, local = self.scales()
        steps = global_scale.unsqueeze(-1)
        if local is not None:
            steps = steps * (2 * local + 1) / 8
        if kernels is not None:
            return kernels.apply_block_scales(levels, steps, inputs.dtype).reshape(inputs.shape)
        values = levels.reshape(-1, steps.shape[-1], self.format.int8_group_size).float()
        return (values * steps.unsqueeze(-1)).reshape(inputs.shape).to(inputs.dtype)

    @torch.no_grad()
    def project_(self):
        """Bound hierarchical scale masters before native projection."""
        self.global_delta.clamp_(-20, 20)
        if self.local_delta is not None:
            self.local_delta.clamp_(-20, 20)

    @torch.no_grad()
    def native_payload(self):
        """Replace only native scale fields in the original packed payload."""
        return replace_scale_fields(self.packed, self.format, *self.scales())

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Persist native scale bits and pin the original element codes."""
        native = self.native_payload()
        shape = torch.tensor(tuple(weight.shape), device=native.device)
        values = self.format.dequantize(native, shape, dtype=weight.dtype)
        _write_prepared_weight(quantizer, weight, values)
        pin_packed_weight(quantizer, quantizer.num_bits, native)


def grouped_native_repair(quantizers, dtype):
    """Batch homogeneous expert fields while preserving each quantizer's masters.

    Mixed or external representations retain their ordinary per-quantizer path.
    No expanded expert weights or integer levels persist across unit replays.
    """
    if not _FUSED_BLOCK_SCALES.get():
        return None
    states: list[GGMLScale] = []
    for quantizer in quantizers:
        state = getattr(quantizer, "_local_repair_state", None)
        if not isinstance(state, GGMLScale):
            return None
        states.append(state)
    if not states:
        return None
    first = states[0]
    if not first.packed.is_cuda or any(
        state.name != first.name or state.weight_shape != first.weight_shape for state in states
    ):
        return None
    kernels = _native_scale_kernels(first.packed)
    if kernels is None:
        return None
    packed = torch.cat([state.packed.reshape(-1, first.format.block_bytes) for state in states])
    rows, columns = first.weight_shape
    shape = (len(states) * rows, columns)
    with torch.no_grad():
        levels = _decode_repair_levels(packed, first.format, shape, kernels)
    base = torch.stack([state.global_base for state in states])
    delta = torch.stack([state.global_delta for state in states])
    sign = torch.stack([state.global_sign for state in states])
    magnitude = _relative_scale(base, delta)
    scales = _ste(magnitude, magnitude.clamp_max(65504).half().float()) * sign
    steps = scales.unsqueeze(-1)
    if first.local_delta is not None:
        base = torch.stack([state.local_base for state in states])
        delta = torch.stack([state.local_delta for state in states])
        shadow = (base * delta.exp() - 1) / 2
        local = _ste(shadow, shadow.round().clamp(0, first.local_max))
        steps = steps * (2 * local + 1) / 8
    return kernels.apply_block_scales(levels, steps, dtype).reshape(len(states), rows, columns)


class CompressedScale(nn.Module):
    """Native QTensor scale fields; the original packed weight is never rewritten."""

    def __init__(self, quantizer, weight):
        """Keep native payload ownership and create eligible scale masters."""
        super().__init__()
        native = weight.get_qtensor()
        self.kind = type(native).__name__
        self.block_sizes = quantizer.block_sizes
        self.native_dtypes = {}
        self.nvfp4_scale_layout = None
        self.deltas = nn.ParameterDict()
        for name in ("scale", "double_scale", "scale_zeros"):
            value = getattr(quantizer, "_" + name, None)
            if value is None:
                continue
            if self.kind == "NVFP4QTensor" and name == "scale" and value.dtype == torch.uint8:
                from .qtensor.nvfp4_tensor import cutlass_fp4_scale_to_modelopt_fp4_scale

                self.nvfp4_scale_layout = tuple(value.shape)
                value = cutlass_fp4_scale_to_modelopt_fp4_scale(
                    value, native.metadata["shape"][-2:]
                )
            self.native_dtypes[name] = value.dtype
            self.register_buffer("base_" + name, value.detach().float().clone())
            # NF4's signed double-quantized scale codes and offset stay fixed;
            # its floating double_scale is independently repairable.
            if name != "scale_zeros" and value.dtype != torch.int8:
                self.deltas[name] = nn.Parameter(torch.zeros_like(value, dtype=torch.float32))

    def values(self):
        """Project each scale field to its actual deployment encoding."""
        values = {}
        for name, dtype in self.native_dtypes.items():
            base = getattr(self, "base_" + name)
            if name not in self.deltas:
                values[name] = base
            elif dtype == torch.uint8 and self.kind.startswith("MX"):
                raw = base + self.deltas[name].reshape_as(base) / math.log(2)
                values[name] = _ste(raw, raw.round().clamp(0, 254))
            else:
                raw = _relative_scale(base, self.deltas[name].reshape_as(base))
                limit = torch.finfo(dtype).max
                projected = raw.clamp_max(limit)
                values[name] = _ste(raw, projected.to(dtype).float())
        return values

    def forward(self, inputs, dtype=None):
        """Decode existing native codes with differentiable scale fields."""
        native = inputs.get_qtensor() if hasattr(inputs, "get_qtensor") else inputs
        dtype = dtype or native.metadata["dtype"]
        fields = self.values()
        if self.kind == "MXQTensor":
            return native.dequantize(dtype=dtype, block_sizes=self.block_sizes, **fields)
        # Decode immutable levels through the existing codec, without retaining
        # a persistent floating shadow of the packed weight. These formats'
        # codebook values are exactly representable in BF16.
        unit = {name: torch.ones_like(value) for name, value in fields.items()}
        if self.kind.startswith("MX"):
            unit["scale"].fill_(127)
        if "scale_zeros" in unit:
            unit["scale_zeros"].zero_()
        with torch.no_grad():
            levels = native.dequantize(dtype=torch.bfloat16, block_sizes=self.block_sizes, **unit)
        scale = fields["scale"]
        if self.kind in ("FP8QTensor", "INT8QTensor"):
            if self.block_sizes:
                for axis, size in self.block_sizes.items():
                    scale = scale.repeat_interleave(size, dim=axis)
                scale = scale[tuple(slice(0, n) for n in levels.shape)]
            output = (
                levels.to(dtype) * scale.to(dtype) if self.kind == "FP8QTensor" else levels * scale
            )
        else:
            block_size = self.block_sizes[-1]
            if self.kind.startswith("MX"):
                factor = torch.exp2(scale - 127)
            elif self.kind == "NVFP4QTensor":
                factor = scale * fields["double_scale"]
            elif self.kind == "INT4QTensor":
                padded = nn.functional.pad(levels.flatten(), (0, -levels.numel() % block_size))
                output = padded.reshape(-1, block_size) / scale.reshape(-1, 1)
                return output.flatten()[: levels.numel()].reshape(levels.shape).to(dtype)
            elif self.kind == "NF4QTensor":
                factor = (scale / fields["double_scale"].unsqueeze(-1)).to(dtype)
                factor = factor + fields["scale_zeros"]
                from .extensions import get_cuda_ext

                if native._quantized_data.is_cuda and get_cuda_ext() is not None:
                    # The native CUDA kernel rounds both scale and output to BF16.
                    factor = _ste(factor, factor.to(torch.bfloat16).float())
            else:
                raise NotImplementedError(f"No differentiable native scale view for {self.kind}.")
            if self.kind == "MXFP8QTensor":
                # MXFP8 pads each logical row, unlike the flat INT4/NF4 codecs.
                padded = nn.functional.pad(levels, (0, -levels.shape[-1] % block_size))
                output = padded.reshape(*padded.shape[:-1], -1, block_size) * factor.unsqueeze(-1)
                output = output.reshape(padded.shape)[..., : levels.shape[-1]]
            else:
                padded = nn.functional.pad(levels.flatten(), (0, -levels.numel() % block_size))
                factor = factor.flatten()[: padded.numel() // block_size]
                output = padded.reshape(-1, block_size) * factor.reshape(-1, 1)
                if (
                    self.kind == "NF4QTensor"
                    and native._quantized_data.is_cuda
                    and get_cuda_ext() is not None
                ):
                    output = _ste(output, output.to(torch.bfloat16).float())
                output = output.flatten()[: levels.numel()].reshape(levels.shape)
        return output.to(dtype)

    @torch.no_grad()
    def project_(self):
        """Bound native scale masters without changing element codes."""
        for delta in self.deltas.values():
            delta.clamp_(-20, 20)

    @torch.no_grad()
    def finalize_quantizer(self, quantizer):
        """Store projected scale fields with their native dtype and layout."""
        for name, value in self.values().items():
            value = value.to(self.native_dtypes[name])
            if name == "scale" and self.nvfp4_scale_layout is not None:
                from .qtensor.nvfp4_tensor import modelopt_fp4_scale_to_cutlass_fp4_scale

                value = modelopt_fp4_scale_to_cutlass_fp4_scale(value).reshape(
                    self.nvfp4_scale_layout
                )
            previous = getattr(quantizer, "_" + name, None)
            if (
                previous is not None
                and previous.shape == value.shape
                and previous.dtype == value.dtype
            ):
                previous.copy_(value)
            else:
                quantizer._set_buffer("_" + name, value)

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Persist native fields while leaving the compressed weight untouched."""
        self.finalize_quantizer(quantizer)


class FakeNativeScale(CompressedScale):
    """Repair a fake-quantized weight using its existing format's native codec."""

    native_scale_buffers = False

    def __init__(self, quantizer, processed_weight, source_weight=None):
        """Encode or reuse the format-native PTQ payload exactly once."""
        from .qtensor import QTensorWrapper
        from .utils.native_weight import get_pinned_native_weight

        native_quantizer = copy.deepcopy(quantizer)
        pinned = get_pinned_native_weight(
            quantizer, processed_weight if source_weight is None else source_weight
        )
        # Static block preparation reshapes logical matrix axes into block axes.
        # Native codecs own their blocking and must receive the logical matrix,
        # including its native zero padding, rather than blocking it a second time.
        codec_weight = (
            processed_weight.reshape(quantizer._original_shape)
            if quantizer.is_static_block_quant
            else processed_weight
        )
        with torch.no_grad():
            if pinned is not None:
                native = pinned
            elif (
                quantizer.block_sizes
                and quantizer.block_sizes.get("type") == "dynamic"
                and not (
                    quantizer.is_mx_format
                    and quantizer.num_bits in ((2, 1), (4, 3))
                    and quantizer.block_sizes[-1] == 32
                )
                and not (
                    quantizer.num_bits == (2, 1)
                    and quantizer.block_sizes.get("scale_bits") == (4, 3)
                )
            ):
                from .qtensor import MXQTensor

                native, scales, global_scale = MXQTensor.quantize(
                    codec_weight,
                    quantizer.num_bits,
                    quantizer.block_sizes[-1],
                    quantizer.block_sizes["scale_bits"],
                    (
                        None
                        if quantizer.block_sizes["scale_bits"] == (8, 0)
                        else quantizer._get_amax(codec_weight)
                    ),
                )
                native_quantizer._set_buffer("_scale", scales)
                if global_scale is not None:
                    native_quantizer._set_buffer("_double_scale", global_scale)
            elif quantizer.num_bits == (2, 1) and quantizer.block_sizes.get("scale_bits") == (4, 3):
                from .qtensor import NVFP4QTensor

                scales, global_scale = NVFP4QTensor.get_weights_scaling_factor_from_quantizer(
                    quantizer, codec_weight
                )
                native, scales, global_scale = NVFP4QTensor.quantize(
                    codec_weight,
                    quantizer.block_sizes[-1],
                    weights_scaling_factor=scales,
                    weights_scaling_factor_2=global_scale,
                )
                native_quantizer._set_buffer("_scale", scales)
                native_quantizer._set_buffer("_double_scale", global_scale)
            else:
                native = native_quantizer._real_quantize(codec_weight)
        super().__init__(native_quantizer, QTensorWrapper(native))
        self.native_metadata = {
            key: value for key, value in native.metadata.items() if key != "qtensor_class"
        }
        self.native_class = type(native)
        self.register_buffer("packed_codes", native._quantized_data.detach().clone())

    def native_tensor(self):
        """Reconstruct the native code container with all codec metadata."""
        native = self.native_class(
            self.native_metadata["shape"], self.native_metadata["dtype"], self.packed_codes
        )
        native.metadata.update(self.native_metadata)
        return native

    def forward(self, inputs):
        """Apply native fields in the incoming compute dtype and shape."""
        return super().forward(self.native_tensor(), inputs.dtype).reshape(inputs.shape)

    @torch.no_grad()
    def finalize_(self, quantizer, weight):
        """Persist codec fields and pin the repaired fake-quantized native weight."""
        super().finalize_(quantizer, weight)
        native = self.native_tensor()
        values = native.dequantize(
            dtype=weight.dtype, block_sizes=quantizer.block_sizes, **self.values()
        )
        _write_prepared_weight(quantizer, weight, values)
        pin_native_weight(quantizer, native, weight)
