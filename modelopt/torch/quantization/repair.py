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

"""Native quantization parameter ownership for independent local distillation."""

from dataclasses import dataclass
from itertools import islice
from types import MethodType
from typing import Any

import torch
from torch import nn

from .nn import GroupedQuantizer, QuantModule, SequentialQuantizer, TensorQuantizer
from .qtensor import QTensorWrapper
from .repair_formats import (
    CompressedScale,
    FakeNativeScale,
    GGMLScale,
    NativeWeightShadow,
    StaticActivationScale,
    StaticIntegerScale,
)

__all__ = [
    "CapabilityReport",
    "QuantizationRepairPolicy",
    "RepairParameterGroup",
    "preflight_quantizers",
    "register_repair_backend",
    "repair_backend_supports_tp",
]


_BACKEND_REPAIR = {}


def register_repair_backend(backend, factory, *, supports_tensor_parallel=False):
    """Register a native repair state with forward/project_/finalize_ and TP capability."""
    if backend in (None, "ggml") or backend in _BACKEND_REPAIR:
        raise ValueError(f"Repair backend {backend!r} is already registered or built in.")
    _BACKEND_REPAIR[backend] = (factory, supports_tensor_parallel)


def repair_backend_supports_tp(backend):
    """Return whether an extension supplies native TP field ownership."""
    return backend in (None, "ggml") or _BACKEND_REPAIR.get(backend, (None, False))[1]


@dataclass(frozen=True)
class CapabilityReport:
    """Read-only capability findings, before installing repair parameters."""

    quantizers: int
    unsupported: tuple[str, ...]

    def require_supported(self):
        """Raise before mutation when any requested field is unsupported."""
        if self.unsupported:
            raise ValueError(
                "Unsupported local quantization repair: " + "; ".join(self.unsupported)
            )


def _scale_adapter(quantizer, mode="scale_only"):
    if not isinstance(quantizer, TensorQuantizer):
        raise ValueError("Composite quantizers require a composite repair adapter.")
    if getattr(quantizer, "_lsq", False):
        raise ValueError("Finalize active LSQ before local repair.")
    if quantizer.rotate_back_is_enabled:
        raise ValueError("Rotate-back weight repair requires a native inverse-output adapter.")
    if quantizer.backend in _BACKEND_REPAIR:
        return _BACKEND_REPAIR[quantizer.backend][0](quantizer, mode)
    if quantizer.backend not in (None, "ggml"):
        raise ValueError(f"No repair adapter for backend {quantizer.backend!r}.")
    if not quantizer.fake_quant:
        return CompressedScale
    if mode == "weight_only":
        return NativeWeightShadow
    if quantizer.bias is not None and not isinstance(quantizer.num_bits, int):
        raise ValueError("Affine floating weights require a native offset-field adapter.")
    if quantizer.backend == "ggml":
        return GGMLScale
    if quantizer.block_sizes and quantizer.block_sizes.get("type") == "dynamic":
        from .tensor_quant import mx_format_map

        if quantizer.num_bits in mx_format_map and quantizer.block_sizes.get("scale_bits") in (
            (8, 0),
            (4, 3),
        ):
            return FakeNativeScale
    if quantizer.is_mx_format:
        return FakeNativeScale
    if quantizer.num_bits in ((4, 3), (2, 1)) and quantizer._is_real_quantize_support():
        return FakeNativeScale
    if quantizer.bias_calibrator is not None and quantizer.bias_type != "static":
        raise ValueError("This scalar adapter requires calibrated static weight fields.")
    if (
        isinstance(quantizer.num_bits, int)
        and quantizer.num_bits <= 16
        and not quantizer.is_mx_format
        and (not quantizer.block_sizes or quantizer.block_sizes.get("type", "static") == "static")
    ):
        return StaticIntegerScale
    raise ValueError(f"No native scale adapter for {quantizer.extra_repr()}.")


def _stored_weight_quantizer(quantizer):
    # Grouped modules' calibration protocol normally yields one member per
    # physical weight. A GroupedQuantizer passed as a whole has a representative
    # forward over member zero, which is also the only member to repair here.
    if isinstance(quantizer, GroupedQuantizer):
        return _stored_weight_quantizer(quantizer[0]) if len(quantizer) else None
    if isinstance(quantizer, SequentialQuantizer):
        for member in quantizer:
            if member.is_enabled:
                return member
            if member.pre_quant_scale is not None or member.rotate_is_enabled:
                raise ValueError(
                    "Disabled leading weight transforms need a composed source adapter."
                )
        return None
    return quantizer if quantizer.is_enabled else None


def preflight_quantizers(units, config):
    """Inspect quantizers without reading sharded weight views or changing state."""
    problems, seen = [], set()
    count = 0
    weight_ids, weight_scale_ids = set(), set()
    for index, unit in enumerate(units):
        owned = 0
        for module in unit.modules():
            if not isinstance(module, QuantModule):
                continue
            validate = getattr(module, "validate_local_repair", None)
            if validate is not None:
                try:
                    validate(config)
                except ValueError as exc:
                    problems.append(f"Unit {index}: {exc}")
            for container in module.iter_weight_quantizers_for_calibration():
                weight_ids.update(
                    id(q) for q in container.modules() if isinstance(q, TensorQuantizer)
                )
                try:
                    quantizer = _stored_weight_quantizer(container)
                except ValueError as exc:
                    problems.append(f"Unit {index}: {exc}")
                    continue
                if quantizer is None:
                    continue
                owned += 1
                count += 1
                if id(quantizer) in seen:
                    problems.append(f"Unit {index} shares a weight quantizer with another owner.")
                seen.add(id(quantizer))
                weight_ids.add(id(quantizer))
                weight_scale_ids.update(
                    id(value)
                    for name in ("_amax", "_global_amax", "_scale", "_double_scale")
                    if (value := getattr(quantizer, name, None)) is not None
                )
                try:
                    factory: Any = _scale_adapter(quantizer, config.mode)
                    if (config.mode != "scale_only" or config.requantize) and factory not in (
                        StaticIntegerScale,
                        NativeWeightShadow,
                    ):
                        raise ValueError(
                            "Weight/joint/requantization requires a native code-update adapter."
                        )
                except ValueError as exc:
                    problems.append(f"Unit {index}: {exc}")
        if not owned:
            problems.append(f"Unit {index} has no enabled weight quantizer.")
    if config.tune_activations:
        for unit in units:
            for quantizer in unit.modules():
                if (
                    not isinstance(quantizer, TensorQuantizer)
                    or not quantizer.is_enabled
                    or id(quantizer) in weight_ids
                ):
                    continue
                try:
                    if id(getattr(quantizer, "_amax", None)) in weight_scale_ids:
                        raise ValueError(
                            "Shared weight/activation thresholds require a compatible joint field adapter."
                        )
                    if (
                        quantizer._dynamic
                        or quantizer.amax is None
                        or _scale_adapter(quantizer) is not StaticIntegerScale
                        or quantizer.bias_calibrator is not None
                    ):
                        raise ValueError(
                            "Activation tuning requires calibrated static integer thresholds."
                        )
                except (ValueError, AssertionError) as exc:
                    problems.append(str(exc))
    return CapabilityReport(count, tuple(problems))


@dataclass
class RepairParameterGroup:
    """Native field parameters, their learning rate and semantic ownership."""

    params: list[nn.Parameter]
    lr: float
    fields: tuple[str, ...]


@dataclass(frozen=True)
class _WeightBinding:
    module: QuantModule
    index: int

    def resolve(self):
        weight = next(islice(self.module.iter_weights_for_calibration(), self.index, None))[0]
        from torch.distributed.tensor import DTensor

        return weight.to_local() if isinstance(weight, DTensor) else weight


def _repair_fake_quantize(quantizer, inputs):
    return quantizer._local_repair_state(inputs)


def _repair_activation_quantize(quantizer, inputs):
    return quantizer._local_repair_state(inputs, quantizer._local_repair_quantize)


class QuantizationRepairPolicy:
    """Own trainable native fields and restore ordinary quantizers on finalization."""

    def __init__(self):
        """Create native-field ownership without attaching model state."""
        self.adapters = []
        self._flags = {}
        self._scale_parameters = {}
        self._activations = []
        self._calibration_flags = {}
        self._gemm_flags = {}

    def _tie_shared_scales(self, state, quantizer):
        bindings = []
        if isinstance(state, (StaticIntegerScale, StaticActivationScale)):
            if getattr(quantizer, "_amax", None) is not None:
                bindings.append((state, "delta", quantizer._amax))
        elif isinstance(state, CompressedScale):
            for name in state.deltas:
                source = getattr(quantizer, "_" + name, None)
                if isinstance(state, FakeNativeScale) and not state.native_scale_buffers:
                    source = None
                    if name == "double_scale":
                        source = getattr(quantizer, "_global_amax", None)
                        if source is None:
                            source = getattr(quantizer, "_amax", None)
                    elif name == "scale" and (
                        state.kind == "FP8QTensor" or getattr(quantizer, "is_nvfp4_static", False)
                    ):
                        source = getattr(quantizer, "_amax", None)
                if source is not None:
                    bindings.append((state.deltas, name, source))
        for owner, name, source in bindings:
            parameter = owner[name] if isinstance(owner, nn.ParameterDict) else getattr(owner, name)
            previous = self._scale_parameters.setdefault(id(source), parameter)
            if previous.numel() != parameter.numel():
                raise ValueError("Shared native scale fields have incompatible layouts.")
            if isinstance(owner, nn.ParameterDict):
                owner[name] = previous
            else:
                setattr(owner, name, previous)

    def prepare(self, unit, config):
        """Freeze a unit and return independent native-field optimizer groups."""
        preflight_quantizers([unit], config).require_supported()
        entries = []
        seen, weight_ids = set(), set()
        for module in unit.modules():
            if not isinstance(module, QuantModule):
                continue
            for index, (weight, container) in enumerate(module.iter_weights_for_calibration()):
                weight_ids.update(
                    id(q) for q in container.modules() if isinstance(q, TensorQuantizer)
                )
                quantizer = _stored_weight_quantizer(container)
                if quantizer is None:
                    continue
                if id(quantizer) in seen:
                    raise ValueError(
                        "A weight quantizer has multiple owners; group their native fields explicitly."
                    )
                seen.add(id(quantizer))
                if isinstance(weight, QTensorWrapper):
                    entries.append((_WeightBinding(module, index), quantizer, CompressedScale))
                    continue
                factory: Any = _scale_adapter(quantizer, config.mode)
                entries.append((_WeightBinding(module, index), quantizer, factory))
        if not entries:
            raise ValueError("The selected repair unit has no enabled weight quantizer.")
        self._flags.setdefault(
            unit, {name: parameter.requires_grad for name, parameter in unit.named_parameters()}
        )
        unit.requires_grad_(False)
        for quantizer in unit.modules():
            if isinstance(quantizer, TensorQuantizer):
                self._calibration_flags.setdefault(quantizer, quantizer._if_calib)
                quantizer._if_calib = False
        for module in unit.modules():
            if hasattr(module, "allow_real_quant_gemm"):
                self._gemm_flags[module] = (
                    "allow_real_quant_gemm" in module.__dict__,
                    module.allow_real_quant_gemm,
                )
                module.allow_real_quant_gemm = False
        groups = []
        parameter_ids = set()
        try:
            for binding, quantizer, factory in entries:
                weight = binding.resolve()
                method = "dequantize" if factory is CompressedScale else "_fake_quantize"
                had_override = method in quantizer.__dict__
                previous_override = quantizer.__dict__.get(method)
                if factory is CompressedScale:
                    state = factory(quantizer, weight)
                else:
                    captured = []

                    def capture_prepared(inputs):
                        captured.append(inputs.detach())
                        return inputs

                    quantizer._fake_quantize = capture_prepared
                    quantizer._local_repair_active = True
                    try:
                        with torch.no_grad():
                            quantizer(weight)
                    finally:
                        delattr(quantizer, "_local_repair_active")
                        if had_override:
                            quantizer._fake_quantize = previous_override
                        else:
                            delattr(quantizer, "_fake_quantize")
                    if not captured:
                        raise ValueError(
                            "Weight preparation was intercepted by a cache; close its cache scope before repair."
                        )
                    state = (
                        factory(quantizer, captured[0], source_weight=weight)
                        if factory is FakeNativeScale
                        else factory(quantizer, captured[0])
                    )
                    if isinstance(state, StaticIntegerScale):
                        state.configure(config, captured[0])
                self._tie_shared_scales(state, quantizer)
                quantizer.add_module("_local_repair_state", state)
                quantizer._local_repair_active = True
                setattr(quantizer, method, MethodType(_repair_fake_quantize, quantizer))
                self.adapters.append(
                    (binding, quantizer, state, method, had_override, previous_override)
                )
                for name, parameter in state.named_parameters():
                    if not parameter.requires_grad or id(parameter) in parameter_ids:
                        continue
                    parameter_ids.add(id(parameter))
                    is_weight = name == "shadow"
                    groups.append(
                        RepairParameterGroup(
                            [parameter],
                            config.weight_lr if is_weight else config.scale_lr,
                            ("weights" if is_weight else "weight_scales",),
                        )
                    )
            if config.tune_activations:
                for quantizer in unit.modules():
                    if (
                        not isinstance(quantizer, TensorQuantizer)
                        or not quantizer.is_enabled
                        or id(quantizer) in weight_ids
                    ):
                        continue
                    state = StaticActivationScale(quantizer)
                    self._tie_shared_scales(state, quantizer)
                    had_override = "_fake_quantize" in quantizer.__dict__
                    previous = quantizer.__dict__.get("_fake_quantize")
                    quantizer._local_repair_quantize = quantizer._fake_quantize
                    quantizer.add_module("_local_repair_state", state)
                    quantizer._fake_quantize = MethodType(_repair_activation_quantize, quantizer)
                    self._activations.append((quantizer, state, had_override, previous))
                    parameters = [
                        parameter
                        for parameter in state.parameters()
                        if id(parameter) not in parameter_ids
                    ]
                    parameter_ids.update(id(parameter) for parameter in parameters)
                    if parameters:
                        groups.append(
                            RepairParameterGroup(
                                parameters, config.activation_lr, ("activation_scales",)
                            )
                        )
        except BaseException:
            self.close()
            raise
        return groups

    @torch.no_grad()
    def project_(self):
        """Bound repair masters before the next native forward."""
        for _, _, state, _, _, _ in self.adapters:
            state.project_()
        for _, state, _, _ in self._activations:
            state.project_()

    @torch.no_grad()
    def finalize_(self):
        """Install legal native fields and remove the training parameterization."""
        for binding, quantizer, state, _, _, _ in self.adapters:
            state.finalize_(quantizer, binding.resolve())
        for quantizer, state, _, _ in self._activations:
            state.finalize_(quantizer)
        self.close()

    def close(self):
        """Remove temporary parameterization, also safe after a failed preparation."""
        for _, quantizer, _, method, had_override, previous_override in self.adapters:
            if had_override:
                setattr(quantizer, method, previous_override)
            else:
                delattr(quantizer, method)
            delattr(quantizer, "_local_repair_state")
            delattr(quantizer, "_local_repair_active")
        self.adapters.clear()
        for quantizer, _, had_override, previous in self._activations:
            if had_override:
                quantizer._fake_quantize = previous
            else:
                delattr(quantizer, "_fake_quantize")
            delattr(quantizer, "_local_repair_state")
            delattr(quantizer, "_local_repair_quantize")
        self._activations.clear()
        for quantizer, flag in self._calibration_flags.items():
            quantizer._if_calib = flag
        self._calibration_flags.clear()
        for module, (had_override, value) in self._gemm_flags.items():
            if had_override:
                module.allow_real_quant_gemm = value
            else:
                delattr(module, "allow_real_quant_gemm")
        self._gemm_flags.clear()
        for unit, flags in self._flags.items():
            parameters = dict(unit.named_parameters())
            for name, flag in flags.items():
                parameters[name].requires_grad_(flag)
        self._flags.clear()
        self._scale_parameters.clear()
