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

"""Internal native IQ repair adapters for :func:`modelopt.torch.distill.bypass`."""

from dataclasses import dataclass
from itertools import islice
from types import MethodType

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from .ggml.registry import IQ_FORMAT_REGISTRY
from .nn import GroupedQuantizer, QuantModule, SequentialQuantizer, TensorQuantizer
from .repair_formats import IQScaleState
from .utils import quantizer_attr_names, weight_attr_names

__all__: list[str] = []


@dataclass(frozen=True)
class _WeightBinding:
    module: QuantModule
    index: int
    quantizer: TensorQuantizer
    name: str


@dataclass(frozen=True)
class IQRepairPlan:
    """Internal lazy bindings and caller flags; inspection does not slice FSDP parameters."""

    units: tuple[nn.Module, ...]
    bindings: tuple[tuple[_WeightBinding, ...], ...]
    parameter_flags: tuple[tuple[tuple[nn.Module, str, bool], ...], ...]


def _resolved_weights(bindings, *, materialize=True):
    """Yield selected weight views once per module, within the caller's onload scope."""
    modules = {}
    for binding in bindings:
        modules.setdefault(binding.module, {})[binding.index] = binding
    for module, selected in modules.items():
        owners = _write_owners(module, {id(b.quantizer) for b in selected.values()})
        for name in set(owners.values()):
            parameter = module._parameters[name]
            if (
                isinstance(parameter, DTensor)
                and parameter.ndim > 2
                and parameter.shape[0] == getattr(module, "num_experts", None)
                and parameter.to_local().shape[0] != module.num_experts
                and any(p.is_shard() and p.dim % parameter.ndim == 0 for p in parameter.placements)
            ):
                raise ValueError("Materialize globally indexed expert shards before IQ repair")
        remaining = set(selected)
        weights = (
            module.iter_weights_for_calibration()
            if materialize
            else ((None, q) for q in module.iter_weight_quantizers_for_calibration())
        )
        for index, (weight, container) in enumerate(islice(weights, max(selected) + 1)):
            if index in selected:
                if _iq_quantizer(container) is not selected[index].quantizer:
                    raise ValueError(
                        "The module's weight quantizer changed after IQ repair planning"
                    )
                remaining.remove(index)
                weight = weight.to_local() if isinstance(weight, DTensor) else weight
                if materialize and weight.numel():
                    parameter = module._parameters[owners[id(selected[index].quantizer)]]
                    parameter = (
                        parameter.to_local() if isinstance(parameter, DTensor) else parameter
                    )
                    span, backing = _storage_span(weight), _storage_span(parameter)
                    if (
                        not _is_dense(parameter)
                        or not _is_dense(weight)
                        or span is None
                        or backing is None
                        or span[0] != backing[0]
                        or not backing[1] <= span[1] <= span[2] <= backing[2]
                    ):
                        raise ValueError("IQ repair needs a dense write-through weight view")
                yield selected[index], weight
        if remaining:
            raise ValueError("The module's calibrated weights changed after IQ repair planning")


def _iq_quantizer(container):
    if isinstance(container, SequentialQuantizer):
        enabled = [q for q in container if q.is_enabled]
        iq = [q for q in enabled if q.backend == "ggml" and q.num_bits in IQ_FORMAT_REGISTRY]
        if iq and (
            len(enabled) != 1
            or any(
                q.pre_quant_scale is not None or q.rotate_is_enabled
                for q in container
                if q is not iq[0]
            )
        ):
            raise ValueError("IQ Bypass requires an unambiguous native weight quantizer")
        return iq[0] if iq else None
    return (
        container
        if isinstance(container, TensorQuantizer)
        and container.is_enabled
        and container.backend == "ggml"
        and container.num_bits in IQ_FORMAT_REGISTRY
        else None
    )


def _router_paths(unit):
    # A feed-forward gate projection belongs to an expert; routing gates belong
    # directly to an MoE owner. Treat only the latter as frozen routers.
    return {
        ".".join(filter(None, (path, name)))
        for path, module in unit.named_modules()
        if any(hasattr(module, attr) for attr in ("experts", "routed_experts"))
        for name in ("router", "gate", "shared_expert_gate", "shared_experts_gate")
        if name in module._modules
    }


def _validate_quantizer(quantizer):
    if (
        not quantizer.fake_quant
        or getattr(quantizer, "_lsq", False)
        or quantizer.rotate_back_is_enabled
    ):
        raise ValueError("Finalize LSQ/rotate-back or compressed IQ state before Bypass")


def _storage_span(tensor):
    """Local storage interval; metadata inspection never gathers a DTensor."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    if tensor.is_meta or not tensor.numel():
        return None
    start = tensor.storage_offset() * tensor.element_size()
    size = (
        1
        + sum(
            (n - 1) * abs(stride) for n, stride in zip(tensor.shape, tensor.stride(), strict=True)
        )
    ) * tensor.element_size()
    return (tensor.device, tensor.untyped_storage().data_ptr()), start, start + size


def _is_dense(tensor):
    expected = 1
    for stride, size in sorted(zip(tensor.stride(), tensor.shape, strict=True)):
        if size > 1:
            if stride != expected:
                return False
            expected *= size
    return True


def _write_owners(module, selected):
    owners = {}
    for name in weight_attr_names(module):
        qname = quantizer_attr_names(name).weight_quantizer
        container = getattr(module, qname, None)
        if container is None:
            container = getattr(module, qname + "s", None)
        if container is None and name == getattr(module, "_first_proj_attr", None):
            container = getattr(module, "gate_up_proj_weight_quantizers", None)
        pairs = (
            [(f"weight{i}", q) for i, q in enumerate(container)]
            if isinstance(container, GroupedQuantizer)
            else [(name, container)]
        )
        for parameter_name, quantizers in pairs:
            if quantizers is None:
                continue
            matches = selected.intersection(id(q) for q in quantizers.modules())
            if matches and module._parameters.get(parameter_name) is not None:
                owners.update(dict.fromkeys(matches, parameter_name))
    if owners.keys() != selected:
        raise ValueError("The module's IQ quantizers have unsupported physical weight ownership")
    return owners


def _validate_ownership(units, plans):
    selected = {}
    for bindings in plans:
        for binding in bindings:
            selected.setdefault(binding.module, set()).add(id(binding.quantizer))
    seen_parameters = {}
    storage_ranges = {}
    for unit in units:
        for _, module in unit.named_modules(remove_duplicate=False):
            owners = (
                set(_write_owners(module, selected[module]).values())
                if module in selected
                else set()
            )
            for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
                write = name in owners
                identity = id(parameter)
                if identity in seen_parameters and (write or seen_parameters[identity]):
                    raise ValueError("Selected Bypass boundaries share physical weight ownership")
                seen_parameters[identity] = write
                span = _storage_span(parameter)
                if span is not None:
                    key, start, stop = span
                    previous = storage_ranges.setdefault(key, [])
                    if any(
                        start < end and begin < stop and (write or previous_write)
                        for begin, end, previous_write in previous
                    ):
                        raise ValueError("Selected Bypass weights overlap physical storage")
                    previous.append((start, stop, write))


def plan_iq_repair(units):
    """Refuse ambiguous ownership before changing flags or materializing weight views."""
    units = tuple(units)
    plans, seen_quantizers = [], set()
    for unit_index, unit in enumerate(units):
        bindings = []
        routers = _router_paths(unit)
        for path, module in unit.named_modules(remove_duplicate=False):
            eligible = isinstance(module, QuantModule) and not any(
                path == router or path.startswith(router + ".") for router in routers
            )
            containers = module.iter_weight_quantizers_for_calibration() if eligible else ()
            for index, container in enumerate(containers):
                quantizer = _iq_quantizer(container)
                if quantizer is None:
                    continue
                if id(quantizer) in seen_quantizers:
                    raise ValueError("Selected Bypass boundaries share a weight quantizer")
                _validate_quantizer(quantizer)
                if hasattr(quantizer, "_local_repair_state"):
                    raise ValueError("An IQ repair operation is already active")
                seen_quantizers.add(id(quantizer))
                bindings.append(
                    _WeightBinding(module, index, quantizer, f"{unit_index}.{path}.{index}")
                )
        if not bindings:
            raise ValueError("A selected Bypass boundary has no enabled native IQ weights")
        plans.append(tuple(bindings))
    if not plans:
        raise ValueError("Bypass requires at least one eligible boundary")
    _validate_ownership(units, plans)
    flags = tuple(
        tuple(
            (module, name, parameter.requires_grad)
            for module in unit.modules()
            for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False)
        )
        for unit in units
    )
    return IQRepairPlan(units, tuple(plans), flags)


def _repair_fake_quantize(quantizer, inputs):
    return quantizer._local_repair_state(inputs)


class IQRepairPolicy:
    """Internal IQ adapter for one serial Bypass run over fixed logical model state.

    The runner owns the units until close; logical weights and quantizer settings
    must stay fixed. Native weight-access contexts may replace temporary views.
    Masters stay outside model checkpoints; Bypass checkpoints own training state.
    """

    def __init__(self):
        """Initialize an inactive, model-independent repair policy."""
        self.adapters = []
        self._flags = {}
        self._calibration_flags = {}
        self._gemm_flags = {}
        self._layouts = {}
        self._units = {}

    def prepare(self, unit):
        """Freeze ordinary parameters and return only native IQ scale masters."""
        return self.prepare_plan(plan_iq_repair([unit]), 0)

    def prepare_plan(self, plan, index):
        """Materialize one unit and return only its masters inside its owner's onload scope."""
        unit = plan.units[index]
        bindings = plan.bindings[index]
        if any(hasattr(binding.quantizer, "_local_repair_state") for binding in bindings):
            raise ValueError("An IQ repair operation is already active")
        list(_resolved_weights(bindings, materialize=False))
        for binding in bindings:
            _validate_quantizer(binding.quantizer)
        _validate_ownership(plan.units, plan.bindings)
        self._units.update(dict.fromkeys(plan.units))
        for module, name, flag in plan.parameter_flags[index]:
            self._flags.setdefault((module, name), flag)
        unit.requires_grad_(False)
        for module in unit.modules():
            if isinstance(module, TensorQuantizer):
                self._calibration_flags.setdefault(module, module._if_calib)
                self._layouts.setdefault(
                    module,
                    (
                        module._axis,
                        module._block_sizes,
                        getattr(module, "_consumed_block_sizes", None),
                        getattr(module._calibrator, "_axis", None),
                    ),
                )
                module._local_repair_properties = {
                    "_if_calib": self._calibration_flags[module],
                    "_axis": self._layouts[module][0],
                    "_block_sizes": self._layouts[module][1],
                }
                module._if_calib = False
            if hasattr(module, "allow_real_quant_gemm"):
                self._gemm_flags.setdefault(
                    module,
                    ("allow_real_quant_gemm" in module.__dict__, module.allow_real_quant_gemm),
                )
                # Some packed forwards have no matching quantized-weight input backward.
                # Native decode + differentiable linear retains the exact frozen operator.
                module.allow_real_quant_gemm = False
        fields = []
        try:
            for binding, weight in _resolved_weights(bindings):
                quantizer = binding.quantizer
                had_override = "_fake_quantize" in quantizer.__dict__
                previous = quantizer.__dict__.get("_fake_quantize")
                captured = []

                def capture_prepared(inputs):
                    captured.append(inputs.detach())
                    return inputs

                quantizer._fake_quantize = capture_prepared
                try:
                    with torch.no_grad():
                        quantizer(weight)
                finally:
                    if had_override:
                        quantizer._fake_quantize = previous
                    else:
                        delattr(quantizer, "_fake_quantize")
                if len(captured) != 1:
                    raise ValueError("Close quantization cache scopes before preparing IQ Bypass")
                state = IQScaleState(quantizer, captured[0], weight.shape)
                object.__setattr__(quantizer, "_local_repair_state", state)
                quantizer._local_repair_active = True
                quantizer._fake_quantize = MethodType(_repair_fake_quantize, quantizer)
                self.adapters.append((binding, state, had_override, previous))
                fields.extend(state.parameters())
        except BaseException:
            self.close()
            raise
        return fields

    @torch.no_grad()
    def project_(self):
        """Project prepared native fields after an optimizer update."""
        for _, state, _, _ in self.adapters:
            state.project_()

    @torch.no_grad()
    def finalize_(self):
        """Commit this policy's native scales, rolling back a failed publication for retry."""
        states = {binding: state for binding, state, _, _ in self.adapters}
        for binding, state in states.items():
            _validate_quantizer(binding.quantizer)
            if binding.quantizer.num_bits != state.name:
                raise ValueError("The native IQ format changed during repair")
        _validate_ownership(self._units, (states,))
        native = {binding: state.native_payload() for binding, state in states.items()}
        list(_resolved_weights(states, materialize=False))
        backups = {}
        try:
            for binding, weight in _resolved_weights(states):
                quantizer = binding.quantizer
                # Keep rollback weights on the host, never retain gathered/onloaded tensors.
                backups[binding] = (
                    weight.to(device="cpu", copy=True),
                    quantizer._buffers.copy(),
                    quantizer._non_persistent_buffers_set.copy(),
                    quantizer._quantizer_cache,
                    quantizer._enable_pre_quant_scale,
                    quantizer._rotate,
                )
                states[binding].finalize_(quantizer, weight, native=native[binding])
        except BaseException:
            for binding, weight in _resolved_weights(backups):
                original, buffers, nonpersistent, cache, pre_scale, rotate = backups[binding]
                weight.copy_(original)
                quantizer = binding.quantizer
                quantizer._buffers = buffers
                quantizer._non_persistent_buffers_set = nonpersistent
                quantizer._quantizer_cache = cache
                quantizer._enable_pre_quant_scale = pre_scale
                quantizer._rotate = rotate
            raise
        self.close()

    def close(self):
        """Idempotently restore quantizers and caller gradient/calibration/GEMM flags."""
        for binding, _, had_override, previous in self.adapters:
            quantizer = binding.quantizer
            if had_override:
                quantizer._fake_quantize = previous
            else:
                delattr(quantizer, "_fake_quantize")
            delattr(quantizer, "_local_repair_state")
            delattr(quantizer, "_local_repair_active")
        self.adapters.clear()
        for quantizer, flag in self._calibration_flags.items():
            quantizer._if_calib = flag
        self._calibration_flags.clear()
        for module, (had_override, value) in self._gemm_flags.items():
            if had_override:
                module.allow_real_quant_gemm = value
            else:
                delattr(module, "allow_real_quant_gemm")
        self._gemm_flags.clear()
        for quantizer, (axis, blocks, consumed, calibrator_axis) in self._layouts.items():
            quantizer._axis, quantizer._block_sizes = axis, blocks
            quantizer._consumed_block_sizes = consumed
            if quantizer._calibrator is not None:
                quantizer._calibrator._axis = calibrator_axis
            delattr(quantizer, "_local_repair_properties")
        self._layouts.clear()
        for (module, name), flag in self._flags.items():
            if (parameter := module._parameters.get(name)) is not None:
                parameter.requires_grad_(flag)
        self._flags.clear()
        self._units.clear()
