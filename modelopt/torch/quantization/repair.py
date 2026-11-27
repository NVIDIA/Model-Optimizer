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
from itertools import chain

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from .ggml.registry import IQ_FORMAT_REGISTRY
from .nn import QuantModule, SequentialQuantizer, TensorQuantizer
from .utils.core_utils import _fsdp2_persistent_parameters, _weight_quantizer_owners

__all__: list[str] = []


@dataclass(frozen=True)
class _WeightBinding:
    module: QuantModule
    quantizer: TensorQuantizer


@dataclass(frozen=True)
class IQRepairPlan:
    """Internal lazy bindings and caller flags; inspection does not slice FSDP parameters."""

    units: tuple[nn.Module, ...]
    bindings: tuple[tuple[_WeightBinding, ...], ...]
    parameter_flags: tuple[tuple[tuple[nn.Module, str, nn.Parameter, bool], ...], ...]
    calibration_flags: dict[TensorQuantizer, bool]
    gemm_flags: dict[nn.Module, tuple[bool, bool]]


def _iq_quantizer(container):
    if isinstance(container, SequentialQuantizer):
        enabled = [q for q in container if q.is_enabled and q._if_quant]
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
        and container._if_quant
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
    try:
        address = tensor.untyped_storage().data_ptr()
    except NotImplementedError as exc:
        raise ValueError("IQ repair cannot inspect physical storage for this layout") from exc
    offsets = [(n - 1) * stride for n, stride in zip(tensor.shape, tensor.stride(), strict=True)]
    start = tensor.storage_offset() + sum(min(0, offset) for offset in offsets)
    stop = tensor.storage_offset() + sum(max(0, offset) for offset in offsets) + 1
    return (
        tensor.device,
        address + start * tensor.element_size(),
        address + stop * tensor.element_size(),
    )


def _storage_spans(tensor):
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    if tensor.layout == torch.sparse_coo:
        components = (tensor._indices(), tensor._values())
    elif tensor.layout in (torch.sparse_csr, torch.sparse_bsr):
        components = (tensor.crow_indices(), tensor.col_indices(), tensor.values())
    elif tensor.layout in (torch.sparse_csc, torch.sparse_bsc):
        components = (tensor.ccol_indices(), tensor.row_indices(), tensor.values())
    else:
        components = (tensor,)
    for component in components:
        if (span := _storage_span(component)) is not None:
            yield span


def _validate_storage_ranges(storage_ranges):
    for ranges in storage_ranges.values():
        end = write_end = -1
        for start, stop, write in sorted(ranges):
            if start < (end if write else write_end):
                raise ValueError("Selected Bypass weights overlap physical storage")
            end = max(end, stop)
            if write:
                write_end = max(write_end, stop)


def _validate_ownership(units, plans):
    selected = {}
    for bindings in plans:
        for binding in bindings:
            selected.setdefault(binding.module, set()).add(id(binding.quantizer))
    seen_tensors, seen_slots = {}, {}
    quantizers = {identity for identities in selected.values() for identity in identities}
    seen_quantizers = set()
    storage_ranges = {}
    seen_modules = set()
    for unit in units:
        backings, slots = {}, {}
        hooks = [getattr(module, "_hf_hook", None) for module in unit.modules()]
        while hooks:
            hook = hooks.pop()
            hooks.extend(getattr(hook, "hooks", ()))
            if getattr(hook, "weights_map", None) is None or not (
                getattr(hook, "offload", False) or hasattr(hook, "original_devices")
            ):
                continue
            # Import the optional adapter only for an initialized offload backing owner.
            from .plugins.accelerate import _offloaded_tensor_backings

            backings, slots = _offloaded_tensor_backings(unit)
            break
        for _, module in unit.named_modules(remove_duplicate=False):
            seen_modules.add(module)
            if id(module) in quantizers:
                if id(module) in seen_quantizers:
                    raise ValueError("Selected Bypass quantizers are shared with another owner")
                seen_quantizers.add(id(module))
            owners = (
                set(_weight_quantizer_owners(module, selected[module]).values())
                if module in selected
                else set()
            )
            for name, tensor in chain(module._parameters.items(), module._buffers.items()):
                if tensor is None:
                    continue
                owner = (module, name)
                write = name in module._parameters and name in owners
                for slot in slots.get(owner, ()):
                    previous = seen_slots.get(slot)
                    if previous is not None and previous[0] != owner and (write or previous[1]):
                        raise ValueError("Selected Bypass owners share an offload publication slot")
                    seen_slots[slot] = (owner, write or (previous is not None and previous[1]))
                # A resident execution value and its retained backing are one logical owner.
                # Inspect both, merging their intervals before comparing with other owners.
                representations = {
                    id(value): value for value in (tensor, backings.get(owner)) if value is not None
                }
                local_ranges = {}
                for identity, value in representations.items():
                    if identity in seen_tensors and (write or seen_tensors[identity]):
                        raise ValueError(
                            "Selected Bypass boundaries share physical weight ownership"
                        )
                    seen_tensors[identity] = write
                    for key, start, stop in _storage_spans(value):
                        local_ranges.setdefault(key, []).append((start, stop))
                for key, ranges in local_ranges.items():
                    merged = []
                    for start, stop in sorted(ranges):
                        if merged and start <= merged[-1][1]:
                            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
                        else:
                            merged.append((start, stop))
                    storage_ranges.setdefault(key, []).extend(
                        (start, stop, write) for start, stop in merged
                    )
    if selected.keys() - seen_modules:
        raise ValueError("The ownership root does not contain all selected Bypass modules")
    _validate_storage_ranges(storage_ranges)


def plan_iq_repair(units, *, root_model=None):
    """Plan units against every owner in the fixed logical ``root_model``.

    Native access may rematerialize views; callers must not change logical ownership.
    """
    units = tuple(units)
    plans, seen_quantizers = [], set()
    for unit in units:
        bindings = []
        routers = _router_paths(unit)
        for path, module in unit.named_modules(remove_duplicate=False):
            eligible = isinstance(module, QuantModule) and not any(
                path == router or path.startswith(router + ".") for router in routers
            )
            containers = module.iter_weight_quantizers_for_calibration() if eligible else ()
            for container in containers:
                quantizer = _iq_quantizer(container)
                if quantizer is None:
                    continue
                if id(quantizer) in seen_quantizers:
                    raise ValueError("Selected Bypass boundaries share a weight quantizer")
                _validate_quantizer(quantizer)
                if hasattr(quantizer, "_local_repair_state"):
                    raise ValueError("An IQ repair operation is already active")
                seen_quantizers.add(id(quantizer))
                bindings.append(_WeightBinding(module, quantizer))
        if not bindings:
            raise ValueError("A selected Bypass boundary has no enabled native IQ weights")
        plans.append(tuple(bindings))
    if not plans:
        raise ValueError("Bypass requires at least one eligible boundary")
    _validate_ownership((root_model,) if root_model is not None else units, plans)
    parameter_owners = {
        key: parameter
        for root in ((root_model,) if root_model is not None else units)
        for key, parameter in _fsdp2_persistent_parameters(root).items()
    }
    flags = []
    for unit in units:
        unit_flags = []
        for module in unit.modules():
            for name, parameter in module.named_parameters(recurse=False, remove_duplicate=False):
                owner = parameter_owners.get((module, name), parameter)
                unit_flags.append((module, name, owner, owner.requires_grad))
        flags.append(tuple(unit_flags))
    # A frozen module may occur in several units with separate policies. Capture
    # its caller flags before any policy can replace them with temporary values.
    calibration_flags = {
        module: module._if_calib
        for unit in units
        for module in unit.modules()
        if isinstance(module, TensorQuantizer)
    }
    gemm_flags = {
        module: ("allow_real_quant_gemm" in module.__dict__, module.allow_real_quant_gemm)
        for unit in units
        for module in unit.modules()
        if hasattr(module, "allow_real_quant_gemm")
    }
    return IQRepairPlan(units, tuple(plans), tuple(flags), calibration_flags, gemm_flags)
