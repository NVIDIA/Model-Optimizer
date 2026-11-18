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

"""Actual caller mesh ownership and native quantizer field placement."""

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from modelopt.torch.quantization.nn import SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.repair_partition import (
    activation_partition,
    native_tp_weight_view,
    native_weight_partition,
    quantized_weight_bindings,
    validate_tp_output_quantizer,
)


def _size(mesh):
    return 1 if mesh is None else mesh.size()


def _submesh(mesh, axis):
    try:
        return None if mesh is None else mesh[axis]
    except (KeyError, RuntimeError, ValueError):
        return None


def torch_mesh_context(mesh):
    """Describe an actual dense Torch mesh without importing Transformers."""
    axes = mesh.mesh_dim_names
    if not axes or not set(axes).issubset(
        {"dp", "fsdp", "dp_replicate", "dp_shard", "tp", "pp", "cp", "ep"}
    ):
        raise ValueError("Bypass needs explicit supported Torch mesh axis names")
    if _size(_submesh(mesh, "ep")) > 1:
        raise ValueError("Use TransformersDeviceMesh for overlapping expert parallelism")

    def combine(names, alias):
        names = [name for name in names if name in axes]
        if not names:
            return None
        return mesh[names[0]] if len(names) == 1 else mesh[tuple(names)]._flatten(alias)

    data = ("dp", "fsdp", "dp_replicate", "dp_shard")
    context = BypassMeshContext(
        device_mesh=mesh,
        data_mesh=combine(data, "modelopt_bypass_data"),
        data_context_mesh=combine((*data, "cp"), "modelopt_bypass_data_context"),
        tensor_mesh=_submesh(mesh, "tp"),
        pipeline_mesh=_submesh(mesh, "pp"),
        context_mesh=_submesh(mesh, "cp"),
        fsdp_mesh=_submesh(mesh, "fsdp"),
    )
    context.expert_replica_mesh = context.data_context_mesh
    return context


@dataclass
class BypassMeshContext:
    """Real data lanes, disjoint weight owners and replica groups, with no launcher sizes."""

    device_mesh: DeviceMesh
    moe_mesh: DeviceMesh | None = None
    process_group: Any = None
    data_mesh: DeviceMesh | None = None
    data_context_mesh: DeviceMesh | None = None
    tensor_mesh: DeviceMesh | None = None
    expert_mesh: DeviceMesh | None = None
    expert_replica_mesh: DeviceMesh | None = None
    pipeline_mesh: DeviceMesh | None = None
    context_mesh: DeviceMesh | None = None
    fsdp_mesh: DeviceMesh | None = None
    replica_mesh: DeviceMesh | None = None
    mesh_manager: Any = None
    tp_quantizer_layouts: dict = field(default_factory=dict)
    parallel_metadata: dict = field(default_factory=dict)

    @property
    def dp_size(self):
        """Number of unique sample lanes, independent of overlapping expert ownership."""
        return _size(self.data_mesh)


def _model_group(context):
    if context.process_group is not None:
        return context.process_group
    mesh = context.device_mesh
    if mesh.size() == dist.get_world_size():
        return None
    return mesh._flatten("modelopt_bypass_model").get_group()


def _all_rank_values(context, value):
    if (
        not dist.is_initialized()
        or context is None
        or context.device_mesh is None
        or context.device_mesh.size() == 1
    ):
        return [value]
    group = _model_group(context)
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def apply_tp_quantizer_fields(quantizer, fields):
    """Apply a validated field plan, including pinned native payload metadata."""
    for name, value in fields.items():
        if isinstance(value, torch.Tensor):
            quantizer._set_buffer(name, value)
        else:
            setattr(quantizer, name, value)


def _tp_quantizer_layouts(model):
    """Plan each physical weight's native fields from its actual Torch placement."""
    plans, aliases = [], {}

    def plan(quantizer, view, source):
        fields, layouts, reset = native_weight_partition(quantizer, view, source)
        for name, value in fields.items():
            original = getattr(quantizer, name, None)
            if not isinstance(original, torch.Tensor):
                continue
            signature = (
                view
                if name.startswith("_native_weight_") or value.numel() != original.numel()
                else None,
                tuple(value.shape),
            )
            previous = aliases.get(id(original))
            if previous is not None:
                if previous[0] != signature:
                    raise ValueError("Shared native fields have incompatible TP layouts.")
                fields[name] = previous[1]
            else:
                aliases[id(original)] = (signature, value)
        plans.append((quantizer, fields, layouts, reset))

    for module in model.modules():
        for weight_name, quantizers_name, input_name in quantized_weight_bindings(module):
            weight = module._parameters.get(weight_name)
            container = getattr(module, quantizers_name, None)
            if not isinstance(weight, DTensor) or container is None:
                continue
            axes = weight.device_mesh.mesh_dim_names
            if not axes or "tp" not in axes:
                continue
            placement = weight.placements[axes.index("tp")]
            view = native_tp_weight_view(
                weight_name, weight.shape, placement, weight.device_mesh["tp"]
            )
            if view is None:
                continue
            fused = weight.ndim == 3 and weight_name != "weight"
            validate_tp_output_quantizer(getattr(module, "output_quantizer", None), view)
            local = None
            members = (
                list(container)
                if isinstance(container, (SequentialQuantizer, torch.nn.ModuleList))
                else [container]
            )
            for index, quantizer in enumerate(members):
                if not isinstance(quantizer, TensorQuantizer):
                    raise ValueError(
                        "Composite TP weights require an explicit physical-view adapter."
                    )
                if not quantizer.is_enabled:
                    continue
                if quantizer.backend not in (None, "ggml"):
                    raise ValueError(
                        f"Backend {quantizer.backend!r} requires a native TP partitioner."
                    )
                source = None
                if getattr(quantizer, "_native_weight_meta", None) is not None:
                    if local is None:
                        local = weight.to_local()
                    source = local[index] if fused else local
                plan(quantizer, view, source)
            input_quantizer = getattr(module, input_name, None)
            if isinstance(input_quantizer, TensorQuantizer):
                fields = activation_partition(input_quantizer, view)
                if fields:
                    plans.append(
                        (input_quantizer, fields, {}, bool(input_quantizer.is_static_block_quant))
                    )
    return plans


def prepare_quantizer_partitions(model, context):
    """Validate every field plan collectively before applying any native field changes."""
    plans, error = [], None
    try:
        plans = _tp_quantizer_layouts(model)
    except (ValueError, RuntimeError, TypeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = sorted({value for value in _all_rank_values(context, error) if value})
    if errors:
        raise ValueError("; ".join(errors))
    layouts = {}
    for quantizer, fields, ownership, reset in plans:
        apply_tp_quantizer_fields(quantizer, fields)
        if reset:
            for name in ("_block_reshape_size", "_original_shape", "_padding", "_slices"):
                if hasattr(quantizer, name):
                    delattr(quantizer, name)
        quantizer._quantizer_cache = None
        layouts[id(quantizer)] = ownership
    context.tp_quantizer_layouts[id(model)] = layouts
    return layouts
