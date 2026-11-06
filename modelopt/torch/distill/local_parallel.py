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

"""Torch mesh collectives and quantizer partitions shared by repair runtimes."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard
from torch.distributed.tensor.placement_types import _StridedShard

from modelopt.torch.quantization.nn import QuantModule, SequentialQuantizer, TensorQuantizer
from modelopt.torch.quantization.repair import repair_backend_supports_tp
from modelopt.torch.quantization.repair_partition import (
    NativeWeightPartition,
    activation_partition,
    native_weight_partition,
    validate_tp_output_quantizer,
)

__all__ = []


def _model_group(context):
    if context.process_group is not None:
        return context.process_group
    mesh = context.device_mesh
    if mesh.size() == dist.get_world_size():
        return None
    return mesh._flatten("local_repair_checkpoint").get_group()


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
        if not isinstance(module, QuantModule):
            continue
        if hasattr(module, "_first_proj_weight_quantizers_attr"):
            bindings = [
                (
                    module._first_proj_attr,
                    module._first_proj_weight_quantizers_attr,
                    module._first_proj_input_quantizer_attr,
                ),
                ("down_proj", "down_proj_weight_quantizers", "down_proj_input_quantizer"),
            ]
        else:
            bindings = [("weight", "weight_quantizer", "input_quantizer")]
        for weight_name, quantizers_name, input_name in bindings:
            weight = module._parameters.get(weight_name)
            container = getattr(module, quantizers_name, None)
            if not isinstance(weight, DTensor) or container is None:
                continue
            axes = weight.device_mesh.mesh_dim_names
            if not axes or "tp" not in axes:
                continue
            placement = weight.placements[axes.index("tp")]
            if not isinstance(placement, (Shard, _StridedShard)):
                continue
            fused = weight.ndim == 3 and weight_name != "weight"
            dim = placement.dim % weight.ndim - int(fused)
            if dim not in (0, 1) or weight.ndim != 2 + int(fused):
                raise ValueError("Native TP fields require physical matrix or fused expert views.")
            tp = weight.device_mesh["tp"]
            view = NativeWeightPartition(
                tuple(weight.shape[int(fused) :]),
                dim,
                tp.get_local_rank(),
                tp.size(),
                placement.split_factor if isinstance(placement, _StridedShard) else 1,
            )
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
                if not repair_backend_supports_tp(quantizer.backend):
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
