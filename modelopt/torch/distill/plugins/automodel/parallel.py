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

"""Preserve quantizer ownership through native Torch/AutoModel partitioning."""

import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer

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


def _quantizers_detached(model):
    from modelopt.torch.quantization.plugins.automodel import _quantizers_detached as native_context

    return native_context(model)


def _tp_quantizer_layouts(model):
    """Plan native field partitions from each actual Torch TP weight placement."""
    from modelopt.torch.quantization.ggml.common import PINNED_PAYLOAD_PREFIX, pinned_packed_weight
    from modelopt.torch.quantization.ggml.registry import GGML_FORMAT_REGISTRY
    from modelopt.torch.quantization.nn import SequentialQuantizer

    plans, aliases = [], {}

    def shared(source, signature, value):
        previous = aliases.get(id(source))
        if previous is not None:
            if previous[0] != signature:
                raise ValueError("Shared native fields have incompatible TP layouts.")
            return previous[1]
        aliases[id(source)] = (signature, value)
        return value

    for module in model.modules():
        if not isinstance(module, QuantModule):
            continue
        weight = module._parameters.get("weight")
        container = getattr(module, "weight_quantizer", None)
        if not isinstance(weight, DTensor) or container is None:
            continue
        axes = weight.device_mesh.mesh_dim_names
        if not axes or "tp" not in axes:
            continue
        placement = weight.placements[axes.index("tp")]
        if not isinstance(placement, Shard):
            continue
        if weight.ndim != 2:
            raise ValueError("Native TP field planning requires a matrix weight.")
        dim = placement.dim % weight.ndim
        mesh = weight.device_mesh["tp"]
        shape = tuple(weight.shape)
        width, remainder = divmod(shape[dim], mesh.size())
        if remainder:
            raise ValueError("Native TP repair requires even weight partitions.")
        start = mesh.get_local_rank() * width
        local_shape = tuple(width if axis == dim else size for axis, size in enumerate(shape))
        members = list(container) if isinstance(container, SequentialQuantizer) else [container]
        for quantizer in members:
            if not isinstance(quantizer, TensorQuantizer):
                raise ValueError(
                    "This TP weight container needs an explicit physical-view adapter."
                )
            if not quantizer.is_enabled:
                continue
            if not quantizer.fake_quant:
                raise ValueError(
                    "Packed Torch TP weights need a native packed-parameter partitioner."
                )
            from modelopt.torch.quantization.repair import repair_backend_supports_tp

            if not repair_backend_supports_tp(quantizer.backend):
                raise ValueError(
                    f"Backend {quantizer.backend!r} TP repair requires a native field partitioner."
                )
            native_pin = getattr(quantizer, "_native_weight_meta", None)
            if native_pin is not None and tuple(native_pin["source_shape"]) != local_shape:
                raise ValueError(
                    "Partition the native code payload before changing this checkpoint's TP layout."
                )
            fields = {}
            ggml = quantizer.backend == "ggml"
            if ggml:
                fmt = GGML_FORMAT_REGISTRY[quantizer.num_bits]
                blocks = {1: fmt.block_size}
            else:
                blocks = {
                    axis % 2: size
                    for axis, size in (quantizer.block_sizes or {}).items()
                    if isinstance(axis, int) and size is not None
                }
            if any(shape[axis] % block for axis, block in blocks.items()) or width % blocks.get(
                dim, 1
            ):
                raise ValueError(
                    "TP partitions must align with complete native quantization blocks."
                )
            grid = tuple(size // blocks.get(axis, 1) for axis, size in enumerate(shape))
            local_grid = tuple(size // blocks.get(axis, 1) for axis, size in enumerate(local_shape))

            def field(name, block=False):
                source = getattr(quantizer, name, None)
                if source is None:
                    return False
                value = source.to_local() if isinstance(source, DTensor) else source
                sharded = False
                signature = ("replicated", tuple(value.shape))
                if block and value.numel() in (grid[0] * grid[1], local_grid[0] * local_grid[1]):
                    if value.numel() == grid[0] * grid[1]:
                        value = (
                            value.reshape(grid)
                            .narrow(dim, start // blocks.get(dim, 1), local_grid[dim])
                            .clone()
                        )
                    else:
                        value = value.reshape(local_grid)
                    if name == "_amax" and quantizer.is_static_block_quant:
                        target = (
                            (-1, 1)
                            if set(blocks) == {1}
                            else tuple(
                                number
                                for axis, size in enumerate(local_grid)
                                for number in ((size, 1) if axis in blocks else (size,))
                            )
                        )
                        value = value.reshape(target)
                    sharded = True
                    signature = ("block", grid, dim, start, width, tuple(value.shape))
                elif value.ndim == 2 and value.shape[dim] != 1:
                    if value.shape[dim] == shape[dim]:
                        value = value.narrow(dim, start, width).clone()
                    elif value.shape[dim] != width:
                        raise ValueError(
                            "Native TP field does not match its physical weight shard."
                        )
                    sharded = True
                    signature = ("axis", dim, start, width, tuple(value.shape))
                fields[name] = shared(source, signature, value)
                return sharded

            amax_sharded = field("_amax", block=bool(blocks) and quantizer.is_static_block_quant)
            global_sharded = field("_global_amax")
            scale_sharded = field("_scale", block=bool(blocks))
            field("_double_scale")
            if ggml:
                payload = pinned_packed_weight(quantizer, quantizer.num_bits)
                if payload is not None:
                    if (
                        payload.numel()
                        == local_shape[0] * local_shape[1] // fmt.block_size * fmt.block_bytes
                    ):
                        packed = payload.reshape(
                            local_shape[0], local_shape[1] // fmt.block_size, fmt.block_bytes
                        )
                    else:
                        packed = payload.reshape(
                            shape[0], shape[1] // fmt.block_size, fmt.block_bytes
                        )
                        packed = packed.narrow(
                            dim,
                            start if dim == 0 else start // fmt.block_size,
                            width if dim == 0 else width // fmt.block_size,
                        ).clone()
                    fields[PINNED_PAYLOAD_PREFIX + quantizer.num_bits] = packed
            elif not quantizer.is_mx_format and quantizer._amax is None:
                raise ValueError("Native TP repair requires calibrated non-MX weight thresholds.")
            layouts = {
                "delta": amax_sharded,
                "shadow": True,
                "global_delta": True,
                "local_delta": True,
                "deltas.scale": True if blocks else amax_sharded or scale_sharded,
                "deltas.double_scale": global_sharded,
            }
            plans.append(
                (quantizer, fields, layouts, bool(blocks) and quantizer.is_static_block_quant)
            )
    return plans
