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

"""Distribute GGML expert updates over an existing FSDP2 data-parallel group."""

import hashlib
import json
from typing import cast

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

from modelopt.torch.utils.distributed import is_fsdp2_model

from ..nn import TensorQuantizer
from .common import pin_packed_weight, pinned_packed_weight, preserving_source_weights
from .registry import GGML_FORMAT_REGISTRY

__all__ = []


def _validate_parallel_experts(
    model, experts, projections, block_size, perc_damp, importance, hessian_input_dtype
):
    if (
        not dist.is_initialized()
        or dist.get_world_size() < 2
        or "cuda:nccl" not in dist.get_backend_config().split(",")
    ):
        raise ValueError("Parallel GPTQ experts require multiple FSDP2 ranks with NCCL.")
    errors = []
    contract: list = [
        (block_size, perc_damp, importance, preserving_source_weights(), hessian_input_dtype)
    ]
    storage_ranges, quantizers_seen = [], set()
    if not is_fsdp2_model(model):
        errors.append("FSDP2 is required")
    try:
        # Disabled activation quantizers skip amax collectives, so their ordering must agree too.
        contract.extend(
            (name, type(quantizer).__name__, quantizer.is_enabled)
            for name, quantizer in model.named_modules()
            if isinstance(quantizer, TensorQuantizer)
        )
        for name, module in experts:
            state = module.parallel_state
            if state.data_parallel_group.group not in (None, dist.group.WORLD):
                errors.append("Only the default data-parallel group is supported")
            if any(
                g.is_initialized() and g.world_size() > 1
                for g in (state.tensor_parallel_group, state.expert_model_parallel_group)
            ):
                errors.append("Tensor/expert parallel layouts are unsupported")
            for projection, weight, quantizers, _ in projections(module):
                local = weight.to_local() if isinstance(weight, DTensor) else weight
                if isinstance(weight, DTensor) and (
                    weight.device_mesh.ndim != 1
                    or weight.device_mesh.size() != dist.get_world_size()
                    or weight.placements != (Shard(0),)
                ):
                    errors.append("Only one-dimensional FSDP weight sharding is supported")
                if local.is_meta or weight.ndim != 3 or not weight.is_contiguous():
                    errors.append(
                        "Experts must have materialized contiguous three-dimensional weights"
                    )
                elif local.numel():
                    start = local.data_ptr()
                    stop = start + local.numel() * local.element_size()
                    if any(
                        device == local.device and max(start, a) < min(stop, b)
                        for device, a, b in storage_ranges
                    ):
                        errors.append("Aliased expert projections are unsupported")
                    storage_ranges.append((local.device, start, stop))
                choices = []
                for quantizer in quantizers:
                    if id(quantizer) in quantizers_seen:
                        errors.append("Shared expert quantizers are unsupported")
                    quantizers_seen.add(id(quantizer))
                    choices.append((quantizer.is_enabled, quantizer.backend, quantizer.num_bits))
                    if quantizer.is_enabled:
                        if weight.element_size() == 1:
                            errors.append("Packed source expert weights must be unpacked first")
                        fmt = GGML_FORMAT_REGISTRY.get(str(quantizer.num_bits))
                        if quantizer.backend != "ggml" or fmt is None:
                            errors.append("Parallel expert updates support GGML GPTQ only")
                        elif block_size % fmt.block_size or weight.shape[-1] % fmt.block_size:
                            errors.append(
                                "GPTQ and weight widths must align with the GGML block size"
                            )
                contract.append((name, projection, tuple(weight.shape), str(weight.dtype), choices))
    except Exception as exc:
        errors.append(str(exc))
    # All ranks reject together, before amax collection or weight updates can change state.
    device = torch.device("cuda", torch.cuda.current_device())
    invalid = torch.tensor(bool(errors), dtype=torch.int32, device=device)
    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
    digest = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).digest()
    packet = torch.tensor(list(digest), dtype=torch.uint8, device=device)
    gathered = torch.empty(
        dist.get_world_size() * packet.numel(), dtype=packet.dtype, device=device
    )
    dist.all_gather_into_tensor(gathered, packet)
    if invalid.item() or not torch.equal(
        gathered.view(-1, 32), packet.expand(dist.get_world_size(), -1)
    ):
        raise ValueError(
            "Unsupported or inconsistent parallel GPTQ expert layout: " + "; ".join(errors)
        )


def _update_parallel_experts(module, handles, block_size, perc_damp):
    rank, size = dist.get_rank(), dist.get_world_size()
    tasks = [
        (weight, q, handles.pop(id(q)))
        for weight, q in module.iter_weights_for_calibration()
        if id(q) in handles
    ]
    unreached = sum(handle.n_samples == 0 for _, _, handle in tasks)
    # Hessian reductions have already completed in the caller's unchanged collective order.
    # Free unowned statistics before updates; only one working expert is live per rank.
    for index, (_, _, handle) in enumerate(tasks):
        if index % size != rank:
            handle.free()
    failure = None
    failed = torch.zeros((), dtype=torch.int32, device=torch.cuda.current_device())
    try:
        for index, (weight, quantizer, handle) in enumerate(tasks):
            if index % size == rank:
                handle.update_weights(block_size, perc_damp, weight)
                handle.free()
                fmt = GGML_FORMAT_REGISTRY[quantizer.num_bits]
                packed = pinned_packed_weight(quantizer, fmt.name)
                expected = (weight.shape[0], weight.shape[1] // fmt.block_size, fmt.block_bytes)
                if (
                    packed is None
                    or tuple(packed.shape) != expected
                    or packed.dtype != torch.uint8
                    or packed.device != weight.device
                    or not packed.is_contiguous()
                ):
                    raise ValueError("Owner packed weight differs from the broadcast layout")
    except Exception as exc:
        failure = exc
        failed.fill_(1)
    finally:
        for _, _, handle in tasks:
            handle.free()
    # A local Python failure must not strand peers waiting for that owner's payload.
    dist.all_reduce(failed, op=dist.ReduceOp.MAX)
    if failed.item():
        raise RuntimeError(
            "Parallel GPTQ expert update failed on a data-parallel rank"
        ) from failure
    for index, (weight, quantizer, _) in enumerate(tasks):
        owner = index % size
        fmt = GGML_FORMAT_REGISTRY[quantizer.num_bits]
        packed = (
            cast("torch.Tensor", pinned_packed_weight(quantizer, fmt.name))
            if rank == owner
            else torch.empty(
                weight.shape[0],
                weight.shape[1] // fmt.block_size,
                fmt.block_bytes,
                dtype=torch.uint8,
                device=weight.device,
            )
        )
        dist.broadcast(packed, src=owner)
        pin_packed_weight(quantizer, fmt.name, packed)
        if rank != owner and not preserving_source_weights():
            chunk = (quantizer.backend_extra_args or {}).get(
                "decode_chunk_size", fmt.decode_chunk_size
            )
            weight.copy_(
                fmt.dequantize(
                    packed, torch.tensor(weight.shape), dtype=weight.dtype, block_chunk_size=chunk
                )
            )
    return len(tasks), unreached
