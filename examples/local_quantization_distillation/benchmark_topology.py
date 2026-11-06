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


"""Explicit parallel layouts and parameter ownership for repair measurements."""

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from modelopt.torch.distill.plugins.transformers import TransformersLocalDistillationAdapter

__all__ = ["data_lane", "parameter_layouts", "partition_models"]


def partition_models(args, student, teacher):
    """Apply explicit Transformers placement after calibrating unpartitioned weights."""
    if args.parameter_sharding == "replicated":
        # Frozen base weights need no DDP reducer; the repair runtime reduces DP fields.
        mesh = init_device_mesh("cuda", (args.dp,), mesh_dim_names=("dp",))
    else:
        # The native HF API is an optional dependency of this benchmark.
        from transformers.distributed.configuration_utils import DistributedConfig
        from transformers.distributed.utils import TransformersDeviceMesh

        config = DistributedConfig(
            tp_size=args.tp,
            fsdp_size=args.dp,
            pp_size=args.pp,
            ep_size=args.ep,
            fsdp_mixed_precision=True,
        )
        dense = init_device_mesh(
            "cuda",
            (args.pp, args.cp, args.dp, args.tp),
            mesh_dim_names=("pp", "cp", "fsdp", "tp"),
        )
        expert = init_device_mesh(
            "cuda",
            (args.pp, args.cp, config.efsdp_size, args.ep),
            mesh_dim_names=("pp", "cp", "efsdp", "ep"),
        )
        mesh = TransformersDeviceMesh(dense, expert)
        for model in (teacher, student):
            model.config.distributed_config = config
    adapter = TransformersLocalDistillationAdapter(mesh)
    for model in (teacher, student):
        adapter.parallelize(model)
    return student, teacher, adapter


def data_lane(adapter):
    """Return the unique-sequence lane; EP overlaps dense FSDP and TP ranks."""
    axis = adapter.context.data_mesh
    return axis.size(), axis.get_local_rank()


def parameter_layouts(model):
    """Record representative physical base-weight layouts without gathering tensors."""
    result, roles = [], set()
    for name, parameter in model.named_parameters():
        role = None
        if ".experts." in name and parameter.ndim == 3:
            role = "expert_down" if "down" in name else "expert_gate_up"
        elif ".self_attn." in name and parameter.ndim == 2:
            role = "attention_" + name.split(".self_attn.", 1)[1].split(".", 1)[0]
        if role is None or role in roles:
            continue
        roles.add(role)
        result.append(
            {
                "name": name,
                "role": role,
                "global_shape": list(parameter.shape),
                "local_shape": list(
                    parameter.to_local().shape
                    if isinstance(parameter, DTensor)
                    else parameter.shape
                ),
                "placements": [str(p) for p in parameter.placements]
                if isinstance(parameter, DTensor)
                else ["Replicate"],
                "mesh_axes": list(parameter.device_mesh.mesh_dim_names)
                if isinstance(parameter, DTensor)
                else [],
            }
        )
    return result
