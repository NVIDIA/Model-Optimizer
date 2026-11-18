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

"""Field-owner reductions and bounded native FSDP weight lifetimes for Bypass."""

from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Shard
from torch.utils._pytree import tree_map

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.utils.core_utils import (
    _get_enclosing_fsdp_module,
    enable_weight_access_and_writeback,
    module_name_maps,
)

from ._bypass_parallel import _all_rank_values, _model_group, _size, prepare_quantizer_partitions


def _sum(value, mesh):
    if _size(mesh) > 1:
        dist.all_reduce(value, group=mesh.get_group())
    return value


class DistributedBypassRuntime:
    """Use unique data lanes separately from overlapping expert weight owners."""

    def __init__(self, context, config):
        self.context = context
        self.local_batch_size, remainder = divmod(config.global_batch_size, context.dp_size)
        if remainder or not self.local_batch_size:
            raise ValueError("Global batch must divide exactly over actual data lanes")
        if _size(context.pipeline_mesh) > 1:
            raise ValueError("Bypass requires a validated native pipeline transport")
        self.student = None
        self.owners, self.parameter_meshes, self.parameter_names = {}, {}, {}
        self.expert_parameters, self.tp_parameters = set(), set()
        self.flags, self.names, self.layouts = {}, None, {}
        self.feature_units = set()
        self.context_modes = set()

    def gather_values(self, value):
        return _all_rank_values(self.context, value)

    def synchronize_error(self, error):
        errors = sorted({value for value in self.gather_values(error) if value})
        if errors:
            raise ValueError("; ".join(errors))

    def checkpoint_group(self):
        return _model_group(self.context)

    def checkpoint_topology(self):
        return {
            "axes": self.context.device_mesh.mesh_dim_names,
            "ranks": self.context.device_mesh.mesh.tolist(),
            "expert_ranks": None
            if self.context.moe_mesh is None
            else self.context.moe_mesh.mesh.tolist(),
        }

    def initialize(self, student, teacher, units, adapter, batch):
        self.student, self.names = student, module_name_maps(student)
        self.layouts = self.context.tp_quantizer_layouts.get(id(student))
        if self.layouts is None:
            self.layouts = prepare_quantizer_partitions(student, self.context)
        for unit in units:
            module = student.get_submodule(unit.student_path)
            self.owners[module] = _get_enclosing_fsdp_module(module, student, self.names)
        if not any(isinstance(module, FSDPModule) for module in student.modules()):
            return
        self.flags = {
            name: parameter.requires_grad for name, parameter in student.named_parameters()
        }
        student.requires_grad_(False)
        calibration = {
            q: q._if_calib
            for model in (student, teacher)
            for q in model.modules()
            if isinstance(q, TensorQuantizer)
        }
        try:
            for q in calibration:
                q._if_calib = False
            for model in (student, teacher):
                adapter.teacher_microbatch(model, [], batch, lambda unit, record: None)
        finally:
            for q, flag in calibration.items():
                q._if_calib = flag

    def prepare_policy(self, plan, index, policy):
        with enable_weight_access_and_writeback(
            plan.units[index], self.student, self.names, writeback=False
        ):
            fields = policy.prepare_plan(plan, index)
        self.register(plan.units[index], policy)
        return fields

    def finalize_policy(self, unit, policy):
        policy.finalize_(
            weight_access=partial(
                enable_weight_access_and_writeback, unit, self.student, self.names
            ),
            rollback_writeback=isinstance(self.owners[unit], FSDPModule),
            synchronize_error=self.synchronize_error,
        )

    def register(self, unit, policy):
        assert self.student is not None
        experts = {
            id(q)
            for module in unit.modules()
            if getattr(module, "_is_expert_parallel", False)
            or getattr(module, "_modelopt_expert_parallel_mode", None)
            for q in module.modules()
            if isinstance(q, TensorQuantizer)
        }
        names = {id(q): name for name, q in self.student.named_modules()}
        metadata = self.context.parallel_metadata.get(id(self.student), {})
        for binding, state, _, _ in policy.adapters:
            quantizer = binding.quantizer
            prefix = names[id(quantizer)]
            for owner, ids in metadata.get("expert_global_ids", {}).items():
                if prefix.startswith(owner + "."):
                    parts = prefix[len(owner) + 1 :].split(".")
                    if len(parts) > 1 and parts[1].isdigit():
                        parts[1] = str(ids[int(parts[1])])
                        prefix = owner + "." + ".".join(parts)
            state.field_prefix = prefix
            expert = id(quantizer) in experts
            for name, parameter in state.named_parameters():
                meshes = [
                    self.context.expert_replica_mesh if expert else self.context.data_context_mesh
                ]
                sharded = self.layouts.get(id(quantizer), {}).get(name)
                if sharded is False:
                    meshes.append(self.context.tensor_mesh)
                if sharded:
                    self.tp_parameters.add(id(parameter))
                if expert:
                    self.expert_parameters.add(id(parameter))
                self.parameter_meshes[id(parameter)] = meshes
                self.parameter_names[id(parameter)] = f"{prefix}.{name}"

    def record(self, unit, record, *, native_context=False):
        target = record.target
        if isinstance(target, DTensor):
            if any(
                isinstance(p, Shard) and p.dim % target.ndim == target.ndim - 1
                for p in target.placements
            ):
                self.feature_units.add(unit.name)
            record = replace(record, target=target.to_local())
        cp = self.context.context_mesh
        if _size(cp) > 1:
            self.context_modes.add("native_sdpa" if native_context else "replicated_full_context")
            if not native_context:
                mask = record.valid_mask
                if mask is None:
                    mask = torch.ones(
                        record.target.shape[:-1], device=target.device, dtype=torch.bool
                    )
                if mask.ndim < 2:
                    raise ValueError("CP replay needs explicit [batch, sequence] token validity")
                indices = torch.arange(mask.shape[-1], device=mask.device).tensor_split(cp.size())[
                    cp.get_local_rank()
                ]
                owned = torch.zeros_like(mask, dtype=torch.bool)
                owned[..., indices] = mask[..., indices]
                record = replace(record, valid_mask=owned)
        return record

    def replay(self, module, record, adapter, training):
        if training:

            def differentiable(value):
                return (
                    value.detach().requires_grad_(True)
                    if isinstance(value, torch.Tensor) and value.is_floating_point()
                    else value
                )

            record = replace(
                record,
                args=tree_map(differentiable, record.args),
                kwargs=tree_map(differentiable, record.kwargs),
            )
        owner = self.owners.get(module)
        if owner is None or owner is module:
            output = adapter.replay(module, record)
        else:
            owned, original = "forward" in owner.__dict__, owner.__dict__.get("forward")
            owner.forward = lambda *args, **kwargs: adapter.replay(
                module, replace(record, args=args, kwargs=kwargs)
            )
            try:
                output = owner(*record.args, **record.kwargs)
            finally:
                if owned:
                    owner.forward = original
                else:
                    delattr(owner, "forward")
        return output.to_local() if isinstance(output, DTensor) else output

    def reduce_statistics(self, statistics, names):
        self.synchronize_error(
            "Missing selected teacher boundary" if set(statistics) != set(names) else None
        )
        for name in names:
            for value in statistics[name]:
                if name in self.feature_units:
                    _sum(value, self.context.tensor_mesh)
                _sum(value, self.context.data_context_mesh)
        self.synchronize_error(
            "No valid target elements"
            if any(count.item() <= 0 for _, count in statistics.values())
            else None
        )
        return statistics

    def _gradient_groups(self, parameters):
        groups = {}
        for parameter in parameters:
            meshes = self.parameter_meshes[id(parameter)]
            groups.setdefault(tuple(id(mesh) for mesh in meshes), (meshes, []))[1].append(parameter)
        return groups.values()

    def validate_gradient_schedules(self, units):
        """Check every reduction group's ordered fields before any training collective."""
        schedule, error = {}, None
        try:
            for unit, parameters in units.items():
                for meshes, fields in self._gradient_groups(parameters):
                    signature = tuple(
                        (self.parameter_names[id(p)], tuple(p.shape), str(p.dtype)) for p in fields
                    )
                    for mesh in meshes:
                        if _size(mesh) > 1:
                            ranks = tuple(mesh.mesh.reshape(-1).tolist())
                            schedule.setdefault((unit, ranks), []).append(signature)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        rank = dist.get_rank() if dist.is_initialized() else 0
        records = self.gather_values((rank, schedule, error))
        errors = sorted({message for _, _, message in records if message})
        if errors:
            raise ValueError("Invalid Bypass gradient reduction schedule: " + "; ".join(errors))
        by_rank = {owner: value for owner, value, _ in records}
        for _, value, _ in records:
            for key, signature in value.items():
                _, ranks = key
                if any(by_rank.get(owner, {}).get(key) != signature for owner in ranks):
                    raise ValueError("Bypass gradient reduction schedules differ across ranks")

    def reduce_gradients(self, parameters):
        # The complete metadata schedule was validated once, before the training loop.
        for meshes, fields in self._gradient_groups(parameters):
            presence = torch.tensor(
                [p.grad is not None for p in fields], device=fields[0].device, dtype=torch.int32
            )
            for mesh in meshes:
                if _size(mesh) > 1:
                    dist.all_reduce(presence, op=dist.ReduceOp.MAX, group=mesh.get_group())
            for parameter, present in zip(fields, presence.tolist(), strict=True):
                if not present:
                    continue
                if parameter.grad is None:
                    parameter.grad = torch.zeros_like(parameter)
                for mesh in meshes:
                    _sum(parameter.grad, mesh)

    def clip_gradients(self, parameters, max_norm):
        sums = [torch.zeros((), device=parameters[0].device) for _ in range(3)]
        for parameter in parameters:
            if parameter.grad is not None:
                group = (
                    0
                    if id(parameter) in self.expert_parameters
                    else 1
                    if id(parameter) in self.tp_parameters
                    else 2
                )
                # Bound temporary gradient squares even for a wide native field.
                for chunk in parameter.grad.reshape(-1).split(262144):
                    sums[group].add_(chunk.square().sum())
        _sum(sums[0], self.context.expert_mesh)
        _sum(sums[1], self.context.tensor_mesh)
        norm = sum(sums).sqrt()
        # Owner reductions make the norm identical on every rank; no object collective is needed.
        if not bool(torch.isfinite(norm)):
            raise ValueError("Nonfinite Bypass gradient norm")
        multiplier = (max_norm / (norm + 1e-6)).clamp_max(1)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad.mul_(multiplier)
        return norm

    def close(self):
        parameters = dict(self.student.named_parameters()) if self.student is not None else {}
        for name, flag in self.flags.items():
            parameters[name].requires_grad_(flag)
        self.flags.clear()
        self.owners.clear()
        self.student = self.names = None
