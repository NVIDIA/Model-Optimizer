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

"""Mesh-owned local objectives and FP32 repair-gradient synchronization."""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule

from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.plugins.automodel import _QuantAutoModelExperts
from modelopt.torch.quantization.utils.core_utils import (
    _get_enclosing_fsdp_module,
    enable_weight_access_and_writeback,
    module_name_maps,
)

from ...local_units import select_output

__all__ = []


def _submesh(mesh, axis):
    if mesh is None:
        return None
    try:
        return mesh[axis]
    except (KeyError, RuntimeError):
        return None


def _size(mesh):
    return mesh.size() if mesh is not None else 1


def _sum(tensor, mesh):
    if _size(mesh) > 1:
        dist.all_reduce(tensor, group=mesh.get_group())
    return tensor


class AutoModelRuntime:
    """Reduce unique-token objectives separately from expert parameter owners."""

    def __init__(self, mesh_context, config):
        self.context = mesh_context
        self.dp = _submesh(mesh_context.device_mesh, "dp")
        self.dp_cp = _submesh(mesh_context.device_mesh, "dp_cp") or self.dp
        self.ep = _submesh(mesh_context.moe_mesh, "ep")
        self.ep_shard = _submesh(mesh_context.moe_mesh, "ep_shard")
        self.tp = _submesh(mesh_context.device_mesh, "tp")
        self.loss = config.loss
        if config.tune_activations and _size(self.tp) > 1:
            raise ValueError(
                "Activation-scale tuning needs an explicit TP activation-field layout."
            )
        self.micro_batch_size = config.micro_batch_size
        if _size(self.dp) != (mesh_context.dp_size or 1):
            raise ValueError(
                "AutoModel MeshContext must expose its realized data-parallel submesh."
            )
        self.local_batch_size, remainder = divmod(config.global_batch_size, _size(self.dp))
        if remainder or not self.local_batch_size:
            raise ValueError("Global batch size must divide exactly across AutoModel data lanes.")
        self.parameter_meshes = {}
        self.expert_parameters = set()
        self.tp_parameters = set()
        self.tp_quantizers = {}
        self.original_flags = {}
        self.owners = {}
        self.student = None
        self.names = None
        self.adapter = None
        self.unit_names = []

    def prepare_quantizers(self, student):
        """Align static native weight fields with actual TP parameter partitions."""
        if _size(self.tp) <= 1:
            return
        from .parallel import _all_rank_values, _tp_quantizer_layouts

        plans, error = [], None
        try:
            plans = _tp_quantizer_layouts(student)
        except ValueError as exc:
            error = str(exc)
        errors = sorted({value for value in _all_rank_values(self.context, error) if value})
        if errors:
            raise ValueError("; ".join(errors))
        for quantizer, fields, layouts, reset_blocks in plans:
            for name, value in fields.items():
                quantizer._set_buffer(name, value)
            if reset_blocks:
                for name in (
                    "_block_reshape_size",
                    "_original_shape",
                    "_padding",
                    "_slices",
                    "_amax_shape_for_export",
                ):
                    if hasattr(quantizer, name):
                        delattr(quantizer, name)
            quantizer._quantizer_cache = None
            self.tp_quantizers[id(quantizer)] = layouts

    def initialize(self, student, teacher, units, adapter, first_batch):
        self.student = student
        self.adapter = adapter
        self.unit_names = [unit.name for unit in units]
        self.names = module_name_maps(student)
        for unit in units:
            module = student.get_submodule(unit.student_path)
            self.owners[module] = _get_enclosing_fsdp_module(module, student, self.names)
        if (
            not any(isinstance(module, FSDPModule) for module in student.modules())
            and adapter._pipeline is None
        ):
            return
        self.original_flags = {
            name: parameter.requires_grad for name, parameter in student.named_parameters()
        }
        student.requires_grad_(False)
        # Initialize the actual root hierarchy before replaying a child owner.
        calibration = {
            q: q._if_calib
            for model in (student, teacher)
            for q in model.modules()
            if isinstance(q, TensorQuantizer)
        }
        try:
            for quantizer in calibration:
                quantizer._if_calib = False
            if adapter._pipeline is not None:
                adapter._pipeline.prime(student, teacher, first_batch, adapter)
            else:
                for model in (student, teacher):
                    adapter.teacher_microbatch(model, [], first_batch, lambda unit, record: None)
        finally:
            for quantizer, flag in calibration.items():
                quantizer._if_calib = flag

    def checkpoint_group(self):
        """Return exactly the ranks belonging to this model's realized mesh."""
        from .parallel import _model_group

        return _model_group(self.context)

    def gather_values(self, value):
        """Collect runtime metrics across the ranks owning this model."""
        from .parallel import _all_rank_values

        return _all_rank_values(self.context, value)

    def checkpoint_topology(self):
        """Describe rank placement so resume cannot silently change field ownership."""
        return {
            "axes": self.context.device_mesh.mesh_dim_names,
            "ranks": self.context.device_mesh.mesh.tolist(),
            "expert_ranks": self.context.moe_mesh.mesh.tolist()
            if self.context.moe_mesh is not None
            else None,
        }

    def prepare_policy(self, unit, policy, config):
        with enable_weight_access_and_writeback(unit, self.student, self.names, writeback=False):
            return policy.prepare(unit, config)

    def finalize_policy(self, unit, policy):
        with enable_weight_access_and_writeback(unit, self.student, self.names):
            policy.finalize_()

    def replay(self, module, record, adapter):
        args, kwargs = adapter.snapshot_inputs(record.args, record.kwargs)
        owner = self.owners.get(module)
        if owner is None or owner is module:
            return select_output(
                adapter.replay_unit(module, args, kwargs, record.replay_state),
                record.output_selector,
            )
        had_override = "forward" in owner.__dict__
        original = owner.__dict__.get("forward")
        object.__setattr__(
            owner,
            "forward",
            lambda *args, **kwargs: adapter.replay_unit(module, args, kwargs, record.replay_state),
        )
        try:
            return select_output(owner(*args, **kwargs), record.output_selector)
        finally:
            if had_override:
                object.__setattr__(owner, "forward", original)
            else:
                object.__delattr__(owner, "forward")

    def close(self):
        parameters = dict(self.student.named_parameters()) if self.student is not None else {}
        for name, flag in self.original_flags.items():
            parameters[name].requires_grad_(flag)
        self.original_flags.clear()
        self.owners.clear()
        self.names = None
        self.student = None
        if self.adapter is not None:
            self.adapter.close_pipeline()
        self.adapter = None

    def register(self, unit, policy):
        expert_quantizers = set()
        for module in unit.modules():
            if isinstance(module, _QuantAutoModelExperts) and _size(self.ep) > 1:
                expert_quantizers.update(
                    id(quantizer) for quantizer in module.iter_weight_quantizers_for_calibration()
                )
        for _, quantizer, state, _, _, _ in policy.adapters:
            expert = id(quantizer) in expert_quantizers
            for name, parameter in state.named_parameters():
                if not parameter.requires_grad:
                    continue
                meshes = [self.ep_shard if expert else self.dp_cp]
                layout = self.tp_quantizers.get(id(quantizer))
                sharded = layout.get(name) if layout is not None else None
                if sharded is False:
                    meshes.append(self.tp)
                if sharded:
                    self.tp_parameters.add(id(parameter))
                self.parameter_meshes[id(parameter)] = meshes
                if expert:
                    self.expert_parameters.add(id(parameter))
        for _, state, _, _ in policy._activations:
            for parameter in state.parameters():
                self.parameter_meshes[id(parameter)] = [self.dp_cp]

    def reduce_statistics(self, totals, training=False):
        from .parallel import _all_rank_values

        missing = sorted(set(self.unit_names) - set(totals))
        errors = _all_rank_values(self.context, missing)
        if any(errors):
            raise ValueError(f"Repair did not execute every selected boundary: {errors}")
        for values in totals.values():
            _sum(values, self.dp_cp)
        invalid = [
            name
            for name, values in totals.items()
            if values[2 if training and self.loss == "normalized_mse" else 1].item() <= 0
        ]
        errors = _all_rank_values(self.context, invalid)
        if any(errors):
            raise ValueError(f"Repair requires valid objective elements on every stage: {errors}")
        return totals

    def reduce_gradients(self, parameters):
        for parameter in parameters:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            for mesh in self.parameter_meshes[id(parameter)]:
                _sum(parameter.grad, mesh)

    def clip_gradients(self, parameters, max_norm):
        device = parameters[0].device
        expert = torch.zeros((), device=device)
        replicated = torch.zeros_like(expert)
        tp_sharded = torch.zeros_like(expert)
        has_experts = False
        for parameter in parameters:
            if id(parameter) in self.expert_parameters:
                expert += parameter.grad.float().square().sum()
                has_experts = True
            elif id(parameter) in self.tp_parameters:
                tp_sharded += parameter.grad.float().square().sum()
            else:
                replicated += parameter.grad.float().square().sum()
        if has_experts:
            _sum(expert, self.ep)
        if self.tp_parameters:
            _sum(tp_sharded, self.tp)
        norm = (expert + tp_sharded + replicated).sqrt()
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite local-repair gradient norm.")
        multiplier = (max_norm / (norm + 1e-6)).clamp_max(1)
        for parameter in parameters:
            parameter.grad.mul_(multiplier)
        return norm
