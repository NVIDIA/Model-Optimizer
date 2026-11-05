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

"""Optional AutoModel execution adapter for local quantization distillation."""

from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.distributed as dist

from ...local_units import ModuleLocalDistillationAdapter, _valid_token_mask, select_output
from .families import native_family
from .packing import prepare_local_batch
from .runners import native_fast_replay

__all__ = ["AutoModelLocalDistillationAdapter"]


class AutoModelLocalDistillationAdapter(ModuleLocalDistillationAdapter):
    """Use AutoModel's realized meshes and model-owned packed/CP input contract.

    Pass the existing AutoModel MeshContext instead of duplicating parallelism
    sizes in the distillation configuration. Importing this class does not import
    AutoModel or its optional CUDA dependencies.
    """

    def __init__(self, mesh_context=None, *, teacher_pipeline=None, padding_token_id=0):
        """Use existing AutoModel infrastructure without constructing new meshes."""
        self.mesh_context = mesh_context
        self.teacher_pipeline = teacher_pipeline
        self.padding_token_id = padding_token_id
        self._runtime = None
        self._pipeline = None
        self._family = None
        self._coverage_valid_mask = None

    @property
    def allows_empty_stage(self):
        """Head/embedding-only PP ranks still participate in execution and saves."""
        return self.mesh_context is not None and self.mesh_context.pp_size > 1

    def _default_unit_paths(self, student, selection):
        family = native_family(student)
        paths = (
            family.units(student, selection)
            if family is not None
            else super()._default_unit_paths(student, selection)
        )
        candidates = list(dict.fromkeys(paths))
        return [
            path
            for path in candidates
            if not any(path.startswith(parent + ".") for parent in candidates if parent != path)
        ]

    def resolve_units(self, student, teacher, selection):
        """Resolve global module names on their owning physical pipeline stage."""
        if not self.allows_empty_stage:
            units, error = [], None
            try:
                units = super().resolve_units(student, teacher, selection)
            except (ValueError, AttributeError) as exc:
                error = str(exc)
            self.synchronize_error(error)
            return units
        from .parallel import _all_rank_values
        from .pipeline import _validate_pipeline

        units, error = [], None
        try:
            _validate_pipeline(self.teacher_pipeline, teacher)
            if selection.micro_batch_size not in (None, self.teacher_pipeline.pp_microbatch_size):
                raise ValueError("Repair microbatch must match AutoPipeline.pp_microbatch_size.")
            if selection.units is not None:
                names = dict(student.named_modules())
                owned = [path for path in selection.units if path in names]
                if "" in owned:
                    raise ValueError("Use globally named decoder/submodule units for PP repair.")
                if owned:
                    units = super().resolve_units(
                        student, teacher, selection.model_copy(update={"units": owned})
                    )
            else:
                paths = self._default_unit_paths(student, selection)
                if paths:
                    units = super().resolve_units(
                        student, teacher, selection.model_copy(update={"units": paths})
                    )
        except (ValueError, AttributeError) as exc:
            error = str(exc)
        values = _all_rank_values(
            self.mesh_context, {"error": error, "names": [unit.name for unit in units]}
        )
        errors = [value["error"] for value in values if value["error"]]
        found = {name for value in values for name in value["names"]}
        if errors or not found or (selection.units is not None and found != set(selection.units)):
            raise ValueError(
                "Invalid PP repair ownership: "
                + "; ".join(errors or ["selected units are missing from the pipeline"])
            )
        return units

    def synchronize_error(self, error):
        """Reject rank-local input/preparation failures before model collectives."""
        from .parallel import _all_rank_values

        errors = sorted({value for value in _all_rank_values(self.mesh_context, error) if value})
        if errors:
            raise ValueError("; ".join(errors))

    def synchronize_microbatch(self, available, error):
        """Agree on iterator progress before teacher/model collective execution."""
        from .parallel import _all_rank_values

        states = _all_rank_values(self.mesh_context, (available, error))
        errors = sorted({message for _, message in states if message})
        if errors:
            raise ValueError("; ".join(errors))
        if any(ready for ready, _ in states) and not all(ready for ready, _ in states):
            raise ValueError(
                "Different microbatch schedules across model ranks; data exhausted unevenly."
            )

    def validate_preflight(self, report):
        """Propagate unsupported fields to every model rank before mutation."""
        from .parallel import _all_rank_values

        errors = _all_rank_values(self.mesh_context, report.unsupported)
        problems = sorted({problem for values in errors for problem in values})
        if problems:
            raise ValueError("Unsupported local quantization repair: " + "; ".join(problems))

    def prepare(self, student, teacher, units, config):
        """Resolve ownership from the caller's realized parallelism context."""
        self._family = native_family(student)
        self._configure_native_attention(student)
        self._configure_native_attention(teacher)
        self._runtime = None
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return None
        if self.mesh_context is None or self.mesh_context.device_mesh is None:
            raise ValueError("Distributed AutoModel repair requires its realized MeshContext.")
        from .runtime import AutoModelRuntime

        error = None
        try:
            self._runtime = AutoModelRuntime(self.mesh_context, config)
        except ValueError as exc:
            error = str(exc)
        self.synchronize_error(error)
        assert self._runtime is not None
        self._runtime.prepare_quantizers(student)
        if self.allows_empty_stage:
            from .pipeline import _TeacherPipeline

            if config.micro_batch_size not in (None, self.teacher_pipeline.pp_microbatch_size):
                raise ValueError("Repair microbatch must match AutoPipeline.pp_microbatch_size.")
            self._pipeline = _TeacherPipeline(self.teacher_pipeline, teacher)
            self._runtime.micro_batch_size = self.teacher_pipeline.pp_microbatch_size
        return self._runtime

    def parallelize(self, model):
        """Preserve PTQ children while using AutoModel's native model parallelizer."""
        if self.mesh_context is None:
            raise ValueError("Parallelization requires a realized AutoModel MeshContext.")
        from nemo_automodel.components.distributed.model_parallelizer import parallelize_model

        from .parallel import _quantizers_detached

        with _quantizers_detached(model):
            return parallelize_model(model, self.mesh_context)

    def configure_quantization(self, model):
        """Bind calibration statistics to the actual weight-replica groups."""
        self._configure_native_attention(model)
        if self.mesh_context is None or not dist.is_initialized():
            return
        from modelopt.torch.quantization.nn import QuantModule
        from modelopt.torch.quantization.plugins.automodel import _QuantAutoModelExperts
        from modelopt.torch.utils.distributed import ParallelState

        from .runtime import _submesh

        mesh = self.mesh_context
        dp = _submesh(mesh.device_mesh, "dp_cp") or _submesh(mesh.device_mesh, "dp")
        tp = _submesh(mesh.device_mesh, "tp")
        ep = _submesh(mesh.moe_mesh, "ep")
        expert_data = _submesh(mesh.moe_mesh, "ep_shard")

        def group(axis):
            return axis.get_group() if axis is not None and axis.size() > 1 else -1

        for module in model.modules():
            if isinstance(module, QuantModule):
                expert = isinstance(module, _QuantAutoModelExperts) and mesh.ep_size > 1
                module.parallel_state = ParallelState(
                    data_parallel_group=group(expert_data if expert else dp),
                    tensor_parallel_group=group(None if expert else tp),
                    expert_model_parallel_group=group(ep if expert else None),
                )

    def _configure_native_attention(self, model):
        """Restore model-owned CP transport after generic AutoModel loading."""
        family = native_family(model)
        if self.mesh_context is None or self.mesh_context.cp_size <= 1 or family is None:
            return
        family.configure_context_parallel(model, self.mesh_context)

    def publish(self, student, config, metrics):
        """Publish native DCP weights and rank-owned ModelOpt fields atomically."""
        from .checkpoint import publish

        return publish(student, self.mesh_context, config, metrics)

    def restore(self, model, directory):
        """Restore a native repaired checkpoint into a matching distributed skeleton."""
        from .checkpoint import restore

        return restore(model, self.mesh_context, directory)

    @contextmanager
    def observe_replay(self, module, training):
        """Count valid routed assignments before EP gather, once per data lane."""
        from modelopt.torch.quantization.plugins.automodel import _QuantAutoModelExperts

        counts, handles = {}, []

        def count(name, child, args, kwargs):
            mask = kwargs.get("token_mask", args[1] if len(args) > 1 else None)
            indices = kwargs.get("indices", args[3] if len(args) > 3 else None)
            if mask is None or indices is None:
                raise ValueError("Native expert coverage requires routing indices and token_mask.")
            mask = mask.bool().reshape(-1)
            if self._coverage_valid_mask is not None:
                mask = mask & self._coverage_valid_mask.reshape(-1)
            assignments = indices.reshape(-1).to(torch.int64)
            valid = mask.reshape(-1, 1).expand(-1, indices.shape[-1]).reshape(-1)
            contributions = (valid & assignments.ge(0)).to(torch.int64)
            # Static output/assignment shapes avoid bincount and masked-index
            # allocation synchronizations in the hot replay path.
            counts[name].scatter_add_(0, assignments.clamp_min(0), contributions)

        try:
            if training:
                for name, child in module.named_modules():
                    if isinstance(child, _QuantAutoModelExperts):
                        weight = child._parameters["gate_and_up_projs"]
                        counts[name] = torch.zeros(
                            weight.shape[0], device=weight.device, dtype=torch.int64
                        )
                        handles.append(
                            child.register_forward_pre_hook(
                                lambda child, args, kwargs, name=name: count(
                                    name, child, args, kwargs
                                ),
                                with_kwargs=True,
                            )
                        )
            yield counts
        finally:
            for handle in handles:
                handle.remove()

    def reduce_coverage(self, counts):
        """Reduce accumulated routed assignments only at reporting boundaries."""
        from .runtime import _sum

        if self._runtime is not None:
            for value in counts.values():
                _sum(value, self._runtime.dp_cp)

    def snapshot_unit_state(self, unit, module, batch):
        """Snapshot native side inputs before forward clears mutable gate state."""
        if self._family is None:
            return None
        state = self._family.snapshot_unit_state(module)
        return self.snapshot_inputs((), state)[1] if state is not None else None

    def replay_unit(self, module, args, kwargs, replay_state=None):
        """Select family execution inside the established FSDP ownership window."""
        if self._family is not None:
            return self._family.replay_unit(module, args, kwargs, replay_state)
        return native_fast_replay(module, args, kwargs)

    def replay(self, module, record):
        """Replay through the FSDP owner when its child weights are sharded."""
        previous = self._coverage_valid_mask
        self._coverage_valid_mask = record.valid_mask
        try:
            if self._runtime is not None:
                return self._runtime.replay(module, record, self)
            args, kwargs = self.snapshot_inputs(record.args, record.kwargs)
            return select_output(
                self.replay_unit(module, args, kwargs, record.replay_state), record.output_selector
            )
        finally:
            self._coverage_valid_mask = previous

    def teacher_microbatch(self, teacher, units, batch, consume):
        """Replay all valid roles through the shared native packing/CP contract."""
        if not isinstance(batch, dict) or "input_ids" not in batch:
            return super().teacher_microbatch(teacher, units, batch, consume)
        valid = _valid_token_mask(batch)
        if valid is None:
            valid = torch.ones_like(batch["input_ids"], dtype=torch.bool)
        # Labels supply layout metadata only; repair includes every valid token.
        batch = {**batch, "labels": torch.where(valid, batch["input_ids"], -100)}
        context, local = prepare_local_batch(
            teacher, batch, self.mesh_context, padding_token_id=self.padding_token_id
        )
        with context():
            if self._pipeline is not None:
                return self._pipeline.run(teacher, units, local, self, consume)
            return super().teacher_microbatch(teacher, units, local, consume)

    def close_pipeline(self):
        """Restore the caller's original teacher schedule and stage metadata."""
        if self._pipeline is not None:
            self._pipeline.close()
            self._pipeline = None

    def capture(self, unit, args, kwargs, output, batch):
        """Align a native flat THD mask with its boundary's token dimensions."""
        record = super().capture(unit, args, kwargs, output, batch)
        mask = record.valid_mask
        token_shape = record.target.shape[:-1]
        if (
            mask is not None
            and mask.shape != token_shape
            and mask.numel() == record.target.numel() // record.target.shape[-1]
        ):
            record = replace(record, valid_mask=mask.reshape(token_shape))
        return record
