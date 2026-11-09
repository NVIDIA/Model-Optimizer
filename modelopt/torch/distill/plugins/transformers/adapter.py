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

"""Standalone Transformers local repair with native Torch DP, TP, EP, CP and PP."""

from contextlib import contextmanager
from dataclasses import replace

import torch
import torch.distributed as dist

from ...local_parallel import _all_rank_values
from ...local_runtime import DistributedLocalRuntime, RealizedMeshContext, _size, _submesh
from ...local_units import ModuleLocalDistillationAdapter, _valid_token_mask, select_output
from .families import transformers_family
from .packing import empty_document_batch

__all__ = ["TransformersLocalDistillationAdapter"]


def _data_view(mesh, axes, alias):
    if not axes:
        return None
    view = mesh[tuple(axes)]
    return view._flatten(alias) if len(axes) > 1 else view


class _TransformersRuntime(DistributedLocalRuntime):
    def __init__(self, mesh_context, config):
        super().__init__(mesh_context, config)
        if _size(self.ep) > 1 and config.tune_activations:
            raise ValueError("HF expert-parallel repair does not support activation-scale tuning.")
        self.objective_tp_units = {}

    def _reduce_feature_statistics(self, totals):
        from ...local_runtime import _sum

        for name in self.unit_names:
            if name in self.objective_tp_units:
                # Features are disjoint, tokens are replicated across TP.
                # The runner divides gradients by this global count/energy.
                _sum(totals[name][:3], self.objective_tp_units[name])

    def expert_quantizers(self, unit):
        # The HF quantization plugin is optional until a Transformers model is used.
        from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

        return {
            id(quantizer)
            for module in unit.modules()
            if isinstance(module, _QuantFusedExperts) and _size(self.ep) > 1
            for container in module.iter_weight_quantizers_for_calibration()
            for quantizer in container.modules()
        }


class TransformersLocalDistillationAdapter(ModuleLocalDistillationAdapter):
    """Repair HF models using caller-owned Torch or native Transformers meshes.

    ``device_mesh`` is an existing Torch mesh or native Transformers mesh manager.
    Native TP/EP/FSDP uses the manager's overlapping dense and expert meshes.
    Models may already be partitioned, or use ``parallelize`` after PTQ.
    Scoped native CP requires full causal, unpadded single-document sequences;
    other inputs retain faithful full-context execution.
    """

    def __init__(self, device_mesh=None, *, family_adapter=None):
        """Use existing meshes without creating process groups."""
        self.family_adapter = family_adapter
        self.student = None
        self._coverage_valid_mask = None
        self.mesh = device_mesh
        self._runtime = None
        self._native_parallel_metadata = {}
        self._cp_full_context = False
        self._cp_modes = set()
        self._cp_activation_compatible = {}
        self.context = None
        if device_mesh is not None and callable(getattr(device_mesh, "get_mesh", None)):
            from .parallel import native_mesh_context

            self.context = native_mesh_context(device_mesh)
            self.mesh = self.context.device_mesh
            return
        if device_mesh is not None:
            names = device_mesh.mesh_dim_names or ()
            data_axes = [
                name for name in ("dp", "fsdp", "dp_replicate", "dp_shard") if name in names
            ]
            if not set(names).issubset(
                {"pp", "dp", "cp", "ep", "tp", "fsdp", "dp_replicate", "dp_shard"}
            ):
                raise ValueError("Unsupported Transformers repair mesh axis.")
            if len(data_axes) > 1 and set(data_axes) != {"dp_replicate", "dp_shard"}:
                raise ValueError("Use dp, fsdp, or dp_replicate/dp_shard for data ownership.")
            if _size(_submesh(device_mesh, "ep")) > 1 and _size(_submesh(device_mesh, "tp")) > 1:
                raise ValueError("Simultaneous HF tensor and expert parallelism is unsupported.")
            data = _data_view(device_mesh, data_axes, "repair_data")
            data_context_axes = [*data_axes, *(["cp"] if "cp" in names else [])]
            data_context = _data_view(device_mesh, data_context_axes, "repair_data_context")
            self.context = RealizedMeshContext(
                device_mesh=device_mesh,
                moe_mesh=device_mesh if "ep" in names else None,
                process_group=None,
                data_mesh=data,
                data_context_mesh=data_context,
                tensor_mesh=_submesh(device_mesh, "tp"),
                expert_mesh=_submesh(device_mesh, "ep"),
                expert_replica_mesh=data_context,
                pipeline_mesh=_submesh(device_mesh, "pp"),
                context_mesh=_submesh(device_mesh, "cp"),
            )

    @property
    def mesh_context(self):
        """Expose realized placement to the shared timing/checkpoint runtime."""
        return self.context

    @property
    def allows_empty_stage(self):
        """Allow embedding/head-only PP ranks to join the same transactions."""
        return self.context is not None and _size(self.context.pipeline_mesh) > 1

    def synchronize_error(self, error):
        """Propagate validation failures before any rank enters model collectives."""
        errors = sorted({value for value in _all_rank_values(self.context, error) if value})
        if errors:
            raise ValueError("; ".join(errors))

    def synchronize_microbatch(self, available, error):
        """Require every model rank to execute the same microbatch schedule."""
        values = _all_rank_values(self.context, (available, error))
        self.synchronize_error(error)
        if any(ready for ready, _ in values) and not all(ready for ready, _ in values):
            raise ValueError("Different microbatch schedules across model ranks.")

    def validate_preflight(self, report):
        """Validate all selected physical quantizer fields before mutation."""
        self.synchronize_error("; ".join(report.unsupported) if report.unsupported else None)

    def _family(self, model):
        return self.family_adapter or transformers_family(model)

    def batch_size(self, batch):
        """Use the text sample axis rather than a multimodal RoPE coordinate axis."""
        if isinstance(batch, dict) and isinstance(batch.get("input_ids"), torch.Tensor):
            return batch["input_ids"].shape[0]
        return super().batch_size(batch)

    def microbatches(self, batch, size):
        """Preserve mRoPE's [coordinates, batch, sequence] position convention."""
        positions = batch.get("position_ids") if isinstance(batch, dict) else None
        if not isinstance(positions, torch.Tensor) or positions.ndim != 3:
            yield from super().microbatches(batch, size)
            return
        batch_size = self.batch_size(batch)
        if positions.shape[1] != batch_size:
            raise ValueError(
                "Multimodal position IDs must have shape [coordinates, batch, sequence]."
            )
        remaining = {key: value for key, value in batch.items() if key != "position_ids"}
        for index, microbatch in enumerate(super().microbatches(remaining, size)):
            start = index * size
            yield {**microbatch, "position_ids": positions[:, start : start + size]}

    def _default_unit_paths(self, student, selection):
        return self._family(student).unit_paths(student, selection)

    def resolve_units(self, student, teacher, selection):
        """Resolve local text units while preserving global PP ownership."""
        if not self.allows_empty_stage:
            value, error = None, None
            try:
                value = self._family(student).resolve_units(student, teacher, selection)
            except (ValueError, AttributeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            self.synchronize_error(error)
            return value
        paths = (
            self._default_unit_paths(student, selection)
            if selection.units is None
            else [path for path in selection.units if path in dict(student.named_modules())]
        )
        value, error = [], None
        if paths:
            try:
                value = self._family(student).resolve_units(
                    student, teacher, selection.model_copy(update={"units": paths})
                )
            except (ValueError, AttributeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        self.synchronize_error(error)
        found = set().union(*_all_rank_values(self.context, set(paths)))
        self.synchronize_error(
            "No repair units, or missing selected PP units."
            if not found or (selection.units is not None and found != set(selection.units))
            else None
        )
        return value

    def parallelize(self, model, *, initialize=None):
        """Use Transformers' own inference stages and native TP plans.

        PP transports the teacher forward only; student gradients are local to
        their independent repair units and need no inter-stage backward.
        A native ``initialize(model)`` callback can materialize a meta model
        after TP/EP ownership is established and before FSDP wraps its storage.
        """
        if self.mesh is None:
            raise ValueError("parallelize requires an explicit device mesh.")
        assert self.context is not None
        if self.context.mesh_manager is not None:
            from .parallel import parallelize_native_transformers

            self._native_parallel_metadata[id(model)] = parallelize_native_transformers(
                model, self.context, initialize=initialize
            )
            return model
        if initialize is not None:
            raise ValueError("An initialization callback requires the native HF mesh manager.")
        tp, pp = _submesh(self.mesh, "tp"), _submesh(self.mesh, "pp")
        ep = _submesh(self.mesh, "ep")
        if _size(ep) > 1:
            raise ValueError("Expert parallelism requires the native HF mesh manager.")
        if _size(tp) > 1:
            from modelopt.torch.quantization.repair_partition import prepare_packed_weights_for_tp

            from .parallel import validate_hf_tensor_parallelism

            error = None
            try:
                validate_hf_tensor_parallelism(model, tp)
            except (ValueError, RuntimeError, AttributeError, KeyError, NotImplementedError) as exc:
                error = f"{type(exc).__name__}: {exc}"
            self.synchronize_error(error)
            prepare_packed_weights_for_tp(model)
        if _size(tp) > 1:
            from .parallel import apply_hf_tensor_parallelism

            apply_hf_tensor_parallelism(model, tp)
        if _size(pp) > 1:
            from transformers.distributed.pipeline_parallel import apply_pipeline_parallelism

            apply_pipeline_parallelism(model, pp)
        return model

    def prepare(self, student, teacher, units, config):
        """Share Torch objective and gradient reductions with native repair."""
        self.student = student
        self._cp_modes.clear()
        if _size(_submesh(self.mesh, "cp")) > 1:
            from modelopt.torch.quantization.repair_partition import (
                requires_full_context_activations,
            )

            self._cp_activation_compatible = {
                id(model): not requires_full_context_activations(model)
                for model in (student, teacher)
            }
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return None
        if self.context is None:
            raise ValueError("Distributed Transformers repair needs its realized device mesh.")
        self._runtime = _TransformersRuntime(self.context, config)
        self._runtime.prepare_quantizers(student)
        return self._runtime

    def replay_unit(self, module, args, kwargs, replay_state=None):
        """Replay native HF math with differentiable format scale kernels."""
        from modelopt.torch.quantization.repair_formats import native_scale_execution

        with native_scale_execution():
            return self._family(self.student).replay_unit(module, args, kwargs, replay_state)

    def snapshot_unit_state(self, unit, module, batch):
        """Capture family-owned side inputs without coupling them to optimization."""
        return self._family(self.student).snapshot_unit_state(unit, module, batch)

    def capture(self, unit, args, kwargs, output, batch):
        """Let the family describe structured outputs and modality-specific validity."""
        record = self._family(self.student).capture(unit, args, kwargs, output, batch)
        if self._runtime is not None:
            assert self.student is not None
            module = self.student.get_submodule(unit.student_path)
            mesh = self._family(self.student).objective_tp_mesh(module, record.target)
            if mesh is not None:
                if self._runtime.tp is None or not torch.equal(mesh.mesh, self._runtime.tp.mesh):
                    raise ValueError("A unit's TP output ownership differs from the repair mesh.")
                self._runtime.objective_tp_units[unit.name] = mesh
        if record.valid_mask is None and isinstance(batch, dict):
            tokens = batch.get("input_ids")
            if isinstance(tokens, torch.Tensor):
                record = replace(record, valid_mask=torch.ones_like(tokens, dtype=torch.bool))
        if self._cp_full_context:
            cp = _submesh(self.mesh, "cp")
            mask = record.valid_mask
            if mask is None:
                mask = torch.ones(
                    record.target.shape[:-1], device=record.target.device, dtype=torch.bool
                )
            if not mask.ndim:
                raise ValueError("CP repair requires a family-owned token validity axis.")
            # The fallback retains complete recurrent/multimodal inputs. Only
            # owned targets enter MSE; CP reductions recover the exact objective.
            owned = torch.arange(mask.shape[-1], device=mask.device).tensor_split(cp.size())[
                cp.get_local_rank()
            ]
            valid = torch.zeros_like(mask, dtype=torch.bool)
            valid[..., owned] = mask[..., owned]
            record = replace(record, valid_mask=valid)
        return record

    def forward_teacher(self, teacher, args, kwargs):
        """Use the family's task entry point while retaining root-level hooks."""
        return self._family(teacher).forward_teacher(teacher, args, kwargs)

    def replay(self, module, record):
        """Preserve local parameter ownership and validity during student replay."""
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

    @contextmanager
    def observe_replay(self, module, training):
        """Count routed assignments before any model-parallel duplication."""
        from modelopt.torch.quantization.plugins.huggingface import _QuantFusedExperts

        counts, handles = {}, []
        dispatched = self._dispatch_coverage()

        def count(name, args, kwargs):
            indices = kwargs.get("top_k_index", args[1] if len(args) > 1 else None)
            if indices is None:
                raise ValueError("HF expert coverage requires routing indices.")
            valid = self._coverage_valid_mask
            if dispatched:
                from torch.distributed.tensor import Shard

                assert self._runtime is not None
                tp = self._runtime.tp
                # Match native dispatch's token selection exactly, including
                # uneven/empty chunks. Each TP-replicated token is counted once.
                indices = Shard(0)._split_tensor(indices, tp.size(), with_padding=False)[0][
                    tp.get_local_rank()
                ]
                if valid is not None:
                    valid = Shard(0)._split_tensor(
                        valid.reshape(-1), tp.size(), with_padding=False
                    )[0][tp.get_local_rank()]
            assignments = indices.reshape(-1).long()
            mask = assignments.ge(0) & assignments.lt(counts[name].numel())
            if valid is not None:
                valid = valid.reshape(-1, 1).expand_as(indices).reshape(-1)
                mask = mask & valid
            counts[name].scatter_add_(
                0, assignments.clamp(0, counts[name].numel() - 1), mask.long()
            )

        try:
            if training:
                for name, child in module.named_modules():
                    if isinstance(child, _QuantFusedExperts):
                        weight = getattr(child, child._first_proj_attr)
                        counts[name] = torch.zeros(
                            child.num_experts
                            * (
                                _size(self._runtime.ep)
                                if dispatched and self._runtime is not None
                                else 1
                            ),
                            device=weight.device,
                            dtype=torch.int64,
                        )
                        handles.append(
                            child.register_forward_pre_hook(
                                lambda module, args, kwargs, name=name: count(name, args, kwargs),
                                with_kwargs=True,
                            )
                        )
            yield counts
        finally:
            for handle in handles:
                handle.remove()

    def reduce_coverage(self, counts):
        """Reduce counters over distinct data/context lanes at report boundaries."""
        from ...local_runtime import _sum

        if self._runtime is not None:
            for value in counts.values():
                _sum(value, self._runtime.dp_cp)
                if self._dispatch_coverage():
                    _sum(value, self._runtime.tp)

    def prepare_batch(self, model, batch):
        """Convert compact document IDs into an HF block-causal attention mask."""
        batch = self._family(model).prepare_batch(model, batch)
        result = {
            key: value
            for key, value in batch.items()
            if key
            not in (
                "valid_mask",
                "_packed_seq_ids",
                "labels",
                "loss_mask",
                "sample_ids",
                "document_ids",
            )
        }
        attention = result.get("attention_mask")
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 4
            and not (attention.dtype == torch.bool or attention.is_floating_point())
        ):
            raise ValueError("A 4D attention mask must be Boolean or floating additive bias.")
        config = getattr(model, "config", None)
        if config is not None and callable(getattr(config, "get_text_config", None)):
            config = config.get_text_config()
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 4
            and attention.dtype == torch.bool
            and getattr(config, "_attn_implementation", None) == "eager"
        ):
            if not bool(attention.any(dim=-1).all()):
                raise ValueError(
                    "Fully masked Boolean query rows require an SDPA attention backend."
                )
            dtype = next(model.parameters()).dtype
            dtype = dtype if dtype.is_floating_point else torch.float32
            attention = torch.zeros_like(attention, dtype=dtype).masked_fill(
                ~attention, torch.finfo(dtype).min
            )
            result["attention_mask"] = attention
        documents = batch.get("_packed_seq_ids")
        if isinstance(documents, torch.Tensor):
            if documents.ndim != 2 or documents.shape != batch["input_ids"].shape:
                raise ValueError("Packed document IDs must match [batch, sequence] token IDs.")
            attention = result.get("attention_mask")
            if bool((documents > 1).any()) or (
                isinstance(attention, torch.Tensor) and attention.ndim == 4
            ):
                n = documents.shape[1]
                causal = torch.arange(n, device=documents.device)
                allowed = (
                    (documents[:, None, :, None] == documents[:, None, None, :])
                    & documents[:, None, :, None].gt(0)
                    & documents[:, None, None, :].gt(0)
                    & (causal[:, None] >= causal[None, :])
                )
                if isinstance(attention, torch.Tensor):
                    if attention.ndim == 2:
                        allowed = allowed & attention[:, None, None, :].bool()
                    elif attention.ndim == 4 and attention.dtype == torch.bool:
                        allowed = allowed & attention
                    elif attention.ndim != 4:
                        raise ValueError("Packed attention masks must have two or four dimensions.")
                # HF eager attention adds its mask to scores; a Boolean mask
                # would add zeros/ones rather than exclude document boundaries.
                if (
                    isinstance(attention, torch.Tensor)
                    and attention.ndim == 4
                    and attention.is_floating_point()
                ):
                    additive = attention.clone()
                else:
                    dtype = next(model.parameters()).dtype
                    dtype = dtype if dtype.is_floating_point else torch.float32
                    additive = torch.zeros(allowed.shape, device=documents.device, dtype=dtype)
                    if isinstance(attention, torch.Tensor) and attention.ndim == 4:
                        additive = additive.masked_fill(~attention, float("-inf"))
                result["attention_mask"] = torch.minimum(
                    additive,
                    additive.masked_fill(~allowed, torch.finfo(additive.dtype).min),
                )
            else:
                result["attention_mask"] = documents.ne(0) & (
                    attention.bool() if isinstance(attention, torch.Tensor) else True
                )
        return result

    def teacher_microbatch(self, teacher, units, batch, consume):
        """Stream documents without leaking recurrent or compressed state."""
        with self._family(teacher).repair_forward_context(teacher, units):
            return self._teacher_microbatch(teacher, units, batch, consume)

    def _teacher_microbatch(self, teacher, units, batch, consume):
        if not isinstance(batch, dict):
            with self._full_context_capture():
                return super().teacher_microbatch(teacher, units, batch, consume)
        family = self._family(teacher)
        if not family.document_isolation_required(teacher):
            return self._teacher_forward(teacher, units, batch, consume)
        documents, error = [], None
        try:
            model = self.student if self.student is not None else teacher
            documents = family.document_batches(model, batch)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.synchronize_error(error)
        # FSDP and model-parallel ranks must execute the same forward schedule,
        # even when their unique data lanes contain different document counts.
        slots = max(_all_rank_values(self.context, len(documents)))
        for index in range(slots):
            document = documents[index] if index < len(documents) else empty_document_batch(batch)
            self._teacher_forward(teacher, units, document, consume)

    @contextmanager
    def _full_context_capture(self):
        """Retain faithful native math when ring SDPA cannot express the model."""
        previous = self._cp_full_context
        self._cp_full_context = _size(_submesh(self.mesh, "cp")) > 1
        if self._cp_full_context:
            self._cp_modes.add("replicated_full_context")
        try:
            yield
        finally:
            self._cp_full_context = previous

    def execution_metadata(self):
        """Report replicated CP work separately from native sequence sharding."""
        metadata = {"context_parallel_modes": sorted(self._cp_modes)} if self._cp_modes else {}
        if self._dispatch_coverage():
            metadata["expert_coverage_id_space"] = "global"
        return metadata

    def _dispatch_coverage(self):
        return (
            self._native_parallel_metadata.get(id(self.student), {}).get("expert_mode")
            == "dispatch"
        )

    def _teacher_forward(self, teacher, units, batch, consume):
        """Execute one faithful teacher microbatch, optionally inside Torch CP."""
        prepared, error = None, None
        try:
            prepared = self.prepare_batch(teacher, batch)
        except (ValueError, TypeError, KeyError, IndexError, AttributeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        self.synchronize_error(error)
        assert prepared is not None
        cp = _submesh(self.mesh, "cp")
        native_cp = _size(cp) > 1
        inputs = None
        if native_cp:
            try:
                inputs = self._cp_inputs(teacher, {**batch, **prepared})
                native_cp = not self._family(teacher).document_isolation_required(teacher)
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                native_cp = False
            choices = _all_rank_values(self.context, native_cp)
            self.synchronize_error(
                "CP ranks must agree on native attention versus full-context execution."
                if any(choices) and not all(choices)
                else None
            )
        if not native_cp:
            with self._full_context_capture():
                return super().teacher_microbatch(
                    teacher, units, {**prepared, "valid_mask": _valid_token_mask(batch)}, consume
                )
        from .context_parallel import CPSequenceLayout

        assert inputs is not None
        tokens, positions, mask = inputs
        layout = CPSequenceLayout(cp, tokens.shape[1], device=tokens.device)
        self._cp_modes.add("native_sdpa")
        tokens = layout.shard(tokens, 1)
        positions = layout.shard(positions, 1)
        mask = layout.shard(mask, 1)
        # Full causal CP cannot accept the local 4D mask generated by HF.
        handles = [
            module.register_forward_pre_hook(self._causal_attention, with_kwargs=True)
            for model in (teacher, self.student)
            if model is not None
            for name, module in model.named_modules()
            if name.endswith("self_attn")
        ]
        try:
            with layout.attention():
                return super().teacher_microbatch(
                    teacher,
                    units,
                    {"input_ids": tokens, "position_ids": positions, "valid_mask": mask},
                    consume,
                )
        finally:
            for handle in handles:
                handle.remove()

    def _cp_inputs(self, teacher, batch):
        from modelopt.torch.quantization.repair_partition import requires_full_context_activations

        for model in (teacher, self.student):
            if model is None:
                continue
            dtype = next(
                (
                    parameter.dtype
                    for parameter in model.parameters()
                    if parameter.is_floating_point()
                ),
                None,
            )
            if dtype not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    "Native ring SDPA requires a half-precision model; use full context."
                )
            compatible = self._cp_activation_compatible.get(id(model))
            if compatible is None:
                compatible = not requires_full_context_activations(model)
            if not compatible:
                raise ValueError(
                    "Calibrated or dynamic activation fields require full-context execution."
                )
        supported = {
            "input_ids",
            "attention_mask",
            "position_ids",
            "valid_mask",
            "_packed_seq_ids",
            "use_cache",
            "labels",
            "loss_mask",
            "sample_ids",
            "document_ids",
        }
        if any(key not in supported and value is not None for key, value in batch.items()):
            raise ValueError("Native CP cannot discard model-specific forward inputs.")
        if any(
            not getattr(model, "supports_context_parallel", False)
            for model in (teacher, self.student)
            if model is not None
        ):
            raise ValueError("Teacher and student must support full causal context parallelism.")
        if any(
            getattr(getattr(model, "config", None), "_attn_implementation", None) != "sdpa"
            for model in (teacher, self.student)
            if model is not None
        ):
            raise ValueError("Torch CP requires SDPA on teacher and student.")
        tokens = batch["input_ids"]
        if not isinstance(tokens, torch.Tensor) or tokens.ndim != 2:
            raise ValueError("Torch CP requires [batch, sequence] token IDs.")
        documents, valid = batch.get("_packed_seq_ids"), _valid_token_mask(batch)
        attention, positions = batch.get("attention_mask"), batch.get("position_ids")
        for value in (documents, valid, attention, positions):
            if value is not None and (
                not isinstance(value, torch.Tensor) or value.shape != tokens.shape
            ):
                raise ValueError("Torch CP metadata must match [batch, sequence] token IDs.")
        if attention is not None and not bool(attention.eq(1).all()):
            raise ValueError("Torch CP cannot discard a custom attention mask.")
        if positions is not None and bool((positions[:, 1:] - positions[:, :-1] != 1).any()):
            raise ValueError("Torch CP cannot discard packed position resets.")
        if (documents is not None and bool((documents != 1).any())) or (
            valid is not None and not bool(valid.all())
        ):
            raise ValueError("Torch CP requires unpadded, single-document causal rows.")
        if positions is None:
            positions = torch.arange(tokens.shape[1], device=tokens.device).expand_as(tokens)
        return tokens.clone(), positions.clone(), torch.ones_like(tokens, dtype=torch.bool)

    @staticmethod
    def _causal_attention(module, args, kwargs):
        return args, {**kwargs, "attention_mask": None, "is_causal": True}

    def publish(self, student, config, metrics):
        """Publish topology-native Torch DCP weights and rank-owned quantizer fields."""
        from .checkpoint import publish

        return publish(student, self.context, config, metrics)

    def restore(self, model, directory):
        """Restore into a matching caller-provided distributed PTQ skeleton."""
        from .checkpoint import restore

        if self.context is None:
            raise ValueError("Native DCP restore requires the checkpoint's realized device meshes.")
        return restore(model, self.context, directory)
