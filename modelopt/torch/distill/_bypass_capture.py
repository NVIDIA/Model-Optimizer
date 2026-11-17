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

"""Disjoint boundaries and immediate teacher-to-student replay for ordinary Torch."""

import copy
import inspect
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch
from torch.utils._pytree import tree_map

from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.repair import _iq_quantizer, _router_paths, _storage_span


@dataclass(frozen=True)
class BypassUnit:
    """One disjoint semantic replay boundary in student and teacher."""

    name: str
    student_path: str
    teacher_path: str
    output_selector: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class CaptureRecord:
    """One current microbatch, released immediately after its boundary backward."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    target: torch.Tensor
    valid_mask: torch.Tensor | None = None
    output_selector: tuple[str | int, ...] = ()
    replay_state: Any = None


def select_output(output, selector=()):
    """Select a tensor output without guessing among model-specific side results."""
    for key in selector:
        output = output[key]
    if not selector and isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("A Bypass boundary must select a Tensor output")
    return output


def has_iq_weights(module):
    """Inspect quantizer metadata without slicing packed or sharded weights."""
    routers = _router_paths(module)
    return any(
        _iq_quantizer(q) is not None
        for path, child in module.named_modules()
        if isinstance(child, QuantModule)
        and not any(path == router or path.startswith(router + ".") for router in routers)
        for q in child.iter_weight_quantizers_for_calibration()
    )


def validate_unit_ownership(student, teacher, units):
    """Refuse overlapping boundaries, shared owners, and aliases of teacher storage."""
    if not units:
        raise ValueError(
            "No eligible Bypass boundaries; provide exact modules or quantize IQ weights"
        )
    paths = [unit.student_path for unit in units]
    routers = _router_paths(student)
    if any(path == router or path.startswith(router + ".") for path in paths for router in routers):
        raise ValueError("Bypass cannot select a frozen routing gate as a repair boundary")
    if any(
        a == b or not a or not b or a.startswith(b + ".") or b.startswith(a + ".")
        for i, a in enumerate(paths)
        for b in paths[:i]
    ):
        raise ValueError("Bypass boundaries overlap; select disjoint owning modules")
    _validate_student_owners(student, paths)
    teacher_storage = {}
    for tensor in [*teacher.parameters(), *teacher.buffers()]:
        span = _storage_span(tensor)
        if span is not None:
            key, start, stop = span
            teacher_storage.setdefault(key, []).append((start, stop))
    owned, teacher_owners = set(), set()
    for unit in units:
        try:
            module = student.get_submodule(unit.student_path)
            teacher_module = teacher.get_submodule(unit.teacher_path)
        except AttributeError as exc:
            raise ValueError(f"Boundary {unit.name!r} must exist in both models") from exc
        for tensor in [*teacher_module.parameters(), *teacher_module.buffers()]:
            if id(tensor) in teacher_owners:
                raise ValueError("Teacher Bypass boundaries share ownership")
            teacher_owners.add(id(tensor))
        for tensor in [*module.parameters(), *module.buffers()]:
            if id(tensor) in owned:
                raise ValueError("Student Bypass boundaries share ownership")
            owned.add(id(tensor))
            span = _storage_span(tensor)
            if span is not None:
                key, start, stop = span
                if any(start < end and begin < stop for begin, end in teacher_storage.get(key, ())):
                    raise ValueError("Student Bypass state aliases teacher storage")


def _validate_student_owners(student, paths):
    modules, tensors, storage = {}, {}, {}
    for path, module in student.named_modules(remove_duplicate=False):
        selected = any(not root or path == root or path.startswith(root + ".") for root in paths)
        if isinstance(module, TensorQuantizer):
            previous = modules.get(id(module))
            if previous is not None and (selected or previous):
                raise ValueError("Selected Bypass state shares a quantizer owner")
            modules[id(module)] = selected
        for tensor in (*module._parameters.values(), *module._buffers.values()):
            if tensor is None:
                continue
            previous = tensors.get(id(tensor))
            if previous is not None and (selected or previous):
                raise ValueError("Selected Bypass state shares a tensor owner")
            tensors[id(tensor)] = selected
            span = _storage_span(tensor)
            if span is not None:
                key, start, stop = span
                storage.setdefault(key, []).append((start, stop, selected))
    for spans in storage.values():
        any_end = selected_end = -1
        for start, stop, selected in sorted(spans):
            if start < (any_end if selected else selected_end):
                raise ValueError("Selected Bypass state overlaps another student's storage")
            any_end = max(any_end, stop)
            if selected:
                selected_end = max(selected_end, stop)


def _valid_token_mask(batch):
    if not isinstance(batch, dict):
        return None
    for name in ("valid_mask", "_packed_seq_ids"):
        if isinstance(batch.get(name), torch.Tensor):
            return batch[name].bool()
    if isinstance(batch.get("padding_mask"), torch.Tensor):
        return ~batch["padding_mask"].bool()
    attention = batch.get("attention_mask")
    if isinstance(attention, torch.Tensor):
        if attention.ndim == 2:
            return attention.bool()
        if attention.ndim == 4 and attention.dtype == torch.bool:
            return attention.any(-1).any(1)
        raise ValueError("Provide valid_mask when using additive attention biases")
    return None


def mse_statistics(prediction, target, valid_mask=None):
    """FP32 squared error sum and the exact count of valid feature elements."""
    if prediction.shape != target.shape:
        raise ValueError("Student and teacher boundary output shapes differ")
    difference = prediction.float() - target.float()
    if valid_mask is None:
        return difference.square().sum(), target.new_tensor(target.numel(), dtype=torch.int64)
    if valid_mask.ndim >= target.ndim or tuple(valid_mask.shape) != tuple(
        target.shape[: valid_mask.ndim]
    ):
        raise ValueError("Valid-token masks must match the boundary's token dimensions")
    features = math.prod(target.shape[valid_mask.ndim :])
    mask = valid_mask.to(device=target.device, dtype=torch.bool).reshape(
        *valid_mask.shape, *((1,) * (target.ndim - valid_mask.ndim))
    )
    return torch.where(mask, difference, 0).square().sum(), mask.sum(dtype=torch.int64) * features


class ModuleBypassAdapter:
    """Resolve ordinary callable Torch boundaries without a Transformers dependency."""

    feedforward_names = ("mlp", "moe", "ffn", "block_sparse_moe")

    def resolve_units(self, student, teacher, config):
        """Use explicit paths or disjoint named MLP/MoE owners containing IQ weights."""
        if config.layer_indices is not None:
            raise ValueError("layer_indices requires a decoder-aware Bypass adapter")
        paths = config.modules
        if paths is None:
            candidates = [
                name
                for name, module in student.named_modules()
                if name.rsplit(".", 1)[-1] in self.feedforward_names and has_iq_weights(module)
            ]
            paths = [
                name
                for name in candidates
                if not any(name.startswith(parent + ".") for parent in candidates if parent != name)
            ]
        units = [BypassUnit(path, path, path) for path in paths]
        validate_unit_ownership(student, teacher, units)
        return units

    def snapshot_inputs(self, args, kwargs):
        """Snapshot before teacher in-place updates, retaining native communication handles."""

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(
                value, (torch.distributed.ProcessGroup, torch.distributed.device_mesh.DeviceMesh)
            ):
                return value
            if is_dataclass(value) and not isinstance(value, type):
                return replace(
                    value,
                    **{
                        f.name: tree_map(snapshot, getattr(value, f.name))
                        for f in fields(value)
                        if f.init
                    },
                )
            return copy.deepcopy(value)

        return tree_map(snapshot, (args, kwargs))

    def snapshot_unit_state(self, unit, module, batch):
        """Allow a model family to capture side inputs before its teacher boundary."""
        return None

    def capture(self, unit, args, kwargs, output, batch):
        """Detach a current target; consumption finishes before teacher forward continues."""
        return CaptureRecord(
            args,
            kwargs,
            select_output(output, unit.output_selector).detach(),
            _valid_token_mask(batch),
            unit.output_selector,
        )

    def replay(self, module, record):
        """Replay once on already detached inputs, avoiding another activation copy."""
        return select_output(module(*record.args, **record.kwargs), record.output_selector)

    def batch_size(self, batch):
        """Read the sample axis from model inputs rather than arbitrary mask tensors."""
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        if isinstance(batch, dict):
            for name in ("input_ids", "hidden_states"):
                if name in batch:
                    return self.batch_size(batch[name])
            batch = batch.values()
        return next(self.batch_size(v) for v in batch if isinstance(v, torch.Tensor) and v.ndim)

    def validate_batch(self, batch, config):
        """Check declared context budgets for tokens or sequence-shaped hidden states."""
        values = batch.get("input_ids") if isinstance(batch, dict) else batch
        length = (
            values.shape[1]
            if isinstance(values, torch.Tensor) and (values.ndim >= 3 or isinstance(batch, dict))
            else None
        )
        if length is not None and length != config.sequence_length:
            raise ValueError("Training input length must match Bypass sequence_length")
        if self.batch_size(batch) < 1:
            raise ValueError("Bypass batches cannot be empty")

    def microbatches(self, batch, size):
        """Slice sample-aligned tensors while retaining shared model metadata."""
        samples = self.batch_size(batch)
        for start in range(0, samples, size):
            yield tree_map(
                lambda v: (
                    v[start : start + size]
                    if isinstance(v, torch.Tensor) and v.ndim and v.shape[0] == samples
                    else v
                ),
                batch,
            )

    def forward_teacher(self, teacher, args, kwargs):
        """Preserve the supplied callable's real forward and outer hooks."""
        parameters = inspect.signature(teacher.forward).parameters
        if "use_cache" in parameters:
            kwargs = {**kwargs, "use_cache": False}
        return teacher(*args, **kwargs)

    def teacher_microbatch(self, teacher, units, batch, consume):
        """Stream each boundary directly into a consumer with no activation bank."""
        handles, pending = [], {}

        def before(unit, module, args, kwargs):
            saved = self.snapshot_inputs(args, kwargs)
            state = self.snapshot_unit_state(unit, module, batch)
            pending.setdefault(unit.name, []).append((*saved, state))

        def after(unit, module, args, kwargs, output):
            args, kwargs, state = pending[unit.name].pop()
            record = replace(self.capture(unit, args, kwargs, output, batch), replay_state=state)
            consume(unit, record)

        try:
            for unit in units:
                module = teacher.get_submodule(unit.teacher_path)
                handles.append(
                    module.register_forward_pre_hook(
                        lambda m, a, k, u=unit: before(u, m, a, k), with_kwargs=True
                    )
                )
                handles.append(
                    module.register_forward_hook(
                        lambda m, a, k, o, u=unit: after(u, m, a, k, o), with_kwargs=True
                    )
                )
            if isinstance(batch, dict):
                args = ()
                kwargs = {
                    k: v
                    for k, v in batch.items()
                    if k not in ("valid_mask", "labels", "loss_mask", "sample_ids", "document_ids")
                }
            else:
                args, kwargs = batch if isinstance(batch, tuple) else (batch,), {}
            with torch.no_grad():
                self.forward_teacher(teacher, args, kwargs)
        finally:
            for handle in handles:
                handle.remove()
            pending.clear()
