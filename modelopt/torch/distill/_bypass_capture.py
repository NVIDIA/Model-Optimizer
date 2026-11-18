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
from itertools import chain
from typing import Any

import torch
from torch.utils._pytree import tree_map

from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer
from modelopt.torch.quantization.repair import (
    _iq_quantizer,
    _router_paths,
    _storage_spans,
    _validate_storage_ranges,
)

_MSE_CHUNK_ELEMENTS = {"cpu": 1 << 18, "cuda": 1 << 24}


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


def _ownership_tensors(module, backings):
    seen = set()
    for owner in module.modules():
        for name, tensor in chain(owner._parameters.items(), owner._buffers.items()):
            for value in (tensor, backings.get((owner, name))):
                if value is not None and id(value) not in seen:
                    seen.add(id(value))
                    yield value


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
    backings, slots = {}, {}
    for root in (student, teacher):
        hooks = [getattr(owner, "_hf_hook", None) for owner in root.modules()]
        while hooks:
            hook = hooks.pop()
            hooks.extend(getattr(hook, "hooks", ()))
            if getattr(hook, "weights_map", None) is None or not (
                getattr(hook, "offload", False) or hasattr(hook, "original_devices")
            ):
                continue
            # Optional Accelerate backings remain owned while native hooks are suspended.
            from modelopt.torch.quantization.plugins.accelerate import _offloaded_tensor_backings

            retained, publications = _offloaded_tensor_backings(root)
            backings.update(retained)
            slots.update(publications)
            break
    _validate_student_owners(student, paths, backings, slots)
    teacher_storage = {}
    for tensor in _ownership_tensors(teacher, backings):
        for key, start, stop in _storage_spans(tensor):
            teacher_storage.setdefault(key, []).append((start, stop))
    teacher_slots = {
        slot
        for owner in teacher.modules()
        for name in (*owner._parameters, *owner._buffers)
        for slot in slots.get((owner, name), ())
    }
    owned, teacher_owners = set(), set()
    for unit in units:
        try:
            module = student.get_submodule(unit.student_path)
            teacher_module = teacher.get_submodule(unit.teacher_path)
        except AttributeError as exc:
            raise ValueError(f"Boundary {unit.name!r} must exist in both models") from exc
        for tensor in _ownership_tensors(teacher_module, backings):
            if id(tensor) in teacher_owners:
                raise ValueError("Teacher Bypass boundaries share ownership")
            teacher_owners.add(id(tensor))
        if any(
            teacher_slots.intersection(slots.get((owner, name), ()))
            for owner in module.modules()
            for name in (*owner._parameters, *owner._buffers)
        ):
            raise ValueError("Student Bypass state shares a teacher publication slot")
        for tensor in _ownership_tensors(module, backings):
            if id(tensor) in owned:
                raise ValueError("Student Bypass boundaries share ownership")
            owned.add(id(tensor))
            for key, start, stop in _storage_spans(tensor):
                if any(start < end and begin < stop for begin, end in teacher_storage.get(key, ())):
                    raise ValueError("Student Bypass state aliases teacher storage")


def _validate_student_owners(student, paths, backings, slots):
    modules, tensors, storage, publications = {}, {}, {}, {}
    for path, module in student.named_modules(remove_duplicate=False):
        selected = any(not root or path == root or path.startswith(root + ".") for root in paths)
        if isinstance(module, TensorQuantizer):
            previous = modules.get(id(module))
            if previous is not None and (selected or previous):
                raise ValueError("Selected Bypass state shares a quantizer owner")
            modules[id(module)] = selected
        for name, tensor in chain(module._parameters.items(), module._buffers.items()):
            if tensor is None:
                continue
            for slot in slots.get((module, name), ()):
                previous = publications.get(slot)
                if previous is not None and (selected or previous):
                    raise ValueError("Selected Bypass state shares an offload publication slot")
                publications[slot] = selected
            local = {}
            representations = {
                id(value): value
                for value in (tensor, backings.get((module, name)))
                if value is not None
            }
            for identity, value in representations.items():
                previous = tensors.get(identity)
                if previous is not None and (selected or previous):
                    raise ValueError("Selected Bypass state shares a tensor owner")
                tensors[identity] = selected
                for key, start, stop in _storage_spans(value):
                    local.setdefault(key, []).append((start, stop))
            for key, spans in local.items():
                merged = []
                for start, stop in sorted(spans):
                    if merged and start <= merged[-1][1]:
                        merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
                    else:
                        merged.append((start, stop))
                storage.setdefault(key, []).extend(
                    (start, stop, selected) for start, stop in merged
                )
    _validate_storage_ranges(storage)


def _valid_token_mask(batch):
    """Use explicit loss validity or standard attention masks, not model-owned metadata."""
    if not isinstance(batch, dict):
        return None
    for key in ("valid_mask", "loss_mask"):
        mask = batch.get(key)
        if mask is not None:
            if not isinstance(mask, torch.Tensor) or mask.is_floating_point() or mask.is_complex():
                raise ValueError(
                    f"{key} must be a boolean or integer Tensor; Bypass does not weight tokens"
                )
            return mask.bool()
    attention = batch.get("attention_mask")
    if attention is not None:
        if (
            isinstance(attention, torch.Tensor)
            and attention.ndim == 2
            and not attention.is_floating_point()
            and not attention.is_complex()
        ):
            return attention.bool()
        raise ValueError("Provide valid_mask or loss_mask with floating or 4D attention masks")
    return None


class _MaskedMSE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, prediction, target, mask, features):
        ctx.save_for_backward(prediction, target, mask)
        ctx.features = features
        left, right = prediction.reshape(-1, features), target.reshape(-1, features)
        valid = None if mask is None else mask.reshape(-1, 1)
        total = prediction.new_zeros((), dtype=torch.float32)
        chunk_elements = _MSE_CHUNK_ELEMENTS.get(prediction.device.type, _MSE_CHUNK_ELEMENTS["cpu"])
        rows = max(1, chunk_elements // features)
        for start in range(0, len(left), rows):
            part = slice(start, start + rows)
            difference = left[part].float() - right[part].float()
            if valid is not None:
                difference.masked_fill_(~valid[part], 0)
            total.add_(difference.square_().sum())
        return total

    @staticmethod
    def backward(ctx, gradient):
        prediction, target, mask = ctx.saved_tensors
        features = ctx.features
        left, right = prediction.reshape(-1, features), target.reshape(-1, features)
        valid = None if mask is None else mask.reshape(-1, 1)
        dp = torch.empty_like(left) if ctx.needs_input_grad[0] else None
        dt = torch.empty_like(right) if ctx.needs_input_grad[1] else None
        chunk_elements = _MSE_CHUNK_ELEMENTS.get(prediction.device.type, _MSE_CHUNK_ELEMENTS["cpu"])
        rows = max(1, chunk_elements // features)
        for start in range(0, len(left), rows):
            part = slice(start, start + rows)
            difference = (left[part].float() - right[part].float()).mul_(2 * gradient)
            if valid is not None:
                difference.masked_fill_(~valid[part], 0)
            if dp is not None:
                dp[part] = difference
            if dt is not None:
                dt[part] = -difference
        return (
            None if dp is None else dp.reshape(prediction.shape),
            None if dt is None else dt.reshape(target.shape),
            None,
            None,
        )


def mse_statistics(prediction, target, valid_mask=None):
    """FP32 squared error sum and the exact count of valid feature elements."""
    if prediction.shape != target.shape:
        raise ValueError("Student and teacher boundary output shapes differ")
    if valid_mask is not None and (
        valid_mask.ndim >= target.ndim
        or tuple(valid_mask.shape) != tuple(target.shape[: valid_mask.ndim])
    ):
        raise ValueError("Valid-token masks must match the boundary's token dimensions")
    features = 1 if valid_mask is None else math.prod(target.shape[valid_mask.ndim :])
    mask = None if valid_mask is None else valid_mask.to(device=target.device, dtype=torch.bool)
    count = (
        target.new_tensor(target.numel(), dtype=torch.int64)
        if mask is None
        else mask.sum(dtype=torch.int64) * features
    )
    return _MaskedMSE.apply(prediction, target, mask, max(features, 1)), count


class ModuleBypassAdapter:
    """Resolve ordinary callable Torch boundaries without a Transformers dependency."""

    feedforward_names = ("mlp", "moe", "ffn", "block_sparse_moe")
    context: Any = None

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
            for name in ("input_ids", "inputs_embeds", "hidden_states"):
                if name in batch:
                    return self.batch_size(batch[name])
            batch = batch.values()
        return next(self.batch_size(v) for v in batch if isinstance(v, torch.Tensor) and v.ndim)

    def validate_batch(self, batch, config):
        """Check declared context budgets for tokens or sequence-shaped hidden states."""
        _valid_token_mask(batch)
        values = (
            batch.get("input_ids", batch.get("inputs_embeds", batch.get("hidden_states")))
            if isinstance(batch, dict)
            else batch
        )
        if isinstance(values, tuple):
            values = next(v for v in values if isinstance(v, torch.Tensor) and v.ndim)
        length = (
            values.shape[1]
            if isinstance(values, torch.Tensor)
            and values.ndim >= 2
            and (
                values.ndim >= 3
                or isinstance(batch, dict)
                or not (values.is_floating_point() or values.is_complex())
            )
            else None
        )
        if length is not None and length != config.sequence_length:
            raise ValueError("Training input length must match Bypass sequence_length")
        if self.batch_size(batch) < 1:
            raise ValueError("Bypass batches cannot be empty")

    def microbatches(self, batch, size):
        """Slice batch-leading inputs; other 1D dict metadata stays shared.

        Adapters with other auxiliary sample layouts must override this method.
        """
        samples = self.batch_size(batch)
        mask = _valid_token_mask(batch)
        if mask is not None and (not mask.ndim or mask.shape[0] != samples):
            raise ValueError("Bypass loss mask must match the original batch's sample dimension")
        for start in range(0, samples, size):

            def sliced(value, sample_vector=True):
                return tree_map(
                    lambda v: (
                        v[start : start + size]
                        if isinstance(v, torch.Tensor)
                        and v.ndim
                        and v.shape[0] == samples
                        and (v.ndim > 1 or sample_vector)
                        else v
                    ),
                    value,
                )

            if isinstance(batch, dict):
                yield {
                    key: sliced(
                        value,
                        key
                        in (
                            "input_ids",
                            "hidden_states",
                            "inputs_embeds",
                            "valid_mask",
                            "loss_mask",
                            "attention_mask",
                            "labels",
                            "sample_ids",
                            "document_ids",
                        ),
                    )
                    for key, value in batch.items()
                }
            else:
                yield sliced(batch)

    def forward_teacher(self, teacher, args, kwargs):
        """Preserve the supplied callable's real forward and outer hooks."""
        signature = inspect.signature(teacher.forward)
        parameters = signature.parameters
        if "use_cache" in parameters:
            positional = [
                name
                for name, parameter in parameters.items()
                if parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
            ]
            if "use_cache" in positional and (index := positional.index("use_cache")) < len(args):
                args = (*args[:index], False, *args[index + 1 :])
            elif parameters["use_cache"].kind == inspect.Parameter.POSITIONAL_ONLY:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                bound.arguments["use_cache"] = False
                args, kwargs = bound.args, bound.kwargs
            else:
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
