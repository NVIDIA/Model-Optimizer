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

"""Semantic repair boundaries shared by local-distillation runtimes."""

import copy
import inspect
import math
from contextlib import nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.utils._pytree import tree_map

__all__ = ["CaptureRecord", "ModuleLocalDistillationAdapter", "RepairUnit", "mse_statistics"]


@dataclass(frozen=True)
class RepairUnit:
    """One student/teacher boundary; paths are relative to their respective models."""

    name: str
    student_path: str
    teacher_path: str
    output_selector: tuple[str | int, ...] = ()


@dataclass(frozen=True)
class CaptureRecord:
    """A detached teacher boundary for one current microbatch."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    target: torch.Tensor
    valid_mask: torch.Tensor | None = None
    output_selector: tuple[str | int, ...] = ()
    replay_state: Any = None


def select_output(output, selector=()):
    for key in selector:
        output = output[key]
    if not selector and isinstance(output, (tuple, list)):
        output = output[0]
    if not isinstance(output, torch.Tensor):
        raise ValueError("A repair boundary must select a Tensor output; provide output_selector.")
    return output


def teacher_forward_kwargs(teacher, kwargs):
    """Avoid cache and unused full-token logits in repair-only teacher passes."""
    kwargs = dict(kwargs)
    parameters = inspect.signature(teacher.forward).parameters
    for name, value in (("use_cache", False), ("logits_to_keep", 1)):
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind != inspect.Parameter.POSITIONAL_ONLY:
            kwargs[name] = value
    return kwargs


def _valid_token_mask(batch):
    if not isinstance(batch, dict):
        return None
    for name in ("valid_mask", "_packed_seq_ids"):
        value = batch.get(name)
        if isinstance(value, torch.Tensor):
            return value.bool()
    padding = batch.get("padding_mask")
    if isinstance(padding, torch.Tensor) and padding.ndim == 2:
        return ~padding.bool()
    attention = batch.get("attention_mask")
    if isinstance(attention, torch.Tensor):
        if attention.ndim == 2:
            return attention.bool()
        if attention.ndim == 4:
            if attention.dtype == torch.bool:
                # Native block-causal masks exclude padded query rows.
                return attention.any(dim=-1).any(dim=1)
            raise ValueError(
                "Provide valid_mask for additive attention masks; padding cannot "
                "be inferred reliably from arbitrary attention biases."
            )
    return None


def mse_statistics(prediction, target, valid_mask=None):
    """Return FP32 squared-error/target-energy sums and an exact element count."""
    if prediction.shape != target.shape:
        raise ValueError("Student and teacher boundary output shapes differ.")
    difference = prediction.float() - target.float()
    if valid_mask is None:
        return (
            difference.square().sum(),
            target.new_tensor(target.numel(), dtype=torch.int64),
            target.float().square().sum(),
        )
    if valid_mask.ndim >= target.ndim or tuple(valid_mask.shape) != tuple(
        target.shape[: valid_mask.ndim]
    ):
        raise ValueError("The valid-token mask must match the boundary's token dimensions.")
    # Decoder boundaries may retain multiple residual streams per token. Treat
    # those as feature dimensions, preserving one validity decision per token.
    feature_elements = math.prod(target.shape[valid_mask.ndim :])
    mask = valid_mask.to(device=target.device, dtype=torch.bool).reshape(
        *valid_mask.shape, *((1,) * (target.ndim - valid_mask.ndim))
    )
    error = torch.where(mask, difference, 0).square().sum()
    energy = torch.where(mask, target.float(), 0).square().sum()
    return error, mask.sum(dtype=torch.int64) * feature_elements, energy


def validate_unit_ownership(student: nn.Module, teacher: nn.Module, units: list[RepairUnit]):
    """Refuse ambiguous selections before changing parameters or installing hooks."""
    if not units:
        raise ValueError(
            "No repair units selected; provide explicit module paths or a model adapter."
        )
    paths = [unit.student_path for unit in units]
    for index, path in enumerate(paths):
        if any(
            path == other
            or not path
            or not other
            or path.startswith(other + ".")
            or other.startswith(path + ".")
            for other in paths[:index]
        ):
            raise ValueError("Repair units overlap; select disjoint semantic modules.")
    owners: dict[int, str] = {}
    pairs = []
    for unit in units:
        try:
            module = student.get_submodule(unit.student_path)
            teacher_module = teacher.get_submodule(unit.teacher_path)
        except AttributeError as exc:
            raise ValueError(
                f"Repair boundary {unit.name!r} is not present in both models."
            ) from exc
        pairs.append((unit, module, teacher_module))
    teacher_parameters = {
        id(parameter) for _, _, teacher_module in pairs for parameter in teacher_module.parameters()
    }
    for unit, module, _ in pairs:
        for parameter in module.parameters():
            if id(parameter) in teacher_parameters:
                raise ValueError("Student repair parameters alias the teacher.")
            previous = owners.setdefault(id(parameter), unit.name)
            if previous != unit.name:
                raise ValueError(
                    f"Repair units {previous!r} and {unit.name!r} share parameters; "
                    "select a common owning module."
                )
        for buffer in module.buffers():
            previous = owners.setdefault(id(buffer), unit.name)
            if previous != unit.name:
                raise ValueError("Repair units share mutable buffers; select their common owner.")


def _has_quantized_weights(module):
    from modelopt.torch.quantization.nn import QuantModule, TensorQuantizer

    return any(
        leaf.is_enabled
        for child in module.modules()
        if isinstance(child, QuantModule)
        for quantizer in child.iter_weight_quantizers_for_calibration()
        for leaf in quantizer.modules()
        if isinstance(leaf, TensorQuantizer)
    )


class ModuleLocalDistillationAdapter:
    """Resolve ordinary module boundaries; architecture-specific adapters extend this."""

    def observe_replay(self, module, training):
        """Optionally expose native routed-expert counts for the current replay."""
        return nullcontext({})

    def _default_unit_paths(self, student, selection):
        candidates = [
            name
            for name, module in student.named_modules()
            if name.rsplit(".", 1)[-1] in selection.unit_types and _has_quantized_weights(module)
        ]
        return [
            name
            for name in candidates
            if not any(name.startswith(parent + ".") for parent in candidates if parent != name)
        ]

    def resolve_units(self, student, teacher, selection):
        """Resolve disjoint selected units, skipping unquantized defaults."""
        paths = selection.units
        if paths is None:
            paths = self._default_unit_paths(student, selection)
        units = [RepairUnit(path, path, path) for path in paths]
        validate_unit_ownership(student, teacher, units)
        return units

    def snapshot_unit_state(self, unit, module, batch):
        """Capture optional family-owned side inputs before the teacher consumes them."""
        return None

    def snapshot_inputs(self, args, kwargs):
        """Detach mutable boundary inputs before the teacher can update them."""

        def snapshot(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, (dist.ProcessGroup, DeviceMesh)):
                return value
            # Native CP metadata owns communication handles and mutable tensors.
            # Deepcopy cannot pickle the former; sharing the latter leaks state
            # from subsequent teacher blocks into independent student replay.
            if is_dataclass(value) and not isinstance(value, type):
                return replace(
                    value,
                    **{
                        field.name: tree_map(snapshot, getattr(value, field.name))
                        for field in fields(value)
                        if field.init
                    },
                )
            return copy.deepcopy(value)

        return tree_map(snapshot, (args, kwargs))

    def capture(self, unit, args, kwargs, output, batch):
        """Keep the selected teacher target and its all-token validity mask."""
        target = select_output(output, unit.output_selector).detach().clone()
        mask = _valid_token_mask(batch)
        if mask is not None:
            mask = mask.detach().clone()
        return CaptureRecord(args, kwargs, target, mask, unit.output_selector)

    def replay(self, module, record):
        """Execute the actual quantized unit with independent captured state."""
        args, kwargs = self.snapshot_inputs(record.args, record.kwargs)
        return select_output(module(*args, **kwargs), record.output_selector)

    def batch_size(self, batch):
        """Read the sample axis of a tensor or nested batch."""
        if isinstance(batch, torch.Tensor):
            return batch.shape[0]
        values = batch.values() if isinstance(batch, dict) else batch
        return next(
            self.batch_size(value)
            for value in values
            if isinstance(value, torch.Tensor) and value.ndim
        )

    def microbatches(self, batch, size):
        """Slice sample-aligned tensors without changing the update budget."""
        batch_size = self.batch_size(batch)
        for start in range(0, batch_size, size):
            yield tree_map(
                lambda value: (
                    value[start : start + size]
                    if isinstance(value, torch.Tensor)
                    and value.ndim
                    and value.shape[0] == batch_size
                    else value
                ),
                batch,
            )

    def teacher_microbatch(self, teacher, units, batch, consume):
        """Stream each boundary directly to its independent repair consumer."""
        # Local import avoids the layerwise wrapper's module initialization cycle.
        from .layerwise_distillation_model import local_replay_boundaries

        if isinstance(batch, dict):
            args = ()
            kwargs = {
                key: value
                for key, value in batch.items()
                if key
                not in (
                    "valid_mask",
                    "labels",
                    "loss_mask",
                    "sample_ids",
                    "document_ids",
                )
            }
        else:
            args, kwargs = (batch if isinstance(batch, tuple) else (batch,)), {}
        with local_replay_boundaries(teacher, units, self, batch, consume), torch.no_grad():
            teacher(*args, **teacher_forward_kwargs(teacher, kwargs))
