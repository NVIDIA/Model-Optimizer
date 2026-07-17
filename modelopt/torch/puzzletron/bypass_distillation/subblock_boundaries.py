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

"""Descriptor-backed capture and isolated replay at decoder subblock boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Sequence

import torch
from torch import nn

if TYPE_CHECKING:
    from ..block_config import BlockConfig

__all__ = [
    "SubblockBoundary",
    "SubblockKey",
    "SubblockReplayRecord",
    "install_teacher_subblock_capture_hooks",
    "replay_subblock",
    "resolve_subblock_boundaries",
    "selected_subblock_kinds",
]

SubblockKey = tuple[int, str, str]


def selected_subblock_kinds(keys_to_learn: Any) -> frozenset[str] | None:
    """Return exact semantic boundary kinds, or ``None`` for an entire block."""

    from .bypass_utils import normalize_keys_to_learn

    keys = set(normalize_keys_to_learn(keys_to_learn)["subblocks"])
    if keys == {"entire_block"}:
        return None
    kinds: set[str] = set()
    if "subblock_attention" in keys:
        kinds.add("attention")
    if "subblock_mamba" in keys:
        kinds.add("mamba")
    if "subblock_ffn" in keys:
        kinds.update(("ffn", "moe"))
    return frozenset(kinds)


def _detach_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, tuple):
        return tuple(_detach_tree(item) for item in value)
    if isinstance(value, list):
        return [_detach_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _detach_tree(item) for key, item in value.items()}
    return value


def _output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            try:
                return _output_tensor(item)
            except TypeError:
                continue
    if isinstance(output, Mapping):
        for key in ("hidden_states", "last_hidden_state", "output"):
            if key in output:
                return _output_tensor(output[key])
        for item in output.values():
            try:
                return _output_tensor(item)
            except TypeError:
                continue
    raise TypeError(f"subblock output contains no tensor: {type(output).__name__}")


@dataclass(frozen=True)
class SubblockBoundary:
    layer_idx: int
    kind: str
    name: str
    module_path: str
    owner: nn.Module
    module: nn.Module

    @property
    def key(self) -> SubblockKey:
        return (self.layer_idx, self.kind, self.name)


@dataclass(frozen=True)
class SubblockReplayRecord:
    args: tuple[Any, ...]
    kwargs: Mapping[str, Any]
    target: torch.Tensor


def resolve_subblock_boundaries(
    layers: Mapping[int, nn.Module],
    descriptor: Any,
    block_configs: Sequence[BlockConfig],
) -> dict[SubblockKey, SubblockBoundary]:
    """Resolve local PP-owned subblocks without model-family branches in the engine."""

    resolver = getattr(descriptor, "local_kd_subblock_module_paths", None)
    if resolver is None:
        raise NotImplementedError(
            f"descriptor {getattr(descriptor, '__name__', descriptor)!r} does not support "
            "descriptor-backed subblock bypass"
        )
    boundaries: dict[SubblockKey, SubblockBoundary] = {}
    for layer_idx, layer in layers.items():
        block = block_configs[layer_idx]
        declared_paths = resolver(block, layer_idx=layer_idx)
        configured = {
            (subblock.kind, subblock.name) for subblock in block.subblock_configs
        }
        unexpected = set(declared_paths) - configured
        if unexpected:
            raise RuntimeError(
                f"descriptor subblock boundary mismatch at layer {layer_idx}: "
                f"unexpected={sorted(unexpected)}"
            )
        expected = {
            (subblock.kind, subblock.name)
            for subblock in block.subblock_configs
            if not subblock.no_op
        }
        paths = {key: path for key, path in declared_paths.items() if key in expected}
        if set(paths) != expected:
            raise RuntimeError(
                f"descriptor subblock boundary mismatch at layer {layer_idx}: "
                f"expected={sorted(expected)}, present={sorted(paths)}"
            )
        for (kind, name), module_path in paths.items():
            module = layer.get_submodule(str(module_path))
            boundary = SubblockBoundary(
                layer_idx=int(layer_idx),
                kind=str(kind),
                name=str(name),
                module_path=str(module_path),
                owner=layer,
                module=module,
            )
            if boundary.key in boundaries:
                raise RuntimeError(f"duplicate subblock boundary {boundary.key!r}")
            boundaries[boundary.key] = boundary
    return boundaries


def install_teacher_subblock_capture_hooks(
    boundaries: Mapping[SubblockKey, SubblockBoundary],
    records: MutableMapping[SubblockKey, list[SubblockReplayRecord]],
    *,
    capture_enabled: Callable[[], bool],
) -> list[Any]:
    """Capture detached teacher arguments and pre-residual subblock outputs."""

    handles: list[Any] = []
    pending: dict[SubblockKey, list[tuple[tuple[Any, ...], Mapping[str, Any]]]] = {}
    for key, boundary in boundaries.items():
        pending[key] = []

        def _pre(module, args, kwargs, *, boundary_key=key):
            if capture_enabled():
                pending[boundary_key].append(
                    (_detach_tree(tuple(args)), _detach_tree(dict(kwargs)))
                )

        def _post(module, args, output, *, boundary_key=key):
            if not capture_enabled():
                return
            if not pending[boundary_key]:
                raise RuntimeError(
                    f"subblock target {boundary_key!r} fired without a captured input"
                )
            replay_args, replay_kwargs = pending[boundary_key].pop(0)
            records[boundary_key].append(
                SubblockReplayRecord(
                    args=replay_args,
                    kwargs=replay_kwargs,
                    target=_detach_tree(_output_tensor(output)),
                )
            )

        handles.append(boundary.module.register_forward_pre_hook(_pre, with_kwargs=True))
        handles.append(boundary.module.register_forward_hook(_post))
    return handles


def replay_subblock(
    boundary: SubblockBoundary,
    record: SubblockReplayRecord,
) -> torch.Tensor:
    """Replay one student subblock from its exact detached teacher boundary input."""

    owner = boundary.owner
    if not callable(getattr(owner, "unshard", None)):
        return _output_tensor(boundary.module(*record.args, **record.kwargs))

    # Calling an FSDP2-managed child directly skips the owning layer's hooks,
    # leaving its parameters as DTensors while the captured teacher input is a
    # regular Tensor. Route the isolated child call through the owner's normal
    # __call__ hook path so FSDP unshards before forward and registers its usual
    # reshard/reduce-scatter hooks for backward. The original layer body is not
    # executed, so the replay remains isolated to this semantic subblock.
    had_instance_forward = "forward" in owner.__dict__
    instance_forward = owner.__dict__.get("forward")

    def _isolated_forward(*args, **kwargs):
        return boundary.module(*args, **kwargs)

    object.__setattr__(owner, "forward", _isolated_forward)
    try:
        return _output_tensor(owner(*record.args, **record.kwargs))
    finally:
        if had_instance_forward:
            object.__setattr__(owner, "forward", instance_forward)
        else:
            object.__delattr__(owner, "forward")
