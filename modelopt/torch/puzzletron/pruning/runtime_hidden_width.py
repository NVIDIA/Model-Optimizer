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

"""Differentiable full-envelope execution for a nested residual width."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from .embedding_pruning import EmbeddingPruningSpec

__all__ = [
    "hidden_width_layer_context",
    "hidden_width_module_context",
    "retained_hidden_prefix",
]


def _mask_last_dim(value: Any, width: int, hidden_size: int):
    if torch.is_tensor(value):
        if value.ndim == 0 or int(value.shape[-1]) != hidden_size:
            return value
        from .dynamic_block_prune import _apply_feature_mask

        mask = torch.arange(hidden_size, device=value.device) < int(width)
        return _apply_feature_mask(value, mask)
    if isinstance(value, tuple):
        return tuple(_mask_last_dim(item, width, hidden_size) for item in value)
    if isinstance(value, list):
        return [_mask_last_dim(item, width, hidden_size) for item in value]
    return value


def retained_hidden_prefix(value: torch.Tensor, width: int) -> torch.Tensor:
    if value.ndim == 0 or int(value.shape[-1]) < int(width):
        raise ValueError(f"cannot retain hidden width {width} from shape {tuple(value.shape)}")
    return value[..., : int(width)]


def _input_prefix_hook(width: int, hidden_size: int):
    def hook(module, args):
        if not args:
            return args
        return (_mask_last_dim(args[0], width, hidden_size), *args[1:])

    return hook


def _output_prefix_hook(width: int, hidden_size: int):
    def hook(module, args, output):
        return _mask_last_dim(output, width, hidden_size)

    return hook


def _functional_replica(module: torch.nn.Module, width: int, hidden_size: int) -> torch.nn.Module:
    """Shallow module replica whose call hooks and normalized shape are isolated."""
    replica = copy(module)
    replica._parameters = module._parameters.copy()
    replica._buffers = module._buffers.copy()
    replica._modules = module._modules.copy()
    replica._forward_pre_hooks = OrderedDict()
    replica._forward_hooks = OrderedDict()
    replica._backward_hooks = OrderedDict()
    normalized_shape = getattr(replica, "normalized_shape", None)
    if normalized_shape in {hidden_size, (hidden_size,)}:
        replica.normalized_shape = (width,)
    return replica


def _active_prefix_state(
    module: torch.nn.Module, width: int, hidden_size: int
) -> dict[str, torch.Tensor]:
    state = {}
    for name, value in (*module.named_parameters(), *module.named_buffers()):
        if value.ndim == 1 and int(value.shape[0]) == hidden_size:
            state[name] = value[:width]
    return state


def _explicit_rms_norm_mode(module: torch.nn.Module) -> str | None:
    """Return an explicit RMSNorm recipe when functional_call is unsafe.

    ``Float32RMSNorm`` needs float32 accumulation before the affine scale.
    ``Qwen3NextRMSNorm`` uses one-centered ``(1 + weight)`` and must slice the
    affine term before the multiply; ``torch.func.functional_call`` can leave the
    full-envelope weight bound and raise a 768-vs-1024 shape error under nested
    width cycling.
    """

    weight = getattr(module, "weight", None)
    if weight is None or not hasattr(module, "eps"):
        return None
    name = type(module).__name__
    if name == "Float32RMSNorm":
        return "scale"
    if name == "Qwen3NextRMSNorm":
        return "offset"
    return None


def _active_norm_forward(module: torch.nn.Module, width: int, hidden_size: int):
    """Run the module's native norm semantics at the physical active width."""
    rms_mode = _explicit_rms_norm_mode(module)
    replica = None if rms_mode is not None else _functional_replica(module, width, hidden_size)

    def forward(*args, **kwargs):
        if not args or not torch.is_tensor(args[0]):
            raise RuntimeError(
                f"nested hidden width expected tensor input for {type(module).__name__}"
            )
        x = args[0]
        if int(x.shape[-1]) != hidden_size:
            raise RuntimeError(
                f"nested normalization expected envelope width {hidden_size}, got {tuple(x.shape)}"
            )
        active_x = x[..., :width]
        if rms_mode is not None:
            input_dtype = active_x.dtype
            normalized = active_x.float()
            normalized = normalized * torch.rsqrt(
                normalized.pow(2).mean(-1, keepdim=True) + float(module.eps)
            )
            weight = module.weight[:width].float()
            if rms_mode == "offset":
                output = (normalized * (1.0 + weight)).to(input_dtype)
            else:
                output = (weight * normalized).to(input_dtype)
        else:
            assert replica is not None
            replica._parameters = module._parameters.copy()
            replica._buffers = module._buffers.copy()
            output = torch.func.functional_call(
                replica,
                _active_prefix_state(module, width, hidden_size),
                (active_x, *args[1:]),
                kwargs,
                strict=False,
            )
        if not torch.is_tensor(output) or int(output.shape[-1]) != width:
            raise RuntimeError(
                f"active normalization {type(module).__name__} returned incompatible output"
            )
        return torch.nn.functional.pad(output, (0, hidden_size - width))

    return forward


@contextmanager
def hidden_width_module_context(
    module: torch.nn.Module,
    *,
    canonical_module_name: str,
    spec: EmbeddingPruningSpec,
    width: int,
    mask_boundary_input: bool = False,
):
    """Apply descriptor-owned width rules to one module tree in a static envelope."""

    width = spec.validate_width(int(width))
    hidden_size = int(spec.hidden_size)
    if width == hidden_size:
        yield
        return

    handles = []
    restored_forwards: list[tuple[torch.nn.Module, Any]] = []
    if mask_boundary_input:
        handles.append(module.register_forward_pre_hook(_input_prefix_hook(width, hidden_size)))
    try:
        for module_name, child in module.named_modules():
            weight = getattr(child, "weight", None)
            if weight is None:
                continue
            suffix = f"{module_name}.weight" if module_name else "weight"
            key = f"{canonical_module_name}.{suffix}"
            rule = spec.rule_for(key)
            if rule is None:
                continue
            axes = tuple(axis if axis >= 0 else weight.ndim + axis for axis in rule.axes)
            if 1 in axes:
                handles.append(
                    child.register_forward_pre_hook(_input_prefix_hook(width, hidden_size))
                )
            if 0 in axes:
                if weight.ndim == 1:
                    restored_forwards.append((child, child.forward))
                    child.forward = _active_norm_forward(child, width, hidden_size)
                else:
                    handles.append(
                        child.register_forward_hook(_output_prefix_hook(width, hidden_size))
                    )
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
        for child, original_forward in reversed(restored_forwards):
            child.forward = original_forward


@contextmanager
def hidden_width_layer_context(
    layer: torch.nn.Module,
    *,
    canonical_layer_name: str,
    spec: EmbeddingPruningSpec,
    width: int,
):
    """Execute one decoder block as a prefix-width child in a static PP envelope.

    Residual inputs and outputs keep the full PP shape, while descriptor-owned
    input columns and output rows outside the active prefix are differentiably
    zeroed. RMSNorm is corrected to use the active-width denominator exactly.
    This preserves TP/FSDP module forwards and their collectives.
    """
    with hidden_width_module_context(
        layer,
        canonical_module_name=canonical_layer_name,
        spec=spec,
        width=width,
        mask_boundary_input=True,
    ):
        yield
