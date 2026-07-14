# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Static-envelope runtime slicing for descriptor-owned PLE channels."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch

from .ple_pruning import PLEPruningSpec

__all__ = ["ple_layer_context"]


def _mask_prefix(value: Any, width: int, full_width: int):
    if not torch.is_tensor(value):
        return value
    if value.ndim == 0 or int(value.shape[-1]) != full_width:
        return value
    mask = torch.arange(full_width, device=value.device) < int(width)
    return value * mask.to(dtype=value.dtype).reshape(
        (1,) * (value.ndim - 1) + (-1,)
    )


def _layer_input_hook(width: int, full_width: int):
    def hook(module, args, kwargs):
        del module
        args = list(args)
        kwargs = dict(kwargs)
        if "per_layer_input" in kwargs:
            kwargs["per_layer_input"] = _mask_prefix(
                kwargs["per_layer_input"], width, full_width
            )
        elif len(args) > 1:
            args[1] = _mask_prefix(args[1], width, full_width)
        return tuple(args), kwargs

    return hook


def _module_input_hook(width: int, full_width: int):
    def hook(module, args):
        del module
        if not args:
            return args
        return (_mask_prefix(args[0], width, full_width), *args[1:])

    return hook


def _module_output_hook(width: int, full_width: int):
    def hook(module, args, output):
        del module, args
        return _mask_prefix(output, width, full_width)

    return hook


@contextmanager
def ple_layer_context(
    layer: torch.nn.Module,
    *,
    spec: PLEPruningSpec,
    width: int,
):
    """Replay one layer as a prefix-sliced PLE child at static tensor shapes.

    The full teacher supplies ``per_layer_input``. The student keeps its static
    PP/FSDP envelope, while the descriptor-declared gate rows and projection
    columns outside the active prefix are masked. This is exactly equivalent to
    the physically sliced layer for the same retained PLE input channels and
    leaves inactive supernet parameters with zero gradients.
    """

    width = int(width)
    full_width = int(spec.width)
    if not 0 < width <= full_width:
        raise ValueError(
            f"PLE runtime width must be in [1, {full_width}], got {width}"
        )
    if width == full_width:
        yield
        return

    gate = getattr(layer, spec.layer_gate_name, None)
    projection = getattr(layer, spec.layer_projection_name, None)
    if gate is None or projection is None:
        raise AttributeError(
            f"{type(layer).__name__} has no descriptor PLE modules "
            f"{spec.layer_gate_name!r}/{spec.layer_projection_name!r}"
        )

    handles = [
        layer.register_forward_pre_hook(
            _layer_input_hook(width, full_width),
            with_kwargs=True,
        ),
        gate.register_forward_hook(_module_output_hook(width, full_width)),
        projection.register_forward_pre_hook(_module_input_hook(width, full_width)),
    ]
    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
