# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in synchronized module tracing for AutoModel solution scoring."""

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

import torch
import torch.distributed as torch_dist
from torch import nn

_ENABLED_ENV = "PUZZLETRON_TRACE_MODULE_SYNCS"
_FILTER_ENV = "PUZZLETRON_TRACE_MODULE_FILTER"
_LAYER_ENV = "PUZZLETRON_TRACE_MODULE_LAYER"
_PREFIX = "[solution/automodel/module-trace]"


def _requested_layer() -> int | None:
    if os.environ.get(_ENABLED_ENV) != "1":
        return None
    raw = os.environ.get(_LAYER_ENV)
    if raw is None:
        raise ValueError(f"{_LAYER_ENV} is required when {_ENABLED_ENV}=1")
    try:
        layer_idx = int(raw)
    except ValueError as error:
        raise ValueError(f"{_LAYER_ENV} must be a non-negative integer, got {raw!r}") from error
    if layer_idx < 0:
        raise ValueError(f"{_LAYER_ENV} must be a non-negative integer, got {raw!r}")
    return layer_idx


def _requested_modules() -> set[str] | None:
    raw = os.environ.get(_FILTER_ENV)
    if raw is None:
        return None
    labels = {item.strip() for item in raw.split(",") if item.strip()}
    if not labels:
        raise ValueError(f"{_FILTER_ENV} must name at least one module")
    return labels


def _rank() -> int:
    if torch_dist.is_available() and torch_dist.is_initialized():
        return torch_dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _emit(layer_idx: int, label: str, phase: str) -> None:
    print(
        f"{_PREFIX} rank={_rank()} layer={layer_idx} module={label} phase={phase}",
        flush=True,
    )


def _cuda_device(value):
    if torch.is_tensor(value):
        return value.device if value.is_cuda else None
    if isinstance(value, Mapping):
        values = value.values()
    elif isinstance(value, (tuple, list)):
        values = value
    else:
        values = ()
    return next(
        (device for item in values if (device := _cuda_device(item)) is not None),
        None,
    )


def _synchronize(value) -> None:
    device = _cuda_device(value)
    if device is None and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    if device is not None:
        torch.cuda.synchronize(device)


def _optional_child(module, *names):
    if module is None:
        return None
    return next(
        (child for name in names if isinstance((child := getattr(module, name, None)), nn.Module)),
        None,
    )


def _trace_targets(layer):
    mlp = _optional_child(layer, "mlp")
    shared = _optional_child(mlp, "shared_experts", "shared_expert")
    candidates = [
        ("decoder", layer),
        ("self_attn", _optional_child(layer, "self_attn")),
        ("linear_attn", _optional_child(layer, "linear_attn")),
        ("mlp", mlp),
        ("shared_experts", shared),
        ("shared_experts.gate_proj", _optional_child(shared, "gate_proj")),
        ("shared_experts.up_proj", _optional_child(shared, "up_proj")),
        ("shared_experts.down_proj", _optional_child(shared, "down_proj")),
        ("routed_experts", _optional_child(mlp, "experts")),
    ]
    seen = set()
    for label, module in candidates:
        if module is None:
            yield label, None
        elif id(module) not in seen:
            seen.add(id(module))
            yield label, module


def _layer_exists_on_any_rank(local_layer) -> bool:
    if local_layer is not None:
        return True
    return (
        torch_dist.is_available()
        and torch_dist.is_initialized()
        and torch_dist.get_world_size() > 1
    )


@contextmanager
def synchronized_module_trace(recipe) -> Iterator[None]:
    """Install opt-in synchronized hooks on one locally owned decoder layer."""

    layer_idx = _requested_layer()
    if layer_idx is None:
        yield
        return
    layer = recipe._find_decoder_layer(layer_idx)
    if not _layer_exists_on_any_rank(layer):
        raise ValueError(f"requested trace layer {layer_idx} is not present on this model")
    if layer is None:
        _emit(layer_idx, "decoder", "not_owned")
        yield
        return
    selected_modules = _requested_modules()
    targets = list(_trace_targets(layer))
    if selected_modules is not None:
        unknown = selected_modules - {label for label, _module in targets}
        if unknown:
            raise ValueError(f"{_FILTER_ENV} contains unknown modules: {sorted(unknown)}")
    handles = []
    try:
        for label, module in targets:
            if selected_modules is not None and label not in selected_modules:
                continue
            if module is None:
                _emit(layer_idx, label, "unavailable")
                continue

            def pre_hook(_module, args, *, module_label=label):
                _emit(layer_idx, module_label, "enter")
                _synchronize(args)
                _emit(layer_idx, module_label, "inputs_synchronized")

            def post_hook(_module, _args, output, *, module_label=label):
                _emit(layer_idx, module_label, "returned")
                _synchronize(output)
                _emit(layer_idx, module_label, "output_synchronized")

            handles.append(module.register_forward_pre_hook(pre_hook, prepend=True))
            handles.append(module.register_forward_hook(post_hook))
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
