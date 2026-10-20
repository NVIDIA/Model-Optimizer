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

"""Reach a converted module from the functions that vLLM's layer ops call.

vLLM's layer ops get the name of their layer, not the layer: its ``torch.ops.vllm`` custom ops
cannot take a module argument, and other ops, such as GLM-5.3-Flash's indexer op, follow the same
convention. The kernels they call get only tensors. :func:`wrap_layer_op` makes the layer that an op
names the :func:`current_layer` while the op runs, :func:`register_layer_owner` maps a layer name to
the converted module that acts for the layer, and :func:`current_layer_owner` gives that module for
the current layer. A function that the op calls, patched with :func:`patch_function`, can thus act
for the converted module.

The registry and the patches are process-wide, for one converted model per process: another model
with the same layer names would get its owners. The patches are permanent. A breakable CUDA graph
replays an op that ``eager_break_during_capture`` decorates by calling the undecorated function, so
the scope goes under that decorator. An op inside a captured graph has its scope while the graph is
captured, so the owner must launch only work that the graph can replay.
"""

import contextvars
import functools
import inspect
import weakref
from collections.abc import Callable
from types import ModuleType
from typing import Any

__all__ = [
    "current_layer",
    "current_layer_owner",
    "patch_function",
    "register_layer_owner",
    "wrap_layer_op",
]

# Layer name -> the converted module that acts for the layer. Weak: it keeps no module alive.
_owners: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
# The name of the layer whose op runs: None outside a wrapped op.
_current_layer: contextvars.ContextVar = contextvars.ContextVar("_current_layer", default=None)


def register_layer_owner(name: str, owner: Any) -> None:
    """Make ``owner`` act for the layer ``name``, as the layer's ops name it; the latest wins."""
    _owners[name] = owner


def current_layer() -> str | None:
    """The name of the layer whose op runs, or None outside a wrapped op."""
    return _current_layer.get()


def current_layer_owner() -> Any:
    """The owner of the layer whose op runs, or None outside a wrapped op or without an owner."""
    name = _current_layer.get()
    return None if name is None else _owners.get(name)


def wrap_layer_op(
    op_module: ModuleType, name: str, layer_arg: str, decorator: Callable | None = None
) -> None:
    """Wrap the op ``op_module.<name>`` to make the layer it names the current layer.

    ``layer_arg`` is the op's argument with the layer name. If ``decorator`` decorates the op, the
    wrap goes under it and ``decorator`` decorates the wrapper again. A second wrap does nothing.
    """
    original = getattr(op_module, name)
    if getattr(original, "_modelopt_layer_op", False):
        return
    inner = getattr(original, "__wrapped__", original) if decorator is not None else original
    position = list(inspect.signature(inner).parameters).index(layer_arg)

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):
        layer = kwargs[layer_arg] if layer_arg in kwargs else args[position]
        # A str or vLLM's LayerName.
        token = _current_layer.set(getattr(layer, "value", layer))
        try:
            return inner(*args, **kwargs)
        finally:
            _current_layer.reset(token)

    if inner is not original:
        wrapper = decorator(wrapper)  # type: ignore[misc]
    wrapper._modelopt_layer_op = True  # type: ignore[attr-defined]
    setattr(op_module, name, wrapper)


def patch_function(module: ModuleType, name: str, wrapper: Callable) -> None:
    """Replace ``module.<name>`` so that its calls run ``wrapper(original, *args, **kwargs)``.

    A second patch with the same ``wrapper`` does nothing; another wrapper wraps the patched one.
    """
    original = getattr(module, name)
    patched_fn = original
    while patched_fn is not None:
        if getattr(patched_fn, "_modelopt_patched_by", None) is wrapper:
            return
        patched_fn = getattr(patched_fn, "__wrapped__", None)

    @functools.wraps(original)
    def patched(*args, **kwargs):
        return wrapper(original, *args, **kwargs)

    patched._modelopt_patched_by = wrapper  # type: ignore[attr-defined]
    setattr(module, name, patched)
