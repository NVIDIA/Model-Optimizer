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

"""Reach a converted module from the functions that vLLM's custom ops call.

A custom op gets a layer name instead of the layer, as an op argument cannot be a module, and the
kernels it calls get only tensors. vLLM maps the name back to the module registered under it in
``static_forward_context``, as its own ops do. :func:`register_layer_owner` links that module to
the converted module that owns it, :func:`wrap_layer_op` makes the owner :func:`layer_owner` while
an op that names the layer runs, and :func:`wrap_layer_callee` hands the owner the calls of a
function that such an op calls.

The wraps are permanent and process-wide. A breakable CUDA graph replays an op that
``eager_break_during_capture`` decorates by calling the undecorated function, so the scope goes
under that decorator. An op inside a captured graph has its scope while the graph is captured, so
the owner must launch only work that the graph can replay.
"""

import contextvars
import functools
import inspect
import weakref
from collections.abc import Callable
from types import ModuleType
from typing import Any

from vllm.forward_context import get_forward_context

__all__ = ["layer_owner", "register_layer_owner", "wrap_layer_callee", "wrap_layer_op"]

_OWNER_ATTR = "_modelopt_layer_owner"
# Inside an op of a layer that no converted module owns.
_NO_OWNER = object()
# The owner of the layer whose op runs: None outside a wrapped op.
_layer_owner: contextvars.ContextVar = contextvars.ContextVar("_layer_owner", default=None)


def register_layer_owner(layer: Any, owner: Any) -> None:
    """Make ``owner`` the owner of ``layer``, a module that vLLM registers under its layer name."""
    # A weak reference: a module assigned to a submodule would become its child.
    setattr(layer, _OWNER_ATTR, weakref.ref(owner))


def layer_owner() -> Any:
    """The owner of the layer whose op runs, or None."""
    owner = _layer_owner.get()
    return None if owner is _NO_OWNER else owner


def _owner_of(layer_name) -> Any:
    """The owner of the layer ``layer_name`` (a str or vLLM's ``LayerName``), or ``_NO_OWNER``."""
    name = getattr(layer_name, "value", layer_name)
    layer = get_forward_context().no_compile_layers.get(name)
    owner = getattr(layer, _OWNER_ATTR, None)
    owner = owner() if owner is not None else None
    return _NO_OWNER if owner is None else owner


def wrap_layer_op(
    op_module: ModuleType, name: str, layer_arg: str, decorator: Callable | None = None
) -> None:
    """Wrap the op ``op_module.<name>`` to put the owner of the layer it names in scope.

    ``layer_arg`` is the op's argument with the layer name. If ``decorator`` decorates the op, the
    wrap goes under it and ``decorator`` decorates the wrapper again.
    """
    original = getattr(op_module, name)
    if getattr(original, "_modelopt_layer_op", False):
        return
    inner = getattr(original, "__wrapped__", original) if decorator is not None else original
    position = list(inspect.signature(inner).parameters).index(layer_arg)

    @functools.wraps(inner)
    def wrapper(*args, **kwargs):
        token = _layer_owner.set(
            _owner_of(kwargs[layer_arg] if layer_arg in kwargs else args[position])
        )
        try:
            return inner(*args, **kwargs)
        finally:
            _layer_owner.reset(token)

    if inner is not original:
        wrapper = decorator(wrapper)  # type: ignore[misc]
    wrapper._modelopt_layer_op = True  # type: ignore[attr-defined]
    setattr(op_module, name, wrapper)


def wrap_layer_callee(
    module: ModuleType,
    name: str,
    call: Callable[[Any, Callable, inspect.BoundArguments], Any],
    applies: Callable[[Any], bool],
    outside: Callable[[], None] | None = None,
) -> None:
    """Wrap ``module.<name>``, a function that ops call, to hand its calls to the layer's owner.

    Inside an op that :func:`wrap_layer_op` wrapped, a call for whose owner ``applies(owner)``
    holds runs ``call(owner, original, bound)`` with the bound arguments of the call; other calls
    run the original function. ``outside()`` runs before a call outside such an op, e.g. to reject
    it.
    """
    original = getattr(module, name)
    if getattr(original, "_modelopt_layer_callee", False):
        return
    signature = inspect.signature(original)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        owner = _layer_owner.get()
        if owner is None and outside is not None:
            outside()
        if owner is None or owner is _NO_OWNER or not applies(owner):
            return original(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return call(owner, original, bound)

    wrapper._modelopt_layer_callee = True  # type: ignore[attr-defined]
    setattr(module, name, wrapper)
