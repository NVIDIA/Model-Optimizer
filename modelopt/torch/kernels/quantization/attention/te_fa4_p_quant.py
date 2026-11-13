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

"""Pass FlashAttention-4's ``p_quant`` mode through Transformer Engine's FA4 call sites.

FlashAttention-4 (``flash_attn.cute``) can fake-quantize the attention probabilities inside
its Blackwell forward kernel (``p_quant="fp8_e4m3"`` or ``"nvfp4"``, straight-through
backward). Transformer Engine (TE) has no argument for the mode, so it is added by wrapping
the FA4 forward names TE binds on its ``backends`` module: ``flash_attn_func_v4``,
``flash_attn_varlen_func_v4`` and ``_flash_attn_fwd_v4`` (TE's context-parallel code looks
the last one up at call time, so p2p, all-gather and a2a are covered).

The mode is carried by a context variable set for the duration of one attention forward
(:func:`p_quant_scope`), not by process state: a teacher model in the same process stays
unquantized, an exception leaves nothing set, and a forward replayed by activation
recomputation is quantized again because it re-enters the scope.

The wrappers also spell TE's unbounded window sides as ``None``. TE marks them with -1 and
passes ``window_size_left=-1, window_size_right=0`` for a causal mask; FA4 revisions that have
``p_quant`` read -1 as unbounded only when both sides are negative, so that pair would be a
window in which no key is visible. The backward name ``_flash_attn_bwd_v4`` gets the same
translation and nothing else: TE passes the windows to the backward separately, and a
backward left untranslated would use another mask than its forward. Only -1 is translated;
other negative sides keep their meaning. The translation applies to every FA4 call once
installed, quantized or not.
"""

import contextvars
import functools
import importlib
from contextlib import contextmanager

__all__ = ["fa4_p_quant_formats", "install", "p_quant_scope"]

TE_BACKENDS = "transformer_engine.pytorch.attention.dot_product_attention.backends"
FORWARD_NAMES = ("flash_attn_func_v4", "flash_attn_varlen_func_v4", "_flash_attn_fwd_v4")
BACKWARD_NAMES = ("_flash_attn_bwd_v4",)


class _Scope:
    """The mode of one attention forward and how many FA4 forward calls received it."""

    def __init__(self, p_quant: str):
        self.p_quant = p_quant
        self.calls = 0


_scope: contextvars.ContextVar[_Scope | None] = contextvars.ContextVar(
    "modelopt_fa4_p_quant", default=None
)


@contextmanager
def p_quant_scope(p_quant: str):
    """Run the FA4 forward calls made inside the block with ``p_quant``.

    Yields an object whose ``calls`` counts those calls, so the caller can tell that the
    attention did not go through FA4 at all (TE chose another backend).
    """
    scope = _Scope(p_quant)
    token = _scope.set(scope)
    try:
        yield scope
    finally:
        _scope.reset(token)


def fa4_p_quant_formats() -> tuple[str, ...]:
    """Formats the installed FlashAttention-4 accepts for ``p_quant``; empty if it has none."""
    try:
        from flash_attn.cute.p_quant import p_quant_format_names
    except ImportError:
        return ()
    return tuple(p_quant_format_names())


def _normalize_te_windows(kwargs: dict) -> None:
    for name in ("window_size_left", "window_size_right"):
        if kwargs.get(name) == -1:
            kwargs[name] = None
    if kwargs.get("window_size") is not None:
        kwargs["window_size"] = tuple(
            None if side == -1 else side for side in kwargs["window_size"]
        )


def _wrap(fn, forward: bool):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        _normalize_te_windows(kwargs)
        scope = _scope.get() if forward else None
        if scope is not None:
            kwargs["p_quant"] = scope.p_quant
            scope.calls += 1
        return fn(*args, **kwargs)

    setattr(wrapper, "_modelopt_fa4_p_quant", True)  # marks the name as wrapped
    return wrapper


def install(backends=None) -> bool:
    """Wrap TE's FA4 names (idempotent).

    Returns False, changing nothing, when the route does not exist: TE has no FA4 backend
    bound, or the installed FA4 has no ``p_quant``.
    """
    if not fa4_p_quant_formats():
        return False
    if backends is None:
        try:
            backends = importlib.import_module(TE_BACKENDS)
        except ImportError:
            return False
    names = FORWARD_NAMES + BACKWARD_NAMES
    if any(getattr(backends, name, None) is None for name in names):
        return False
    for name in names:
        fn = getattr(backends, name)
        if not getattr(fn, "_modelopt_fa4_p_quant", False):
            setattr(backends, name, _wrap(fn, forward=name in FORWARD_NAMES))
    return True
