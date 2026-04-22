# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wan 2.2 specific helpers for the DMD2 GAN branch.

The :class:`~modelopt.torch.fastgen.methods.dmd.DMDPipeline` GAN path needs intermediate
activations from the teacher transformer. Rather than modifying the Wan model class,
:func:`attach_feature_capture` installs PyTorch forward hooks on the requested blocks
and stashes their outputs on ``teacher._fastgen_captured``; ``DMDPipeline`` drains that
buffer via :func:`pop_captured_features` after each teacher forward.

Hooks are installed against ``teacher.blocks``, which is the attribute name used by the
Hugging Face diffusers ``WanTransformer3DModel``. Subclasses / custom forks that expose
the transformer stack under a different attribute should pass ``blocks_attr``.

Importing this module transitively imports ``diffusers`` only if the optional dependency
is available — see :mod:`modelopt.torch.fastgen.plugins` for the gating logic.
"""

from __future__ import annotations

import contextlib
from typing import Any

from torch import Tensor, nn

__all__ = [
    "attach_feature_capture",
    "pop_captured_features",
    "remove_feature_capture",
]

_CAPTURED_ATTR = "_fastgen_captured"
_HANDLES_ATTR = "_fastgen_capture_handles"
_INDICES_ATTR = "_fastgen_capture_indices"


def _extract_tensor(output: Any) -> Tensor:
    """Return a single ``Tensor`` from a hook output, unwrapping tuples / ModelOutput."""
    if isinstance(output, Tensor):
        return output
    if isinstance(output, tuple):
        return output[0]
    if hasattr(output, "sample"):
        return output.sample
    # Some transformer blocks return (hidden_states, residual) or similar; take the first tensor-like value.
    if hasattr(output, "__iter__"):
        for item in output:
            if isinstance(item, Tensor):
                return item
    raise TypeError(f"Cannot extract a Tensor from block output of type {type(output).__name__!r}.")


def attach_feature_capture(
    teacher: nn.Module,
    feature_indices: list[int],
    *,
    blocks_attr: str = "blocks",
) -> None:
    """Install forward hooks on ``teacher.<blocks_attr>[i]`` for every ``i`` in ``feature_indices``.

    On every forward of the teacher, each hooked block appends its output tensor to
    ``teacher._fastgen_captured``. The list is drained by
    :func:`pop_captured_features` (usually called by :class:`DMDPipeline` after each
    teacher forward).

    Calling this function a second time removes the previous hooks first, so it is safe
    to reinstall with a different index set.

    Args:
        teacher: The teacher transformer module.
        feature_indices: Block indices to capture (e.g. ``[15, 22, 29]`` for a 30-block
            Wan 2.2 5B teacher).
        blocks_attr: Attribute under which the teacher exposes its transformer block
            stack. Default ``"blocks"`` matches diffusers' ``WanTransformer3DModel``.
    """
    remove_feature_capture(teacher)

    blocks = getattr(teacher, blocks_attr, None)
    if blocks is None:
        raise AttributeError(
            f"Teacher {type(teacher).__name__!r} does not expose a ``{blocks_attr}`` attribute; "
            f"pass ``blocks_attr='<attr>'`` to :func:`attach_feature_capture` if the block stack "
            f"is named differently."
        )
    try:
        num_blocks = len(blocks)
    except TypeError as exc:
        raise TypeError(
            f"Teacher ``{blocks_attr}`` is not a sequence (got {type(blocks).__name__!r})."
        ) from exc

    sorted_indices = sorted(set(feature_indices))
    for idx in sorted_indices:
        if not (0 <= idx < num_blocks):
            raise IndexError(
                f"feature_indices entry {idx} is out of range for teacher with {num_blocks} blocks."
            )

    captured: list[Tensor] = []
    setattr(teacher, _CAPTURED_ATTR, captured)
    setattr(teacher, _INDICES_ATTR, list(sorted_indices))

    handles: list[Any] = []
    for idx in sorted_indices:
        block = blocks[idx]

        def _hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            captured.append(_extract_tensor(output))

        handles.append(block.register_forward_hook(_hook))

    setattr(teacher, _HANDLES_ATTR, handles)


def remove_feature_capture(teacher: nn.Module) -> None:
    """Remove previously installed feature-capture hooks (no-op if none are installed)."""
    handles = getattr(teacher, _HANDLES_ATTR, None)
    if handles:
        for h in handles:
            h.remove()
    for attr in (_HANDLES_ATTR, _CAPTURED_ATTR, _INDICES_ATTR):
        if hasattr(teacher, attr):
            with contextlib.suppress(AttributeError):
                delattr(teacher, attr)


def pop_captured_features(teacher: nn.Module) -> list[Tensor]:
    """Return captured features in block order and clear the internal buffer.

    Callers should invoke this immediately after each teacher forward to avoid stacking
    features across forwards.
    """
    captured = getattr(teacher, _CAPTURED_ATTR, None)
    if captured is None:
        raise RuntimeError(
            "Teacher has no captured features — did you forget to call "
            ":func:`attach_feature_capture`?"
        )
    out = list(captured)
    captured.clear()
    return out
