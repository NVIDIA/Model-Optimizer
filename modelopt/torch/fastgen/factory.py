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

"""Convenience factory helpers for constructing the auxiliary DMD networks.

These helpers are intentionally tiny — the training framework is free to build the
fake score directly (e.g. under a meta-init context for FSDP2) instead of calling
:func:`create_fake_score`. See the ModelOpt ↔ FastGen design doc (FASTGEN_MODELOPT.md,
section "How the framework can build the fake_score") for both options.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn

__all__ = ["create_fake_score"]


def _looks_fsdp_wrapped(module: nn.Module) -> bool:
    """Best-effort detection of an FSDP-wrapped module.

    Matches two shapes:

    - **FSDP1** (``torch.distributed.fsdp.FullyShardedDataParallel``): child modules
      carry an ``_fsdp_wrapped_module`` attribute.
    - **FSDP2** (``torch.distributed._composable.fsdp.fully_shard``): parameters are
      ``DTensor`` instances exposing a ``full_tensor`` method.

    Probes only the first parameter to avoid iterating a large model. Intended for
    the ``create_fake_score`` fast-fail check, not as a general-purpose predicate.
    """
    if any(hasattr(m, "_fsdp_wrapped_module") for m in module.modules()):
        return True
    for p in module.parameters():
        if hasattr(p, "full_tensor"):
            return True
        break  # only probe the first param
    return False


def create_fake_score(teacher: nn.Module, *, deep_copy: bool = True) -> nn.Module:
    """Return a trainable fake-score network initialized from the teacher.

    This is the unit-test / single-script path; frameworks that do meta-init + FSDP2
    wrapping will typically construct the fake score themselves and pass it directly
    into :class:`~modelopt.torch.fastgen.methods.dmd.DMDPipeline`.

    Args:
        teacher: The already-built teacher module. Must already have its weights loaded.
        deep_copy: If True, :func:`copy.deepcopy` the teacher; if False, reuse the same
            instance (only sensible if the caller can guarantee it is no longer held
            elsewhere as the frozen teacher).

    Returns:
        A copy of ``teacher`` in training mode with all parameters requiring gradients.

    FSDP2 caveat
    ------------
    ``copy.deepcopy(teacher)`` is **not safe** when the teacher is already FSDP2-wrapped
    (DTensor parameters + FSDP pre/post hooks + meta-init bookkeeping). For Stage-2 FSDP2
    training, skip this factory and construct the fake score under meta-init, then
    rank-0-load weights and let ``sync_module_states`` broadcast::

        with meta_init_context():
            fake_score = build_teacher_from_config(teacher_config)
        if is_rank0():
            fake_score.load_state_dict(teacher.state_dict(), strict=False)
        # Wrap with FSDP2(..., sync_module_states=True) to broadcast from rank 0.

    The pattern mirrors FastGen's
    ``methods/distribution_matching/dmd2.py::DMD2Model.build_model``. A dedicated
    ``create_fake_score_meta`` factory is planned alongside the Stage-2 training example.

    Raises:
        RuntimeError: When ``deep_copy=True`` and the teacher looks FSDP-wrapped
            (either FSDP1 via ``_fsdp_wrapped_module`` or FSDP2 via DTensor
            parameters). The ``deep_copy=False`` branch skips the check because
            reusing the teacher directly is compatible with an FSDP-wrapped input.
    """
    if deep_copy and _looks_fsdp_wrapped(teacher):
        raise RuntimeError(
            "create_fake_score(deep_copy=True) is not safe on an FSDP-wrapped teacher "
            "(DTensor parameters + FSDP hooks + meta-init bookkeeping don't survive "
            "copy.deepcopy). Construct the fake score under meta-init and rank-0-load "
            "weights instead — see the 'FSDP2 caveat' section of this function's docstring."
        )
    fake_score = copy.deepcopy(teacher) if deep_copy else teacher
    fake_score.train()
    fake_score.requires_grad_(True)
    return fake_score
