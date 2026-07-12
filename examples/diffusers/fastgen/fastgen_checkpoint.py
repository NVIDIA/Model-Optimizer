# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""FSDP2 partial-load-tolerant optimizer restore for the DMD2 recipe.

Stock upstream ``nemo_automodel`` restores the optimizer state with a strict DCP load,
which can trip on FSDP2-sharded params whose shards are zero-length on some ranks. The
DMD2 recipe needs the tolerant behavior for both the parent student-optimizer restore
(inside ``super().load_checkpoint``) and the fake-score optimizer restore
(``_restore_dmd_extras``).

Rather than modify ``nemo_automodel`` (which the published example cannot do), this module
provides a thin :class:`Checkpointer` subclass that overrides **only** ``load_optimizer`` to
pass ``DefaultLoadPlanner(allow_partial_load=True)``. The model-state load (``load_model``)
and everything else are inherited unchanged from stock upstream, so model checkpoints still
load strictly.

The recipe upgrades its already-constructed ``self.checkpointer`` instance in place via
:func:`make_optimizer_partial_load_tolerant` (an instance-scoped re-bless — it does NOT
patch the global ``Checkpointer`` class or any other recipe's checkpointer).
"""

from __future__ import annotations

import os
from typing import Any

import torch
import torch.distributed.checkpoint as dcp
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.stateful_wrappers import OptimizerState
from torch import nn
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

__all__ = ["PartialLoadCheckpointer", "make_optimizer_partial_load_tolerant"]


class PartialLoadCheckpointer(Checkpointer):
    """``Checkpointer`` whose optimizer restore tolerates FSDP2 partial shards.

    Overrides only ``load_optimizer`` (model load stays strict). The body mirrors stock
    upstream's ``load_optimizer`` exactly except that the DCP load uses
    ``DefaultLoadPlanner(allow_partial_load=True)`` so params with no saved state simply
    keep their freshly-initialised optimizer defaults instead of raising on missing keys.
    """

    def load_optimizer(
        self,
        optimizer: torch.optim.Optimizer,
        model: nn.Module,
        weights_path: str,
        scheduler: Any | None = None,
    ) -> None:
        """Load optimizer (and optional scheduler) state from ``weights_path/optim`` via DCP."""
        optimizer_state = OptimizerState(model, optimizer, scheduler, is_peft=self.config.is_peft)
        state_dict = optimizer_state.state_dict()
        planner = DefaultLoadPlanner(allow_partial_load=True)
        path = os.path.join(weights_path, "optim")
        dcp.load(state_dict, checkpoint_id=path, planner=planner)
        optimizer_state.load_state_dict(state_dict)


def make_optimizer_partial_load_tolerant(checkpointer: Checkpointer) -> Checkpointer:
    """Upgrade an existing ``Checkpointer`` instance in place to tolerate partial optimizer loads.

    Re-blesses the instance's class to :class:`PartialLoadCheckpointer`. This is safe because
    the subclass adds no new state and only overrides ``load_optimizer``; all existing instance
    state and other behavior are preserved. Instance-scoped (does not touch the global
    ``Checkpointer`` class). Idempotent.

    Returns the same instance for convenience.
    """
    if not isinstance(checkpointer, PartialLoadCheckpointer):
        # Instance-scoped upgrade: only this checkpointer object gains the override.
        checkpointer.__class__ = PartialLoadCheckpointer
    return checkpointer
