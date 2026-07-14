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

"""Stable identities for recomputed sublayer-depth trajectories."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..distributed_eval.schema import StrictModel
from ..granularity import Granularity  # noqa: TC001 - Pydantic resolves this type at runtime.
from ..identity import stable_hash

__all__ = ["DepthScenario", "SublayerRemoval"]


class SublayerRemoval(StrictModel):
    layer_idx: int = Field(ge=0)
    kind: Literal["block", "attention", "mamba", "ffn", "moe"]


class DepthScenario(StrictModel):
    parent_checkpoint_identity: str
    hidden_width: int = Field(gt=0)
    removals: tuple[SublayerRemoval, ...] = ()
    data_identity: str
    evaluator_revision: str
    metric: Literal["lm_loss"] = "lm_loss"
    granularity: Granularity = "subblock"

    @property
    def scenario_id(self) -> str:
        return stable_hash(self.model_dump(mode="python"), prefix="depth_scenario")
