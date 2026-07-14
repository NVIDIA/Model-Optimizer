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

"""Shared block/subblock granularity contracts for Puzzletron stages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeAlias, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["Granularity", "resolve_granularity"]

Granularity: TypeAlias = Literal["block", "subblock"]

_DEFAULTS: dict[str, Granularity] = {
    "depth": "subblock",
    "vllm_stats": "block",
    "calc_subblock_stats": "block",
    "scoring": "block",
    "bypass": "block",
}


def resolve_granularity(stage: str, config: Mapping[str, Any] | None = None) -> Granularity:
    """Resolve and validate a stage's analysis granularity."""

    config = config or {}
    value = config.get("granularity")
    if value is None:
        if stage not in _DEFAULTS:
            raise ValueError(f"stage {stage!r} has no granularity default")
        value = _DEFAULTS[stage]
    value = str(value)
    if value not in ("block", "subblock"):
        raise ValueError(f"granularity must be 'block' or 'subblock', got {value!r}")
    return cast("Granularity", value)
