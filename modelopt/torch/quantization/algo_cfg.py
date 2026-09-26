# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""What each calibration algorithm reads, writes and assumes."""

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "WRITABLE_TOKENS",
    "AlgoCapabilities",
    "capabilities_for",
]


@dataclass(frozen=True)
class AlgoCapabilities:
    """What one calibration algorithm reads, writes and assumes."""

    #: Writes *every* quantizer of each linear it touches, not one quantizer at a time.
    writes_whole_module: bool
    #: Role this algorithm *improves*. Narrower than what it writes: weight-side algorithms
    #: also seed input amax via an internal `max_calibrate`, which `may_write` records.
    refines: Literal["weight", "input", "both"]
    #: Tokens this algorithm reads. ``weight`` and ``acts`` are ambient, so never counted.
    #: Not a precondition: every algorithm listing ``weight_amax`` also seeds its own via an
    #: internal ``max_calibrate``, so an algorithm run on its own is fine.
    requires: frozenset[str] = frozenset()
    #: Tokens this may write. An *upper* bound: it may write fewer on a given model (smoothquant
    #: only touches INT8 layers), never more. Over-declaring is safe for conflict detection and
    #: unsafe for the hand-off, which is why the hand-off also checks coverage.
    may_write: frozenset[str] = frozenset()
    #: Tokens whose presence makes this algorithm incorrect: ``awq_lite`` folds a scale into
    #: the weight assuming an unsmoothed start, so it conflicts with ``pre_quant_scale``.
    invalid_if_present: frozenset[str] = frozenset()
    #: Can be restricted to a scope, i.e. threads the ``should_process`` write-mask through
    #: everything it writes. ``False`` forces whole-model scope; the compiler rejects the rest.
    scopable: bool = True


WEIGHT_AMAX = "weight_amax"
INPUT_AMAX = "input_amax"
PRE_QUANT_SCALE = "pre_quant_scale"
WEIGHT = "weight"
ACTS = "acts"

#: Conservative default for an algorithm that declares nothing: over-reporting conflicts is
#: the safe direction.
WRITABLE_TOKENS = frozenset({WEIGHT, WEIGHT_AMAX, INPUT_AMAX, PRE_QUANT_SCALE})


def capabilities_for(algo: str | None, cfg: dict | None = None) -> AlgoCapabilities | None:
    """Capabilities of ``algo``, read off its calibrate-mode descriptor."""
    if algo is None:
        return None
    # Imported lazily: `mode` imports this module while the package is still initializing.
    from .mode import BaseCalibrateModeDescriptor, CalibrateModeRegistry

    descriptor = CalibrateModeRegistry.get(BaseCalibrateModeDescriptor._get_mode_name(algo))
    if descriptor is None:
        return None
    return type(descriptor).capabilities_for_cfg(cfg or {})
