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

"""Explicit exact-dense coefficients, never presented as measured sparse statistics."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

from .mask_reuse import MaskReuseCalibrationError

__all__ = ["ANALYTICAL_DENSE_EVIDENCE", "validate_analytical_dense"]

ANALYTICAL_DENSE_EVIDENCE = {
    "reason": "threshold_disabled",
    "basis": "threshold_log2>=0 implies native dense attention",
    "measured": False,
}


def validate_analytical_dense(raw: Mapping[str, Any], *, version_field: str) -> bool:
    """Validate the version-2 all-keep branch before ordinary structural validation."""
    if type(raw.get(version_field)) is not int or raw[version_field] not in (1, 2):
        raise MaskReuseCalibrationError(f"{version_field} must be an integer supported version")
    if raw.get(version_field) != 2:
        return False
    if raw.get("analytical_dense") != ANALYTICAL_DENSE_EVIDENCE:
        raise MaskReuseCalibrationError(
            "analytical dense coefficients need explicit non-measured provenance"
        )
    invocation = raw.get("invocation", {})
    try:
        log_threshold = float.fromhex(invocation["threshold_log2_hex"])
        threshold = float.fromhex(invocation["threshold_lambda_hex"])
    except (KeyError, TypeError, ValueError) as error:
        raise MaskReuseCalibrationError("analytical dense threshold is invalid") from error
    if (
        not math.isfinite(log_threshold)
        or not math.isfinite(threshold)
        or log_threshold < 0
        or threshold < 1
    ):
        raise MaskReuseCalibrationError(
            "analytical dense coefficients require a disabled threshold"
        )
    eligible = raw.get("eligible_tiles")
    for stats in raw.get("anchor_stats_by_layer", {}).values():
        if any(
            type(value) is not int or value != eligible for value in stats.get("retained_tiles", [])
        ):
            raise MaskReuseCalibrationError(
                "analytical dense anchor must retain every eligible tile"
            )
        if any(
            type(value) not in (int, float) or value != 0 for value in stats.get("dropped_mass", [])
        ):
            raise MaskReuseCalibrationError("analytical dense anchor must have zero dropped mass")
    consumers = raw.get("consumer_layers")
    matrices = (
        [entry["dropped_mass"] for entry in consumers.values()]
        if consumers is not None
        else [
            entry["dropped_mass"]
            for by_anchor in raw.get("consumer_candidates_by_layer", {}).values()
            for entry in by_anchor.values()
        ]
    )
    if any(
        type(value) not in (int, float) or value != 0
        for matrix in matrices
        for row in matrix
        for value in row
    ):
        raise MaskReuseCalibrationError("analytical dense consumers must have zero dropped mass")
    return True
