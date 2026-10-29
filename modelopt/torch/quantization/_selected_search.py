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

"""Bind frozen assignments to reusable search checkpoints without copying payloads."""

import hashlib
from pathlib import Path

from modelopt.torch.utils import safe_load

from ._calibration_store import _CalibrationStore
from ._calibration_workers import _check_provenance, _digest
from ._layerwise_checkpoint import _fingerprint
from ._source_estimation import _SOURCE_ACTIVATION_ESTIMATOR
from .algorithms import _AutoQuantizeBaseSearcher

__all__ = []


def _selected_search_binding(checkpoint, assignment, max_bytes):
    """Fingerprint an accepted coordinator checkpoint under its existing writer lease."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Selected search recovery requires a positive checkpoint byte limit")
    store = _CalibrationStore(Path(checkpoint).parent)
    path = Path(checkpoint) / "rank0.pth"
    with store.exclusive(checkpoint):
        if path.stat().st_size > max_bytes:
            raise ValueError("Selected search checkpoint exceeds its byte limit")
        return {"assignment": assignment.digest, "checkpoint": _fingerprint(path)}


def _load_selected_search(checkpoint, binding, assignment, plan, identity, steps, max_bytes):
    """Recover one frozen solver result after the search model has left scope.

    The caller publishes the binding only after coordinator validation, retains its workspace
    lease and recomputes source/input/callback identity on freshly loaded original weights.
    """
    # Recipe planning imports quantization; resolve the contract after package initialization.
    from modelopt.recipe.planner import freeze_auto_quantize_assignment

    plan.require_supported()
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("Selected search recovery requires a positive checkpoint byte limit")
    store = _CalibrationStore(Path(checkpoint).parent)
    path = Path(checkpoint) / "rank0.pth"
    with store.exclusive(checkpoint):
        if path.stat().st_size > max_bytes:
            raise ValueError("Selected search checkpoint exceeds its byte limit")
        if binding != {"assignment": assignment.digest, "checkpoint": _fingerprint(path)}:
            raise ValueError("Selected search checkpoint or assignment differs from its binding")
        state = safe_load(path, map_location="cpu", weights_only=True)
    provenance = state["calibration_provenance"]
    contract = provenance["contract"]
    if (
        provenance["digest"] != _digest(contract)
        or contract["version"] != 2
        or contract.get("layerwise_estimator") != _SOURCE_ACTIVATION_ESTIMATOR
        or contract["steps"] != list(steps)
    ):
        raise ValueError("Selected search estimation or sample-count provenance differs")
    expected = dict(contract, identity=identity)
    _check_provenance(provenance, {"digest": _digest(expected), "contract": expected})
    selected = assignment.as_dict()
    if selected["requested_constraints"] != plan.as_dict()["constraints"]:
        raise ValueError("Selected search budget differs from the frozen assignment")
    recipes = {}
    for group in selected["groups"]:
        candidates = state["candidate_stats"][group["name"]]["formats"]
        matches = [
            recipe
            for recipe in candidates
            if (
                None
                if recipe.is_no_quant
                else hashlib.sha256(recipe.checkpoint_signature.encode()).hexdigest()
            )
            == group["format_id"]
        ]
        if len(matches) != 1:
            raise ValueError("Selected search recipe is missing or ambiguous")
        recipes[group["name"]] = matches[0]
    # Native search checkpoints precede solving. Reuse their canonical recipe objects: rebuilding
    # QuantRecipe from serialized configs would append output-quantizer rules a second time.
    state["best"] = {
        "recipe": recipes,
        "constraints": selected["constraints"],
        "is_satisfied": True,
    }
    state["cost_denominator"] = (
        _AutoQuantizeBaseSearcher._get_total_weight_size_from_candidate_stats(
            state["candidate_stats"]
        )
    )
    state["best"]["score"] = sum(
        stat["scores"][stat["formats"].index(recipes[name])]
        for name, stat in state["candidate_stats"].items()
    )
    if freeze_auto_quantize_assignment(state, plan, identity).digest != assignment.digest:
        raise ValueError("Selected search mappings or configuration differ from the assignment")
    return state
