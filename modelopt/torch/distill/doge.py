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

"""Data-blend weight update API for DoGE distillation."""

import hashlib
import math
from collections.abc import Mapping, Sequence

__all__ = [
    "DoGEWeightUpdater",
    "apply_source_max_blend_weights",
    "apply_source_min_blend_weights",
    "normalize_data_path_weights",
    "resolve_source_max_blend_weights",
    "resolve_source_min_blend_weights",
    "sample_data_path_by_weight",
]


def normalize_data_path_weights(data_paths: Sequence[str]) -> dict[str, float]:
    """Normalize a Megatron WEIGHT PATH list into weights keyed by dataset path.

    For example, ``["2", "/data/a", "1", "/data/b"]`` becomes
    ``{"/data/a": 2 / 3, "/data/b": 1 / 3}``.
    """
    if len(data_paths) % 2 != 0:
        raise ValueError("data path list must contain WEIGHT PATH pairs")

    blend_weights: dict[str, float] = {}
    for weight_value, path in zip(data_paths[::2], data_paths[1::2]):
        if path in blend_weights:
            raise ValueError(f"duplicate dataset path in data blend: {path}")
        weight = float(weight_value)
        if weight <= 0:
            raise ValueError(f"blend weights must be positive, got {weight_value!r}")
        blend_weights[path] = weight

    total_weight = sum(blend_weights.values())
    return {path: weight / total_weight for path, weight in blend_weights.items()}


def sample_data_path_by_weight(weights: Mapping[str, float], iteration: int, seed: int) -> str:
    """Return a deterministic weighted sample from normalized data-path weights.

    The sample is derived only from ``iteration`` and ``seed``, so tensor-parallel ranks that use
    the same blend state choose the same source without communication.
    """
    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}.")

    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise ValueError("sampled data-blend weights must sum to a positive value.")
    if any(weight < 0 for weight in weights.values()):
        raise ValueError("sampled data-blend weights must be non-negative.")

    digest = hashlib.sha256(f"{seed}:{iteration}".encode()).digest()
    threshold = int.from_bytes(digest[:8], byteorder="big") / 2**64 * total_weight
    cumulative_weight = 0.0
    last_path = None
    for path, weight in weights.items():
        last_path = path
        cumulative_weight += weight
        if threshold < cumulative_weight:
            return path

    if last_path is None:
        raise ValueError("sampled data-blend weights cannot be empty.")
    return last_path


def _resolve_source_weight_constraints(
    constraints: Mapping[str, float] | None, source_paths: Sequence[str], constraint_name: str
) -> dict[str, float]:
    """Resolve source-specific weight constraints to concrete source paths.

    Keys can be full source paths or unique path suffixes. This keeps command lines readable while
    still failing early if a suffix is ambiguous.
    """
    if not constraints:
        return {}

    resolved: dict[str, float] = {}
    for source_spec, weight in constraints.items():
        if weight < 0:
            raise ValueError(
                f"source {constraint_name} blend weight must be non-negative: {source_spec}"
            )
        if weight > 1:
            raise ValueError(
                f"source {constraint_name} blend weight must be at most 1: {source_spec}"
            )

        if source_spec in source_paths:
            source_path = source_spec
        else:
            matches = [path for path in source_paths if path.endswith(source_spec)]
            if not matches:
                raise ValueError(
                    f"source {constraint_name} blend weight did not match any source: {source_spec}"
                )
            if len(matches) > 1:
                raise ValueError(
                    f"source {constraint_name} blend weight matched multiple sources: {source_spec}"
                )
            source_path = matches[0]

        if source_path in resolved:
            raise ValueError(
                f"duplicate source {constraint_name} blend weight for source: {source_path}"
            )
        resolved[source_path] = weight

    return resolved


def resolve_source_min_blend_weights(
    source_min_blend_weights: Mapping[str, float] | None, source_paths: Sequence[str]
) -> dict[str, float]:
    """Resolve source-specific minimum weights to concrete source paths."""
    resolved = _resolve_source_weight_constraints(source_min_blend_weights, source_paths, "minimum")

    if sum(resolved.values()) >= 1:
        raise ValueError("source minimum blend weights must sum to less than 1.")
    return resolved


def resolve_source_max_blend_weights(
    source_max_blend_weights: Mapping[str, float] | None, source_paths: Sequence[str]
) -> dict[str, float]:
    """Resolve source-specific maximum weights to concrete source paths."""
    return _resolve_source_weight_constraints(source_max_blend_weights, source_paths, "maximum")


def apply_source_min_blend_weights(
    weights: Mapping[str, float], source_min_blend_weights: Mapping[str, float]
) -> dict[str, float]:
    """Apply source-specific minimum weights and renormalize the remaining sources."""
    if not source_min_blend_weights:
        return dict(weights)

    missing_paths = set(source_min_blend_weights) - set(weights)
    if missing_paths:
        raise ValueError(f"source minimum blend weights contain unknown sources: {missing_paths}")

    active_min_weights = {
        path: min_weight
        for path, min_weight in source_min_blend_weights.items()
        if weights[path] < min_weight
    }
    if not active_min_weights:
        return dict(weights)

    min_weight_sum = sum(active_min_weights.values())
    if min_weight_sum >= 1:
        raise ValueError("active source minimum blend weights must sum to less than 1.")

    remaining_paths = [path for path in weights if path not in active_min_weights]
    remaining_weight = 1.0 - min_weight_sum
    current_remaining_weight = sum(weights[path] for path in remaining_paths)
    if current_remaining_weight <= 0:
        raise ValueError(
            "cannot apply source minimum blend weights when no mass remains to rescale."
        )

    scale = remaining_weight / current_remaining_weight
    constrained_weights = {
        path: weights[path] * scale for path in remaining_paths
    } | active_min_weights
    return {path: constrained_weights[path] for path in weights}


def apply_source_max_blend_weights(
    weights: Mapping[str, float], source_max_blend_weights: Mapping[str, float]
) -> dict[str, float]:
    """Apply source-specific maximum weights and renormalize the remaining sources."""
    if not source_max_blend_weights:
        return dict(weights)

    missing_paths = set(source_max_blend_weights) - set(weights)
    if missing_paths:
        raise ValueError(f"source maximum blend weights contain unknown sources: {missing_paths}")

    capped_weights: dict[str, float] = {}
    free_paths = set(weights)
    while True:
        capped_weight_sum = sum(capped_weights.values())
        remaining_weight = 1.0 - capped_weight_sum
        if remaining_weight < 0:
            raise ValueError("source maximum blend weights over-constrain the blend.")

        current_free_weight = sum(weights[path] for path in free_paths)
        if current_free_weight <= 0:
            if remaining_weight == 0:
                constrained_weights = capped_weights
                break
            raise ValueError(
                "cannot apply source maximum blend weights when no mass remains to rescale."
            )

        scale = remaining_weight / current_free_weight
        candidate_free_weights = {path: weights[path] * scale for path in free_paths}
        newly_capped_weights = {
            path: source_max_blend_weights[path]
            for path, weight in candidate_free_weights.items()
            if path in source_max_blend_weights and weight > source_max_blend_weights[path]
        }
        if not newly_capped_weights:
            constrained_weights = capped_weights | candidate_free_weights
            break

        capped_weights.update(newly_capped_weights)
        free_paths -= set(newly_capped_weights)

    return {path: constrained_weights[path] for path in weights}


class DoGEWeightUpdater:
    """Outer-loop updater for DoGE data-blend weights.

    Args:
        meta_lr: Learning rate for exponentiated blend-weight updates.
        min_weight: Optional minimum normalized weight for each source after every update.

    Outputs:
        ``update`` returns normalized blend weights after applying the update.
    """

    def __init__(self, meta_lr: float, min_weight: float = 0.0) -> None:
        """Initialize the updater."""
        if min_weight < 0:
            raise ValueError(f"min_weight must be non-negative, got {min_weight}.")
        self.meta_lr = meta_lr
        self.min_weight = min_weight

    def update(self, weights: Mapping[str, float], scores: Mapping[str, float]) -> dict[str, float]:
        """Return updated blend weights from training-dataset alignment scores.

        Args:
            weights: Current normalized blend weights keyed by training dataset name.
            scores: Gradient-alignment scores keyed by training dataset name. Higher scores
                increase weights relative to lower scores.

        Returns:
            Updated normalized blend weights keyed by training dataset name.
        """
        if self.min_weight * len(weights) >= 1:
            raise ValueError(
                "min_weight is too large for the number of sources: "
                f"{self.min_weight} * {len(weights)} must be less than 1."
            )

        logits: dict[str, float] = {}
        for key, weight in weights.items():
            score = scores[key]
            # Non-log formula: raw_weight = weight * exp(meta_lr * score).
            # Use this exponentiated update instead of weight + meta_lr * score so dataset
            # probability weights stay positive and can be normalized by a simple sum.
            # This line stores log(raw_weight) so large scores are handled more stably.
            logits[key] = math.log(weight) + self.meta_lr * score

        max_logit = max(logits.values())
        # Move out of log space with the standard stable-softmax trick: subtract max_logit so the
        # largest exponent is exp(0), avoiding overflow. Subtracting the same constant from every
        # logit does not change the final normalized weights.
        unnormalized = {key: math.exp(logit - max_logit) for key, logit in logits.items()}
        total = sum(unnormalized.values())
        updated = {key: value / total for key, value in unnormalized.items()}
        if self.min_weight == 0:
            return updated

        # Reserve a floor for each source, then distribute the remaining mass according to the
        # normal exponentiated DoGE update. This keeps every source trainable while preserving the
        # relative preference from the alignment scores in the non-floor probability mass.
        remaining_weight = 1.0 - self.min_weight * len(updated)
        return {key: self.min_weight + remaining_weight * value for key, value in updated.items()}
