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

import math
from collections.abc import Mapping, Sequence

__all__ = ["DoGEWeightUpdater", "normalize_data_path_weights"]


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
