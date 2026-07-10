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

    Outputs:
        ``update`` returns normalized blend weights after applying the update.
    """

    def __init__(self, meta_lr: float) -> None:
        """Initialize the updater."""
        self.meta_lr = meta_lr

    def update(self, weights: Mapping[str, float], scores: Mapping[str, float]) -> dict[str, float]:
        """Return updated blend weights from training-dataset alignment scores.

        Args:
            weights: Current normalized blend weights keyed by training dataset name.
            scores: Gradient-alignment scores keyed by training dataset name. Higher scores
                increase weights relative to lower scores.

        Returns:
            Updated normalized blend weights keyed by training dataset name.
        """
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
        return {key: value / total for key, value in unnormalized.items()}
