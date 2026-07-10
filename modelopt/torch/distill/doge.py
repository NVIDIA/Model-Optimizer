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
    for weight_value, path in zip(data_paths[::2], data_paths[1::2], strict=True):
        weight = float(weight_value)
        if weight <= 0:
            raise ValueError(f"blend weights must be positive, got {weight_value!r}")
        blend_weights[path] = blend_weights.get(path, 0.0) + weight

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

    def update(
        self, weights: Mapping[str, float], scores: Mapping[str, float]
    ) -> Mapping[str, float]:
        """Return updated blend weights from training-dataset alignment scores.

        Args:
            weights: Current normalized blend weights keyed by training dataset name.
            scores: Gradient-alignment scores keyed by training dataset name.

        Returns:
            Updated normalized blend weights keyed by training dataset name.
        """
        raise NotImplementedError("DoGE weight updates are not implemented yet.")
