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

from collections.abc import Mapping

__all__ = ["DoGEWeightUpdater"]


class DoGEWeightUpdater:
    """Outer-loop updater for DoGE data-blend weights.

    Args:
        meta_lr: Learning rate for exponentiated source-weight updates.

    Outputs:
        ``update`` returns normalized source weights after applying the update.
    """

    def __init__(self, meta_lr: float) -> None:
        """Initialize the updater."""
        self.meta_lr = meta_lr

    def update(
        self, weights: Mapping[str, float], scores: Mapping[str, float]
    ) -> Mapping[str, float]:
        """Return updated source weights from source-to-target alignment scores.

        Args:
            weights: Current normalized source weights keyed by source name.
            scores: Source-to-target gradient-alignment scores keyed by source name.

        Returns:
            Updated normalized source weights keyed by source name.
        """
        raise NotImplementedError("DoGE weight updates are not implemented yet.")
