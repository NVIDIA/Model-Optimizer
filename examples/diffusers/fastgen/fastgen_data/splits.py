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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic train/validation membership for FastGen cache ordinals."""

from __future__ import annotations

import torch

__all__ = ["make_train_validation_indices"]


def _require_integer(name: str, value: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer; got {type(value).__name__}")
    return value


def make_train_validation_indices(
    num_samples: int,
    validation_count: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return disjoint ordered metadata ordinals using a local CPU generator."""
    num_samples = _require_integer("num_samples", num_samples)
    validation_count = _require_integer("validation_count", validation_count)
    seed = _require_integer("seed", seed)
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if not 1 <= validation_count < num_samples:
        raise ValueError("validation_count must be in [1, num_samples)")
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator, device="cpu").tolist()
    validation = sorted(permutation[:validation_count])
    train = sorted(permutation[validation_count:])
    return train, validation
