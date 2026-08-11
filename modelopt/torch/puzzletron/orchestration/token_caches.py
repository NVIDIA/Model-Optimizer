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

"""Dependency-light fixed-token cache resolution shared by producers and consumers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["resolve_tokenize_caches"]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sample_count(value: Any, *, default: int = 0) -> int:
    count = int(default if value in (None, "") else value)
    if count < 0:
        raise ValueError(f"token cache sample counts must be non-negative, got {count}")
    return count


def _consumer_samples(config: Mapping[str, Any], section: str, *, default: int = 0) -> int:
    stage_config = _mapping(config.get(section))
    if stage_config.get("enabled") is False:
        return 0
    if "eval_samples" not in stage_config and default <= 0:
        return 0
    return _sample_count(stage_config.get("eval_samples"), default=default)


def resolve_tokenize_caches(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return explicit token caches or derive them from canonical campaign paths."""

    stage_config = _mapping(config.get("tokenize_data"))
    raw_caches = stage_config.get("caches") or ()
    if raw_caches:
        return [dict(cache) for cache in raw_caches]

    data_config = _mapping(config.get("data"))
    calibration = _mapping(data_config.get("calibration"))
    train_samples = _sample_count(calibration.get("num_samples"), default=32768)
    sequence_length = _sample_count(
        calibration.get("seq_len") or data_config.get("max_sample_length"),
        default=4096,
    )

    scoring_data = _mapping(data_config.get("replacement_scoring"))
    replacement_default = _sample_count(scoring_data.get("num_samples"), default=128)
    validation_samples = max(
        _consumer_samples(config, "replacement_scoring", default=replacement_default),
        _consumer_samples(config, "depth_importance"),
        _consumer_samples(config, "sort_sanity"),
        _consumer_samples(config, "width_sanity"),
    )

    configured_seed = _mapping(config.get("pruning")).get("shuffle_seed")
    train_seed = 444 if configured_seed is None else int(configured_seed)

    caches: list[dict[str, Any]] = []
    train_path = config.get("train_token_cache_path")
    if train_path:
        caches.append(
            {
                "output": str(train_path),
                "split": "train",
                "num_samples": train_samples,
                "seq_length": sequence_length,
                "shuffle_seed": train_seed,
            }
        )
    validation_path = config.get("validation_token_cache_path")
    if validation_path:
        caches.append(
            {
                "output": str(validation_path),
                "split": "validation",
                "num_samples": validation_samples,
                "seq_length": sequence_length,
                "shuffle_seed": train_seed + 1,
            }
        )
    return caches
