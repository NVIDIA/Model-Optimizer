# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Stable validation-artifact writer shared by AutoModel scoring paths."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from modelopt.torch.utils import json_dump

__all__ = ["scoring_result_matches", "write_results"]

_OPERATIONAL_SCORING_KEYS = frozenset(
    {"force_rescore", "skip_existing_solutions", "solutions_to_validate"}
)


def _resolved_scoring_args(args: Any) -> dict[str, Any]:
    if isinstance(args, DictConfig):
        value = OmegaConf.to_container(args, resolve=True)
    elif isinstance(args, Mapping):
        value = dict(args)
    else:
        value = vars(args)
    if not isinstance(value, dict):
        raise TypeError(f"scoring arguments must resolve to a mapping, got {type(value).__name__}")
    return {key: item for key, item in value.items() if key not in _OPERATIONAL_SCORING_KEYS}


def scoring_result_matches(
    path: str | Path,
    args: Any,
    *,
    expected_payload: Mapping[str, Any] | None = None,
) -> bool:
    """Return whether an existing score matches its config and expected identity fields."""

    try:
        result = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return False
    recorded = result.get("args") if isinstance(result, dict) else None
    if not isinstance(recorded, dict):
        return False
    if _resolved_scoring_args(recorded) != _resolved_scoring_args(args):
        return False
    return expected_payload is None or all(
        result.get(key) == value for key, value in expected_payload.items()
    )


def write_results(
    output_dir: str | Path,
    result_name: str,
    args: DictConfig,
    payload: dict[str, Any],
) -> None:
    output_path = Path(output_dir) / f"{result_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        **payload,
        "args": OmegaConf.to_container(args, resolve=True)
        if isinstance(args, DictConfig)
        else args.__dict__,
    }
    json_dump(results, output_path)
