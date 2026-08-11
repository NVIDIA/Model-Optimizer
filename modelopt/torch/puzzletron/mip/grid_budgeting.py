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

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from ..identity import stable_hash
from ..tools.logger import mprint
from .mip_and_realize_models import launch_mip_and_realize_model

__all__ = ["build_grid_budget_entries", "run_grid_budgeted_mip"]


def _as_plain(value: Any) -> Any:
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _list(value: Any) -> list[Any]:
    value = _as_plain(value)
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _named_constraints(
    name: str,
    values: list[Any],
    *,
    target: str,
    kind: str = "mip",
) -> list[dict[str, Any]]:
    entries = []
    for value in values:
        entries.append(
            {
                "name": f"{name}_{str(value).replace('/', '_')}",
                f"{kind}_constraints": {target: value},
            }
        )
    return entries


def build_grid_budget_entries(grid_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a grid-budget config into concrete MIP constraint entries.

    Explicit entries are passed through, for example::

        grid_budgeting:
          enabled: true
          entries:
            - name: depth_24
              mip_constraints: {"stats.not_no_op": 24}
            - name: params_8b
              human_constraints: {"num_params": "8B"}

    Convenience fields map to the aggregate stats already computed by
    ``run_puzzle.py``. Unknown future axes can be expressed through ``entries``
    without changing this function.
    """
    grid_cfg = dict(grid_cfg or {})
    explicit_entries = [
        dict(item) for item in _list(grid_cfg.get("entries")) if isinstance(item, dict)
    ]
    entries: list[dict[str, Any]] = explicit_entries

    entries.extend(
        _named_constraints(
            "not_no_op",
            _list(grid_cfg.get("not_no_op_counts") or grid_cfg.get("depth_budgets")),
            target="stats.not_no_op",
        )
    )
    entries.extend(
        _named_constraints(
            "no_op",
            _list(grid_cfg.get("no_op_counts")),
            target="stats.no_op_count",
        )
    )
    entries.extend(
        _named_constraints(
            "experts",
            _list(grid_cfg.get("expert_budgets")),
            target="stats.num_experts",
        )
    )
    entries.extend(_named_constraints("top_k", _list(grid_cfg.get("top_k")), target="stats.top_k"))
    entries.extend(
        _named_constraints(
            "mamba_heads",
            _list(grid_cfg.get("mamba_heads")),
            target="stats.num_mamba_heads",
        )
    )
    entries.extend(
        _named_constraints(
            "params",
            _list(grid_cfg.get("params") or grid_cfg.get("param_budgets")),
            target="num_params",
            kind="human",
        )
    )
    entries.extend(
        _named_constraints(
            "params_ratio",
            _list(grid_cfg.get("param_ratios") or grid_cfg.get("target_params_ratio")),
            target="num_params_ratio",
            kind="human",
        )
    )
    entries.extend(
        _named_constraints(
            "latency_ratio",
            _list(grid_cfg.get("latency_ratios") or grid_cfg.get("target_latency_ratio")),
            target="target_latency_ratio",
            kind="human",
        )
    )
    entries.extend(
        _named_constraints(
            "memory",
            _list(grid_cfg.get("memory_mib") or grid_cfg.get("target_memory_mib")),
            target="target_memory",
            kind="human",
        )
    )

    dimensions = dict(grid_cfg.get("dimensions") or {})
    if dimensions:
        keys = sorted(dimensions)
        for values in product(*[_list(dimensions[key]) for key in keys]):
            mip_constraints = {}
            human_constraints = {}
            parts = []
            for key, value in zip(keys, values):
                parts.append(f"{key}_{value}")
                if key.startswith("human."):
                    human_constraints[key.removeprefix("human.")] = value
                else:
                    mip_constraints[key.removeprefix("mip.")] = value
            entry = {"name": "-".join(str(part).replace("/", "_") for part in parts)}
            if mip_constraints:
                entry["mip_constraints"] = mip_constraints
            if human_constraints:
                entry["human_constraints"] = human_constraints
            entries.append(entry)

    if not entries:
        entries.append({"name": "default"})
    return entries


def _set_mip_constraints(hydra_cfg: Any, entry: dict[str, Any], output_root: Path) -> None:
    if "mip_constraints" in entry:
        hydra_cfg.mip.mip_constraints = dict(entry["mip_constraints"])
        hydra_cfg.mip.human_constraints = None
    if "human_constraints" in entry:
        hydra_cfg.mip.human_constraints = dict(entry["human_constraints"])
        hydra_cfg.mip.mip_constraints = None
    if "subblock_stats_args" in entry:
        hydra_cfg.mip.subblock_stats_args = dict(entry["subblock_stats_args"])

    name = entry.get("name") or stable_hash(entry, prefix="grid", length=10)
    if hasattr(hydra_cfg.mip, "output_path") and hydra_cfg.mip.output_path is not None:
        hydra_cfg.mip.output_path = output_root / name


def run_grid_budgeted_mip(hydra_cfg: Any, grid_cfg: dict[str, Any]) -> list[str]:
    entries = build_grid_budget_entries(grid_cfg)
    original_mip = deepcopy(hydra_cfg.mip)
    output_root = Path(getattr(hydra_cfg.mip, "output_path", "mip_grid"))
    all_solution_paths: list[str] = []
    mprint(f"Running grid-budgeted MIP over {len(entries)} budget point(s)")
    try:
        for entry in entries:
            hydra_cfg.mip = deepcopy(original_mip)
            _set_mip_constraints(hydra_cfg, entry, output_root)
            mprint(f"Grid MIP budget {entry.get('name', '<unnamed>')}: {entry}")
            all_solution_paths.extend(str(path) for path in launch_mip_and_realize_model(hydra_cfg))
    finally:
        hydra_cfg.mip = original_mip
    return all_solution_paths
