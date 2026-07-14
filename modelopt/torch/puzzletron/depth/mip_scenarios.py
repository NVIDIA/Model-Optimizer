# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Solve, realize, and validate every prefix of an iterative depth trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import modelopt.torch.utils.distributed as dist
from modelopt.torch.utils import json_dump

from ..mip.run_puzzle import run_puzzle
from ..tools.validate_puzzle_with_multi_replacements import validate_puzzle_solutions

__all__ = ["run_depth_mip_scenarios"]


_DEPTH_REPORT_ADDITIONAL_COSTS = (
    "stats.num_params",
    "stats.active_params",
    "stats.memory_mib",
    "stats.weight_memory_mib",
    "stats.kv_cache_memory_mib",
    "stats.kv_cache_bytes_per_token",
    "stats.state_cache_bytes_per_sequence",
    "stats.runtime_ms",
    "stats.prefill_runtime_ms",
    "stats.decode_runtime_ms",
    "stats.decode_runtime_ms_per_token",
    "stats.prefill_flops",
    "stats.decode_flops",
    "stats.num_kv_heads",
    "stats.num_query_heads",
    "stats.num_experts",
    "stats.top_k",
    "stats.has_attention",
    "stats.has_mamba",
    "stats.has_ffn",
    "stats.has_moe",
    "stats.not_no_op",
)


def _broadcast(value):
    values = [value]
    torch.distributed.broadcast_object_list(values, src=0)
    return values[0]


def run_depth_mip_scenarios(cfg: DictConfig) -> list[str]:
    """Run all scenario MIPs, then realize/evaluate the winners in one parent sweep."""
    trajectory_path = Path(cfg.mip.depth_trajectory_path)
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"depth trajectory does not exist: {trajectory_path}")
    trajectory = json.loads(trajectory_path.read_text())
    scenarios = list(trajectory.get("scenarios") or [])
    expected = int(cfg.mip.get("depth_scenario_count", 11))
    if len(scenarios) != expected:
        raise RuntimeError(f"expected {expected} depth scenarios, found {len(scenarios)}")

    tournament_root = Path(cfg.mip.output_path) / "depth_tournament"
    solution_paths: list[str] | None = None
    combined_path = tournament_root / "solutions.json"
    scenario_results_path = tournament_root / "scenario_results.json"
    if dist.is_master():
        solution_paths = []
        combined = []
        scenario_results = []
        for index, scenario in enumerate(scenarios):
            scenario_cfg = OmegaConf.create(OmegaConf.to_container(cfg.mip, resolve=True))
            # ``run_puzzle``'s established config contract uses ``Path``
            # objects for path fields and composes output directories with
            # ``/``.  Keep that type in derived depth scenarios.
            scenario_cfg.output_path = tournament_root / f"depth_{index:02d}"
            scenario_cfg.forced_removals = list(scenario.get("removals") or [])
            if not scenario_cfg.get("report_additional_costs"):
                scenario_cfg.report_additional_costs = list(
                    _DEPTH_REPORT_ADDITIONAL_COSTS
                )
            paths = run_puzzle(args=scenario_cfg)
            if len(paths) != 1:
                raise RuntimeError(
                    f"depth scenario {index} produced {len(paths)} solution files; expected one"
                )
            path = Path(paths[0])
            raw = json.loads(path.read_text())
            if len(raw) > 1:
                raise RuntimeError(
                    f"depth scenario {index} produced {len(raw)} solutions; expected one"
                )
            solution_paths.append(str(path))
            result = {
                **scenario,
                "index": index,
                "status": "feasible" if raw else "infeasible",
                "mip_solution_path": str(path),
            }
            scenario_results.append(result)
            if not raw:
                continue
            solution = raw[0]
            solution["depth_scenario"] = result
            combined.append(solution)
        tournament_root.mkdir(parents=True, exist_ok=True)
        json_dump(combined, combined_path)
        json_dump(scenario_results, scenario_results_path)
    solution_paths = _broadcast(solution_paths)
    dist.barrier()

    if not bool(cfg.get("skip_realize_model", False)) and combined_path.is_file() and json.loads(
        combined_path.read_text()
    ):
        cfg.realize_model.solutions_path = combined_path
        cfg.realize_model.solutions_to_validate = list(
            range(len(json.loads(combined_path.read_text())))
        )
        cfg.realize_model.output_dir = str(tournament_root / "exact_lm_evaluation")
        validate_puzzle_solutions(args=cfg.realize_model, hydra_cfg=cfg)
        dist.barrier()
    return [*solution_paths, str(combined_path), str(scenario_results_path)]
