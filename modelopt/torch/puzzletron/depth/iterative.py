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

"""Recomputed iterative attention/Mamba/FFN sublayer removal with one resident model."""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

import modelopt.torch.utils.distributed as dist

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..block_config import BlockConfig, maybe_cast_block_configs
from ..granularity import Granularity, resolve_granularity
from ..identity import canonicalize, stable_hash
from ..plugins.automodel.config import build_solution_recipe_config, solution_scoring_params
from ..plugins.automodel.launch import _free_scoring_memory
from ..plugins.automodel.patch import apply_patch
from ..plugins.automodel.solution_launch import (
    _extract_teacher_targets,
    _run_recipe,
    _score_candidate,
    _solution_prune_target,
)
from ..plugins.automodel.teacher_cache import TeacherTargetCache
from ..tools.checkpoint_utils import load_model_config
from ..tools.hydra_utils import clone_hydra_config
from ..tools.validation_utils import write_results
from .schema import DepthScenario, SublayerRemoval

__all__ = ["launch_iterative_depth_automodel"]


_REMOVABLE_KINDS = ("attention", "mamba", "ffn", "moe")
_REVISION = "puzzletron-iterative-depth-v1"


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _depth_scoring_config(hydra_cfg: DictConfig) -> DictConfig:
    # Puzzletron's Hydra loader deliberately enables ``allow_objects`` before
    # resolving registered mixins and dataset callables.  Carry that contract
    # through derived configs instead of rejecting those resolved objects while
    # cloning an otherwise valid campaign configuration.
    cfg = clone_hydra_config(hydra_cfg)
    depth = clone_hydra_config(cfg.get("depth", {}))
    scoring = clone_hydra_config(cfg.scoring)
    depth_automodel = clone_hydra_config(depth.get("automodel", {}))
    # Inherit scoring backend/metric knobs while allowing depth RPC to own an
    # independent model mesh through ``depth.automodel.parallel``.
    scoring.automodel = OmegaConf.merge(scoring.get("automodel", {}), depth_automodel)
    for key in (
        "eval_samples",
        "micro_batch_size",
        "block_size",
        "seed",
        "shuffle_seed",
        "varlen",
        "dataset_path",
        "realized_dataset_cache_dir",
        "val_dataset_name",
        "data_column",
        "load_dataset_fn",
    ):
        if key in depth:
            scoring[key] = depth[key]
    source = depth.get("source_checkpoint_dir", None) or str(
        Path(cfg.puzzle_dir) / "ckpts" / "elastic_sorted_teacher"
    )
    scoring.teacher_dir = str(source)
    scoring.source_checkpoint_dir = str(source)
    scoring.target_teacher_dir = str(source)
    scoring.output_dir = str(depth.get("output_dir", Path(cfg.puzzle_dir) / "depth" / "iterative"))
    cfg.scoring = scoring
    return cfg


def _available_removals(
    blocks: list[BlockConfig], *, granularity: Granularity = "subblock"
) -> list[SublayerRemoval]:
    if granularity == "block":
        return [
            SublayerRemoval(layer_idx=layer_idx, kind="block")
            for layer_idx, block in enumerate(blocks)
            if any(not subblock.no_op for subblock in block.subblock_configs)
        ]
    return [
        SublayerRemoval(layer_idx=layer_idx, kind=kind)
        for layer_idx, block in enumerate(blocks)
        for kind in _REMOVABLE_KINDS
        if (subblock := block.get_subblock(kind)) is not None and not subblock.no_op
    ]


def _child_blocks(
    teacher_blocks: list[BlockConfig],
    removals: tuple[SublayerRemoval, ...],
) -> dict[int, BlockConfig]:
    children: dict[int, BlockConfig] = {}
    for removal in removals:
        child = children.get(removal.layer_idx, teacher_blocks[removal.layer_idx])
        if removal.kind == "block":
            child = BlockConfig(
                subblock_configs=tuple(
                    dataclasses.replace(subblock, no_op=True) for subblock in child.subblock_configs
                )
            )
        else:
            subblock = child.require_subblock(removal.kind)
            child = child.with_subblock(dataclasses.replace(subblock, no_op=True))
        children[removal.layer_idx] = child
    return children


def _prune_targets(
    teacher_blocks: list[BlockConfig],
    removals: tuple[SublayerRemoval, ...],
    *,
    num_query_heads: int,
    head_dim: int,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for layer_idx, child in sorted(_child_blocks(teacher_blocks, removals).items()):
        replacement = {
            "parent_layer_indices": [layer_idx],
            "child_block_configs": [child],
        }
        target = _solution_prune_target(
            replacement,
            teacher_blocks,
            num_query_heads,
            head_dim,
        )
        if target is not None:
            targets.append(target)
    return targets


def _metric(path: Path, name: str = "lm_loss") -> float:
    raw = json.loads(path.read_text())
    value = raw[name]
    if isinstance(value, dict):
        value = value.get("avg")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite depth metric {name}={result} in {path}")
    return result


def _load_selected(path: Path) -> list[SublayerRemoval]:
    if not path.is_file():
        return []
    raw = json.loads(path.read_text())
    return [SublayerRemoval.model_validate(item) for item in raw.get("selected", [])]


def launch_iterative_depth_automodel(hydra_cfg: DictConfig) -> dict[str, Any]:
    """Create the full removal trajectory, resuming at candidate granularity."""
    apply_patch()
    cfg = _depth_scoring_config(hydra_cfg)
    depth_cfg = cfg.get("depth", {})
    granularity = resolve_granularity("depth", depth_cfg)
    scoring = cfg.scoring
    output_dir = Path(scoring.output_dir)
    trajectory_path = output_dir / "trajectory.json"
    source = Path(scoring.source_checkpoint_dir)
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"iterative-depth source checkpoint is incomplete: {source}")

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    model_config = load_model_config(
        source,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    teacher_blocks = list(maybe_cast_block_configs(model_config.block_configs))
    lm = descriptor.get_language_model_config(model_config)
    num_query_heads = int(lm.num_attention_heads)
    head_dim = int(getattr(lm, "head_dim", None) or lm.hidden_size // num_query_heads)
    available = _available_removals(teacher_blocks, granularity=granularity)
    expected = depth_cfg.get("expected_initial_sublayers", None)
    if expected is not None and len(available) != int(expected):
        raise RuntimeError(
            f"expected {expected} removable sublayers, descriptor exposed {len(available)}"
        )
    max_removals = min(int(depth_cfg.get("max_removals", 10)), len(available))
    selected = _load_selected(trajectory_path)
    if len(selected) > max_removals:
        raise RuntimeError(f"trajectory already has too many removals: {len(selected)}")

    params = solution_scoring_params(cfg)
    recipe = _run_recipe(
        build_solution_recipe_config(cfg, source),
        scoring,
        params["eval_iters"],
        params["use_puzzletron_dataloader"],
        params["data_cfg"],
    )
    cache = TeacherTargetCache(device=params["teacher_cache_device"])
    try:
        teacher_scores = _extract_teacher_targets(recipe, cache, params)
        if teacher_scores is not None and bool(getattr(recipe, "_puzzletron_output_writer", False)):
            write_results(output_dir, "teacher", scoring, teacher_scores)
        dist.barrier()

        for iteration in range(len(selected), max_removals):
            prefix = tuple(selected)
            iteration_dir = output_dir / f"iteration_{iteration:02d}"
            baseline_path = iteration_dir / "baseline.json"
            if not baseline_path.is_file():
                _score_candidate(
                    recipe,
                    cache,
                    params,
                    iteration_dir,
                    scoring,
                    name="baseline",
                    payload={"removals": [item.model_dump() for item in prefix]},
                    prune_target=_prune_targets(
                        teacher_blocks,
                        prefix,
                        num_query_heads=num_query_heads,
                        head_dim=head_dim,
                    ),
                )
                dist.barrier()
            baseline = _metric(baseline_path)
            selected_keys = {(item.layer_idx, item.kind) for item in selected}
            remaining = [
                item for item in available if (item.layer_idx, item.kind) not in selected_keys
            ]
            expected_remaining = len(available) - iteration
            if len(remaining) != expected_remaining:
                raise RuntimeError(
                    f"depth iteration {iteration} has {len(remaining)} candidates; "
                    f"expected {expected_remaining}"
                )

            for candidate in remaining:
                name = f"candidate_layer_{candidate.layer_idx:03d}_{candidate.kind}"
                result_path = iteration_dir / f"{name}.json"
                if result_path.is_file():
                    continue
                trial = (*prefix, candidate)
                _score_candidate(
                    recipe,
                    cache,
                    params,
                    iteration_dir,
                    scoring,
                    name=name,
                    payload={
                        "iteration": iteration,
                        "candidate": candidate.model_dump(),
                        "prefix": [item.model_dump() for item in prefix],
                        "baseline_lm_loss": baseline,
                    },
                    prune_target=_prune_targets(
                        teacher_blocks,
                        trial,
                        num_query_heads=num_query_heads,
                        head_dim=head_dim,
                    ),
                )
                dist.barrier()

            choice_payload = None
            if dist.is_master():
                ranked = []
                for candidate in remaining:
                    name = f"candidate_layer_{candidate.layer_idx:03d}_{candidate.kind}"
                    result_path = iteration_dir / f"{name}.json"
                    value = _metric(result_path)
                    ranked.append(
                        {
                            "candidate": candidate.model_dump(),
                            "lm_loss": value,
                            "delta_lm_loss": value - baseline,
                            "result_path": str(result_path),
                        }
                    )
                ranked.sort(
                    key=lambda row: (
                        row["delta_lm_loss"],
                        row["candidate"]["layer_idx"],
                        row["candidate"]["kind"],
                    )
                )
                choice_payload = ranked[0]
                _atomic_json(iteration_dir / "ranking.json", {"baseline": baseline, "rows": ranked})
            values = [choice_payload]
            torch.distributed.broadcast_object_list(values, src=0)
            chosen = SublayerRemoval.model_validate(values[0]["candidate"])
            selected.append(chosen)

            if dist.is_master():
                scenarios = []
                data_identity = stable_hash(
                    {
                        key: scoring.get(key, None)
                        for key in (
                            "dataset_path",
                            "eval_samples",
                            "block_size",
                            "seed",
                            "shuffle_seed",
                        )
                    },
                    prefix="depth_data",
                )
                parent_identity = stable_hash(
                    json.loads((source / "config.json").read_text()),
                    prefix="depth_parent",
                )
                for length in range(len(selected) + 1):
                    scenario = DepthScenario(
                        parent_checkpoint_identity=parent_identity,
                        hidden_width=int(lm.hidden_size),
                        removals=tuple(selected[:length]),
                        data_identity=data_identity,
                        evaluator_revision=_REVISION,
                        granularity=granularity,
                    )
                    scenarios.append(
                        {
                            **scenario.model_dump(mode="python"),
                            "scenario_id": scenario.scenario_id,
                        }
                    )
                _atomic_json(
                    trajectory_path,
                    {
                        "version": 1,
                        "granularity": granularity,
                        "status": "complete" if len(selected) == max_removals else "running",
                        "source_checkpoint_dir": str(source),
                        "available_count": len(available),
                        "max_removals": max_removals,
                        "selected": [item.model_dump() for item in selected],
                        "scenarios": scenarios,
                    },
                )
            dist.barrier()
    finally:
        recipe.teardown_capture()
        _free_scoring_memory(recipe)

    return {
        "trajectory_path": str(trajectory_path),
        "selected": [item.model_dump() for item in selected],
        "scenario_count": len(selected) + 1,
    }
