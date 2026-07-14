# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Validates and scores model compression solutions by evaluating puzzle solution candidates."""

# mypy: ignore-errors
import os
import re
from glob import glob
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist

from .granularity import resolve_granularity
from .tools.hydra_utils import register_hydra_resolvers
from .tools.logger import mprint

__all__ = ["launch_scoring", "resolve_scoring_output_dir", "resolve_scoring_paths"]


def extract_solution_id(filename):
    pattern = r"solution_(\d+)\.json"
    match = re.search(pattern, filename)

    if match:
        solution_id = match.group(1)
        return int(solution_id)
    else:
        mprint(f"Couldn't extract solutions_id from file {filename}")


def find_missing_solutions(solutions_df, validation_dir):
    all_solutions = np.arange(solutions_df.shape[0])

    benchmarked_solutions = list(glob(f"{validation_dir}/solution*.json"))
    benchmarked_solutions = [
        extract_solution_id(os.path.basename(s)) for s in benchmarked_solutions
    ]
    benchmarked_solutions = [s for s in benchmarked_solutions if s is not None]

    unbenchmarked_solutions = np.setdiff1d(all_solutions, benchmarked_solutions)
    return unbenchmarked_solutions.tolist()


def partition_for_node(items: list, num_nodes: int, node_index: int) -> list:
    """Keep only the items this node owns, interleaved by value.

    Interleaving (``value % num_nodes``) rather than contiguous slicing balances
    the heavier solutions (e.g. attention layers with more variants) across nodes.
    Works on solution ids, so it is stable across resumes and composes with
    ``skip_existing_solutions``.
    """
    if num_nodes <= 1:
        return items
    if not (0 <= node_index < num_nodes):
        raise ValueError(f"node_index {node_index} must be in [0, {num_nodes})")
    return [x for x in items if int(x) % num_nodes == node_index]


def resolve_scoring_paths(cfg: DictConfig) -> tuple[Path, Path]:
    """Resolve input/output paths for block or atomic subblock scoring."""

    granularity = resolve_granularity("scoring", cfg.scoring)
    solutions_path = cfg.scoring.get(f"{granularity}_solutions_path", None)
    if solutions_path is None:
        solutions_path = cfg.scoring.solutions_path
    output_dir = cfg.scoring.get(f"{granularity}_output_dir", None)
    if output_dir is None:
        output_dir = cfg.scoring.get("output_dir", None)
    solutions_path = Path(str(solutions_path))
    if output_dir is None:
        output_dir = solutions_path.with_name(f"{solutions_path.stem}--validation")
    return solutions_path, Path(str(output_dir))


def resolve_scoring_output_dir(cfg: DictConfig) -> str:
    """Output dir where ``solution_*.json`` are written — must match the writer's resolution.

    The AutoModel writer falls back to ``<solutions_path stem>--validation`` when
    ``scoring.output_dir`` is unset; resume detection has to use the same path.
    """
    return str(resolve_scoring_paths(cfg)[1])


def get_solutions_to_validate(cfg: DictConfig, num_nodes: int = 1, node_index: int = 0):
    _solutions_to_validate = cfg.scoring.solutions_to_validate
    if _solutions_to_validate is None:
        solutions_path, _ = resolve_scoring_paths(cfg)
        single_block_replacement_solutions = pd.read_json(solutions_path)
        if cfg.scoring.skip_existing_solutions:
            _solutions_to_validate = find_missing_solutions(
                single_block_replacement_solutions, resolve_scoring_output_dir(cfg)
            )
        else:
            _solutions_to_validate = np.arange(single_block_replacement_solutions.shape[0]).tolist()
    return partition_for_node(_solutions_to_validate, num_nodes, node_index)


def launch_scoring(cfg: DictConfig, num_nodes: int = 1, node_index: int = 0):
    solutions_path, output_dir = resolve_scoring_paths(cfg)
    cfg.scoring.solutions_path = str(solutions_path)
    cfg.scoring.output_dir = str(output_dir)
    cfg.scoring.solutions_to_validate = get_solutions_to_validate(cfg, num_nodes, node_index)
    mprint(
        f"Solutions to validate (node {node_index}/{num_nodes}): "
        f"{cfg.scoring.solutions_to_validate}"
    )
    backend = str(cfg.scoring.get("backend", "automodel") or "automodel")
    if backend != "automodel":
        raise ValueError("Replace-one scoring only supports scoring.backend=automodel.")

    from .plugins.automodel.solution_launch import launch_score_solutions_automodel

    launch_score_solutions_automodel(cfg, num_nodes=num_nodes, node_index=node_index)


@hydra.main("", version_base="1.3")
def main(cfg: DictConfig) -> None:
    cfg = hydra.utils.instantiate(cfg)
    mprint(cfg)
    dist.setup(timeout=cfg.nccl_timeout_minutes)
    try:
        launch_scoring(cfg)
    finally:
        dist.cleanup()


if __name__ == "__main__":
    register_hydra_resolvers()
    main()
