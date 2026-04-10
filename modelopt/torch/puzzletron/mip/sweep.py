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

"""MIP sweep functionality for exploring multiple memory compression rates."""

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf
from transformers import PretrainedConfig

import modelopt.torch.utils.distributed as dist

from ..anymodel import models  # noqa: F401 — register ModelDescriptorFactory entries
from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..tools.checkpoint_utils_hf import load_model_config
from ..tools.logger import mprint
from . import mip_and_realize_models
from .run_puzzle import _get_block_stats, filter_subblock_stats_by_args

__all__ = [
    "get_teacher_memory_from_subblock_stats",
    "get_teacher_num_params_from_subblock_stats",
    "extract_solution_results",
    "write_results_to_csv",
    "run_mip_sweep",
]


def _load_teacher_subblock_stats(hydra_cfg: DictConfig) -> tuple[dict[str, Any], PretrainedConfig]:
    """Load filtered subblock_stats and teacher ``model_config`` for the current MIP scenario."""
    puzzle_dir = Path(hydra_cfg.puzzle_dir)
    teacher_dir = Path(hydra_cfg.teacher_dir)

    descriptor = ModelDescriptorFactory.get(hydra_cfg.descriptor)
    trust_remote_code = descriptor.requires_trust_remote_code()
    model_config = load_model_config(teacher_dir, trust_remote_code=trust_remote_code)
    lm_config = descriptor.get_language_model_config(model_config)
    hidden_size = lm_config.hidden_size

    mip_subblock_args = hydra_cfg.mip.subblock_stats_args[0]
    subblock_stats_args = OmegaConf.to_container(mip_subblock_args, resolve=True)
    # Subblock_stats.json can list multiple runs that share batch/dtypes but differ by hidden size;
    # filter_subblock_stats_by_args needs n_embd so exactly one row matches the teacher.
    subblock_stats_args = {**subblock_stats_args, "n_embd": hidden_size}

    batch_size = subblock_stats_args["batch_size"]
    weights_dtype = str(subblock_stats_args["weights_dtype"])
    activations_dtype = str(subblock_stats_args["activations_dtype"])
    kv_cache_dtype = str(subblock_stats_args["kv_cache_dtype"])

    subblock_stats_path = puzzle_dir / "subblock_stats.json"
    if not subblock_stats_path.exists():
        raise FileNotFoundError(
            f"subblock_stats.json not found at {subblock_stats_path}. "
            "Please run the full pipeline first without --mip-only flag."
        )

    with open(subblock_stats_path) as f:
        subblock_stats_list = json.load(f)

    try:
        subblock_stats = filter_subblock_stats_by_args(subblock_stats_list, subblock_stats_args)
    except AssertionError as e:
        raise ValueError(
            f"No unique subblock_stats entry for batch_size={batch_size}, "
            f"dtypes=({weights_dtype}, {activations_dtype}, {kv_cache_dtype}), "
            f"n_embd={hidden_size}"
        ) from e

    return subblock_stats, model_config


def get_teacher_memory_from_subblock_stats(hydra_cfg: DictConfig) -> float:
    """Calculate teacher model memory from subblock_stats.json.

    Sums ``non_block`` and per-layer ``_get_block_stats(subblock_stats, block_config, layer_index)``
    over ``model_config.block_configs``, matching :func:`run_puzzle._get_block_stats`.

    Args:
        hydra_cfg: Hydra configuration object

    Returns:
        Total teacher memory in MiB
    """
    subblock_stats, model_config = _load_teacher_subblock_stats(hydra_cfg)

    total_memory = subblock_stats.get("non_block", {}).get("memory_mib", 0.0)

    for layer_idx, block_config in enumerate(model_config.block_configs):
        block_stats = _get_block_stats(subblock_stats, block_config, layer_idx)
        total_memory += block_stats["memory_mib"]

    return total_memory


def get_teacher_num_params_from_subblock_stats(hydra_cfg: DictConfig) -> int:
    """Calculate total teacher parameter count from subblock_stats.json.

    Sums ``non_block`` and per-layer ``_get_block_stats(...)["num_params"]`` over
    ``model_config.block_configs``, matching :func:`run_puzzle._get_block_stats`.

    Args:
        hydra_cfg: Hydra configuration object

    Returns:
        Total teacher parameter count (same units as subblock_stats JSON).
    """
    subblock_stats, model_config = _load_teacher_subblock_stats(hydra_cfg)

    total_params = subblock_stats.get("non_block", {}).get("num_params", 0)

    for layer_idx, block_config in enumerate(model_config.block_configs):
        block_stats = _get_block_stats(subblock_stats, block_config, layer_idx)
        total_params += block_stats["num_params"]

    return int(total_params)


def extract_solution_results(
    solution_path: Path,
    target_memory_mib: float,
    teacher_memory_mib: float,
    compression_rate: float,
) -> dict:
    """Extract results from a completed MIP solution.

    Args:
        solution_path: Path to the solutions.json file (not the directory!)
        target_memory_mib: Target memory constraint used for MIP
        teacher_memory_mib: Teacher model memory in MiB
        compression_rate: Compression rate applied

    Returns:
        Dictionary containing extracted metrics
    """
    result = {
        "compression_rate": compression_rate,
        "target_memory_mib": target_memory_mib,
        "teacher_memory_mib": teacher_memory_mib,
    }

    # solution_path is the path to solutions.json file, get parent directory
    solution_dir = solution_path.parent

    # Load solutions.json for actual memory and parameters
    solutions_file = solution_dir / "solutions.json"
    with open(solutions_file) as f:
        solutions_data = json.load(f)
        solution = solutions_data[0]  # First solution
        total_costs = solution.get("total_costs", {})
        result["actual_memory_mib"] = total_costs.get("stats.memory_mib", None)
        result["num_params"] = total_costs.get("stats.num_params", None)

    # Load solution_0.json for accuracy metrics
    validation_dir = solution_dir / "solutions--validation"
    # TODO: There could be multiple solutions, but we only need the first one. Is it the best solution?
    solution_0_file = validation_dir / "solution_0.json"

    with open(solution_0_file) as f:
        validation_data = json.load(f)
        result["lm_loss"] = validation_data.get("lm_loss", {}).get("avg", None)
        result["token_accuracy_top_1"] = validation_data.get("token_accuracy_top_1", {}).get(
            "avg", None
        )
        result["token_accuracy_top_5"] = validation_data.get("token_accuracy_top_5", {}).get(
            "avg", None
        )
        result["token_accuracy_top_10"] = validation_data.get("token_accuracy_top_10", {}).get(
            "avg", None
        )

    return result


def write_results_to_csv(results: list, output_csv: str):
    """Write sweep results to CSV file.

    Args:
        results: List of result dictionaries
        output_csv: Path to output CSV file
    """
    import csv

    # Define CSV columns in desired order
    columns = [
        "compression_rate",
        "target_memory_mib",
        "actual_memory_mib",
        "teacher_memory_mib",
        "num_params",
        "lm_loss",
        "token_accuracy_top_1",
        "token_accuracy_top_5",
        "token_accuracy_top_10",
    ]

    # Write CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    mprint(f"Results written to: {output_path}")


def run_mip_sweep(hydra_cfg):
    """Run MIP for multiple memory compression rates and generate CSV with results.

    This function is called when mip.sweep.enabled is True in the config.

    Args:
        hydra_cfg: Hydra configuration object with mip.sweep settings
    """
    mprint("=" * 80)
    mprint("MIP Sweep Mode Enabled")
    mprint("=" * 80)

    # Get sweep configuration
    sweep_cfg = hydra_cfg.mip.sweep
    compression_rates = sweep_cfg.memory_compression_rates
    output_csv = sweep_cfg.output_csv
    puzzle_dir = Path(hydra_cfg.puzzle_dir)

    mprint(f"Compression rates: {compression_rates}")
    mprint(f"Output CSV: {output_csv}")
    mprint(f"Puzzle directory: {puzzle_dir}")

    # Calculate teacher memory from subblock_stats
    teacher_memory = get_teacher_memory_from_subblock_stats(hydra_cfg)
    mprint(
        f"Teacher memory (from subblock_stats): {teacher_memory:.1f} MiB ({teacher_memory / 1024:.1f} GiB)"
    )

    # Collect results
    all_results = []

    # Run MIP for each compression rate
    for compression_rate in compression_rates:
        target_memory_mib = teacher_memory * compression_rate
        mprint("\n" + "=" * 80)
        mprint(
            f"Running MIP for compression_rate={compression_rate:.2f} "
            f"(target={target_memory_mib:.1f} MiB = {target_memory_mib / 1024:.1f} GiB)"
        )
        mprint("=" * 80)

        # Modify config dynamically
        hydra_cfg.mip.human_constraints.target_memory = target_memory_mib

        # Run MIP and realize models (reuse existing distributed logic!)
        solution_paths = mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg)

        # Extract results (only on master rank)
        if dist.is_master():
            for solution_path in solution_paths:
                result = extract_solution_results(
                    solution_path=Path(solution_path),
                    target_memory_mib=target_memory_mib,
                    teacher_memory_mib=teacher_memory,
                    compression_rate=compression_rate,
                )
                all_results.append(result)

                mem = (
                    f"{result['actual_memory_mib']:.1f}"
                    if result["actual_memory_mib"] is not None
                    else "N/A"
                )
                loss = f"{result['lm_loss']:.4f}" if result["lm_loss"] is not None else "N/A"
                mprint(f"✓ Results: actual_memory={mem} MiB, lm_loss={loss}")

    # Write results to CSV (only on master rank)
    if dist.is_master():
        mprint("\n" + "=" * 80)
        mprint("MIP Sweep Complete - Writing Results")
        mprint("=" * 80)
        write_results_to_csv(all_results, output_csv)
        mprint(f"Completed {len(all_results)} sweep runs")
        mprint("=" * 80)
