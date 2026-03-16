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

import modelopt.torch.puzzletron.mip.mip_and_realize_models as mip_and_realize_models
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.logger import mprint


def get_teacher_memory_from_subblock_stats(hydra_cfg) -> float:
    """Calculate teacher model memory from subblock_stats.json.

    Replicates the MIP solver's memory calculation logic:
    - Loads subblock_stats.json which contains memory measurements for all subblock configs
    - Finds the teacher FFN subblock (with full intermediate_size)
    - Finds the teacher Attention subblock (full attention, not no_op)
    - Calculates: non_block_memory + (ffn_memory + attention_memory) * num_layers

    This matches how the MIP solver computes total model memory via _get_block_stats().

    Args:
        hydra_cfg: Hydra configuration object

    Returns:
        Total teacher memory in MiB
    """
    puzzle_dir = Path(hydra_cfg.puzzle_dir)

    # Read config.json directly from the teacher model path
    teacher_dir = Path(hydra_cfg.teacher_dir)
    config_file = teacher_dir / "config.json"

    with open(config_file) as f:
        config_dict = json.load(f)

    num_layers = config_dict["num_hidden_layers"]
    teacher_ffn_intermediate = config_dict["intermediate_size"]
    teacher_num_kv_heads = config_dict["num_key_value_heads"]

    # Get the MIP configuration
    mip_subblock_args = hydra_cfg.mip.subblock_stats_args[0]
    batch_size = mip_subblock_args["batch_size"]
    weights_dtype = str(mip_subblock_args["weights_dtype"])
    activations_dtype = str(mip_subblock_args["activations_dtype"])
    kv_cache_dtype = str(mip_subblock_args["kv_cache_dtype"])

    # Load subblock_stats.json
    subblock_stats_path = puzzle_dir / "subblock_stats.json"
    if not subblock_stats_path.exists():
        raise FileNotFoundError(
            f"subblock_stats.json not found at {subblock_stats_path}. "
            "Please run the full pipeline first without --mip-only flag."
        )

    with open(subblock_stats_path) as f:
        subblock_stats_list = json.load(f)

    # Find the entry matching our MIP configuration and teacher's n_embd
    matching_stats = None
    for stats_entry in subblock_stats_list:
        args = stats_entry["args"]
        if (
            args["batch_size"] == batch_size
            and args["weights_dtype"] == weights_dtype
            and args["activations_dtype"] == activations_dtype
            and args["kv_cache_dtype"] == kv_cache_dtype
            and args.get("n_embd") == config_dict["hidden_size"]
        ):
            matching_stats = stats_entry
            break

    if matching_stats is None:
        raise ValueError(
            f"No subblock_stats entry found for batch_size={batch_size}, "
            f"dtypes=({weights_dtype}, {activations_dtype}, {kv_cache_dtype}), "
            f"n_embd={config_dict['hidden_size']}"
        )

    # Get non-block memory (embeddings, LM head, etc.)
    total_memory = matching_stats.get("non_block", {}).get("memory_mib", 0.0)

    # Find the teacher FFN and Attention subblocks
    # Note: Each subblock is EITHER attention OR ffn, not both
    # We need to find BOTH and add their memory together
    teacher_ffn_subblock = None
    teacher_attention_subblock = None

    for subblock in matching_stats.get("subblocks", []):
        subblock_class = subblock.get("subblock_config_class", "")
        subblock_config = subblock.get("subblock_config", {})

        # Check for FFN subblocks with teacher's intermediate_size
        if "FFN" in subblock_class:
            ffn_size = subblock_config.get("intermediate_size")
            if ffn_size == teacher_ffn_intermediate and not subblock_config.get("no_op", False):
                teacher_ffn_subblock = subblock

        # Check for Attention subblocks with teacher's num_key_value_heads
        elif "Attention" in subblock_class:
            kv_heads = subblock_config.get("num_key_value_heads")
            if kv_heads == teacher_num_kv_heads and not subblock_config.get("no_op", False):
                teacher_attention_subblock = subblock

    if teacher_ffn_subblock is None:
        raise ValueError(
            f"Could not find teacher FFN subblock with intermediate_size={teacher_ffn_intermediate}"
        )

    if teacher_attention_subblock is None:
        raise ValueError(
            f"Could not find teacher Attention subblock with num_key_value_heads={teacher_num_kv_heads}"
        )

    # Calculate total teacher memory: non_block + (ffn_memory + attention_memory) * num_layers
    per_layer_memory = teacher_ffn_subblock["memory_mib"] + teacher_attention_subblock["memory_mib"]
    total_memory += per_layer_memory * num_layers

    return total_memory


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

                mprint(
                    f"✓ Results: actual_memory={result['actual_memory_mib']:.1f} MiB, "
                    f"lm_loss={result['lm_loss']:.4f}"
                )

    # Write results to CSV (only on master rank)
    if dist.is_master():
        mprint("\n" + "=" * 80)
        mprint("MIP Sweep Complete - Writing Results")
        mprint("=" * 80)
        write_results_to_csv(all_results, output_csv)
        mprint(f"Completed {len(all_results)} sweep runs")
        mprint("=" * 80)
