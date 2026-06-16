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

"""
Main script for running the puzzletron algorithm on large language models (based on Puzzle paper https://arxiv.org/abs/2411.19146).

This script provides three modes:
1. Default mode: Runs the full puzzletron pipeline
2. MIP-only mode: Runs only the MIP search and realize models phase
3. MIP sweep mode: Runs MIP for multiple memory compression rates (enabled via config)

Usage:
    # Full puzzletron pipeline
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml

    # Only MIP search and realize models phase
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only

    # MIP sweep mode (set mip.sweep.enabled: true in config)
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only
"""

import argparse
from datetime import timedelta
from pathlib import Path

import hydra

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron as mtpz
import modelopt.torch.utils.distributed as dist

# Stages that can be run in isolation via ``--stage``. ``bypass`` and ``scoring``
# additionally support multi-node work splitting via ``--nodes``/``--idx``.
STAGES = ("full", "bypass", "build_library", "scoring", "mip")
MULTI_NODE_STAGES = ("bypass", "scoring")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compress large language models using the Puzzletron algorithm (based on Puzzle paper https://arxiv.org/abs/2411.19146)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the main config YAML file (e.g., ./configs/llama_3.2_1B_pruneffn_memory.yaml)",
    )
    parser.add_argument(
        "--mip-only",
        action="store_true",
        help="Deprecated alias for --stage mip.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="full",
        choices=STAGES,
        help=(
            "Which pipeline stage to run. 'full' (default) runs everything end to end. "
            "'bypass'/'build_library'/'scoring'/'mip' run a single stage and assume the "
            "prior stages already completed (their outputs live in puzzle_dir). "
            "'bypass' and 'scoring' can be split across nodes with --nodes/--idx."
        ),
    )
    parser.add_argument(
        "--nodes",
        type=int,
        default=1,
        help="Total number of nodes splitting a multi-node stage (bypass/scoring).",
    )
    parser.add_argument(
        "--idx",
        type=int,
        default=0,
        help="This node's index in [0, --nodes). Each node processes a disjoint slice.",
    )

    args = parser.parse_args()
    if args.mip_only:
        args.stage = "mip"
    if args.nodes < 1:
        parser.error("--nodes must be >= 1")
    if not (0 <= args.idx < args.nodes):
        parser.error("--idx must be in [0, --nodes)")
    if args.nodes > 1 and args.stage not in MULTI_NODE_STAGES:
        parser.error(
            f"--nodes > 1 is only supported for stages {MULTI_NODE_STAGES} "
            f"(got --stage {args.stage}); other stages are single-node."
        )
    return args


def run_full_puzzletron(hydra_config_path: str):
    """Run the full puzzletron pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """
    # Register Hydra custom resolvers (needed for config resolution)
    mtpz.tools.register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config to determine total step count (bypass adds one step)
    hydra_cfg = mtpz.tools.initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )
    start_step, total_steps = mtpz.puzzletron_nas_plugin._progress_step(hydra_cfg, "start")

    mtpz.tools.mprint(
        f"Puzzletron Progress {start_step}/{total_steps}: starting puzzletron pipeline"
    )

    # Default timeout: 10 minutes, or extended to nccl_timeout_minutes if set in config
    if hasattr(hydra_cfg, "nccl_timeout_minutes"):
        timeout_minutes = hydra_cfg.nccl_timeout_minutes
    else:
        timeout_minutes = timedelta(minutes=10)

    dist.setup(timeout=timeout_minutes)

    # Convert model (convert from HF to DeciLM, score pruning activations,
    # prune the model and save pruned checkpoints)
    input_model = mtpz.puzzletron_nas_plugin.PuzzletronModel()
    converted_model = mtn.convert(
        input_model,
        mode=[
            (
                "puzzletron",
                {
                    "puzzle_dir": str(hydra_cfg.puzzle_dir),
                    "input_model_path": hydra_cfg.input_hf_model_path,
                    "hydra_config_dir": hydra_config_dir,
                    "hydra_config_name": hydra_config_name,
                    "dataset_path": str(hydra_cfg.dataset_path),
                },
            )
        ],
    )

    # Run NAS search (build replacement library and compute stats,
    # compute one block scores, run MIP and realize models)
    mtn.search(
        converted_model,
        constraints={},  # this is not used as the search space is defined in the hydra config
        dummy_input=None,  # Not used
        config={},  # this is not used as the search space is defined in the hydra config
    )

    dist.cleanup()
    complete_step, _ = mtpz.puzzletron_nas_plugin._progress_step(hydra_cfg, "complete")
    mtpz.tools.mprint(
        f"Puzzletron Progress {complete_step}/{total_steps}: puzzletron pipeline completed (multi-gpu)"
    )


def run_mip_only(hydra_config_path: str):
    """Run only the MIP search and realize models phase.

    This assumes that pruning, replacement library building, NAS scoring, and subblock stats calculation
    have already been completed.

    Args:
        hydra_config_path: Path to the YAML configuration file
    """
    dist.setup(timeout=timedelta(minutes=10))

    # Register Hydra custom resolvers (needed for config resolution)
    mtpz.tools.register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    hydra_cfg = mtpz.tools.initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )
    mip_step, total_steps = mtpz.puzzletron_nas_plugin._progress_step(hydra_cfg, "mip")

    # Check if sweep mode is enabled
    if hasattr(hydra_cfg.mip, "sweep") and hydra_cfg.mip.sweep.get("enabled", False):
        mtpz.tools.mprint(
            f"Puzzletron Progress {mip_step}/{total_steps}:"
            " running MIP sweep for multiple compression rates (multi-gpu)"
        )
        mtpz.mip.run_mip_sweep(hydra_cfg)
    else:
        # mip_and_realize_models (distributed processing)
        # TODO: How to make it part of mnt.search() api, similarly to run_full_puzzletron() API
        mtpz.tools.mprint(
            f"Puzzletron Progress {mip_step}/{total_steps}: running MIP and realizing models (multi-gpu)"
        )
        mtpz.mip.launch_mip_and_realize_model(hydra_cfg)

    dist.cleanup()
    complete_step, _ = mtpz.puzzletron_nas_plugin._progress_step(hydra_cfg, "complete")
    mtpz.tools.mprint(
        f"Puzzletron Progress {complete_step}/{total_steps}: puzzletron pipeline completed (multi-gpu)"
    )


def _load_hydra_cfg(hydra_config_path: str):
    """Load, resolve, and instantiate the Hydra config for a single-stage run."""
    mtpz.tools.register_hydra_resolvers()
    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_cfg = mtpz.tools.initialize_hydra_config_for_dir(
        config_dir=str(hydra_config_path.parent),
        config_name=hydra_config_path.stem,
        overrides=[],
    )
    # Instantiate nested Hydra configs (e.g., pruning_mixin, hook_class), matching
    # how the full pipeline (puzzletron_nas_plugin.run_search) prepares the config.
    return hydra.utils.instantiate(hydra_cfg)


def _stage_timeout(hydra_cfg):
    """NCCL/process-group timeout from config, defaulting to 10 minutes."""
    if hasattr(hydra_cfg, "nccl_timeout_minutes"):
        return hydra_cfg.nccl_timeout_minutes
    return timedelta(minutes=10)


def run_stage_bypass(hydra_config_path: str, num_nodes: int, node_index: int):
    """Run only the bypass-distillation stage (splittable across nodes).

    Assumes teacher conversion and pruning checkpoints already exist in puzzle_dir.
    """
    hydra_cfg = _load_hydra_cfg(hydra_config_path)
    if hydra_cfg.get("bypass", None) is None:
        mtpz.tools.mprint("No 'bypass' section in config; nothing to do.")
        return
    dist.setup(timeout=_stage_timeout(hydra_cfg))
    try:
        mtpz.tools.mprint(
            f"Running bypass distillation stage (node {node_index}/{num_nodes})"
        )
        mtpz.bypass_distillation.launch_bypass_distillation(
            hydra_cfg, num_nodes=num_nodes, node_index=node_index
        )
    finally:
        dist.cleanup()


def run_stage_build_library(hydra_config_path: str):
    """Run only the build-replacement-library + subblock-stats stage.

    Single-process by design: the vLLM runtime benchmark fans out across all
    visible GPUs internally, so launch this with ``--nproc_per_node=1``.
    """
    hydra_cfg = _load_hydra_cfg(hydra_config_path)
    dist.setup(timeout=_stage_timeout(hydra_cfg))
    try:
        if dist.is_master():
            mtpz.tools.mprint("Running build replacement library + subblock stats stage")
            mtpz.build_library_and_stats.launch_build_library_and_stats(hydra_cfg)
        dist.barrier()
    finally:
        dist.cleanup()


def run_stage_scoring(hydra_config_path: str, num_nodes: int, node_index: int):
    """Run only the scoring stage (splittable across nodes).

    Assumes the replacement library and single-block solutions already exist.
    Per-solution result files are written to a shared output dir, so nodes
    coordinate (and resume) implicitly via ``skip_existing_solutions``.
    """
    hydra_cfg = _load_hydra_cfg(hydra_config_path)
    dist.setup(timeout=_stage_timeout(hydra_cfg))
    try:
        mtpz.tools.mprint(f"Running scoring stage (node {node_index}/{num_nodes})")
        mtpz.scoring.launch_scoring(hydra_cfg, num_nodes=num_nodes, node_index=node_index)
    finally:
        dist.cleanup()


def main():
    args = parse_args()

    if args.stage == "full":
        run_full_puzzletron(hydra_config_path=args.config)
    elif args.stage == "bypass":
        run_stage_bypass(args.config, num_nodes=args.nodes, node_index=args.idx)
    elif args.stage == "build_library":
        run_stage_build_library(args.config)
    elif args.stage == "scoring":
        run_stage_scoring(args.config, num_nodes=args.nodes, node_index=args.idx)
    elif args.stage == "mip":
        run_mip_only(hydra_config_path=args.config)
    else:
        raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
