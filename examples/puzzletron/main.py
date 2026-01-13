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

This script provides two modes:
1. Default mode: Runs the full puzzletron pipeline
2. MIP-only mode: Runs only the MIP search and realize models phase

Usage:
    # Full puzzletron pipeline
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml

    # Only MIP search and realize models phase
    torchrun main.py --config ./configs/llama_3.2_1B_pruneffn_memory.yaml --mip-only
"""

import argparse
from datetime import timedelta
from pathlib import Path

import modelopt.torch.nas as mtn
import modelopt.torch.puzzletron.mip.mip_and_realize_models as mip_and_realize_models
import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.nas.plugins.puzzletron_nas_plugin import PuzzletronModel
from modelopt.torch.puzzletron.tools.hydra_utils import (
    initialize_hydra_config_for_dir,
    register_hydra_resolvers,
)
from modelopt.torch.puzzletron.tools.logger import mprint


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
        help="Run only the MIP search and realize models phase (skip pruning and NAS scoring)",
    )

    return parser.parse_args()


def run_full_puzzletron(hydra_config_path: str):
    """Run the full puzzletron pipeline.

    Args:
        config_path: Path to the YAML configuration file
    """
    mprint("Puzzletron Progress 1/8: starting puzzletron pipeline")
    dist.setup(timeout=timedelta(10))

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )

    # Convert model (convert from HF to DeciLM, score pruning activations,
    # prune the model and save pruned checkpoints)
    input_model = PuzzletronModel()
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
    mprint("Puzzletron Progress 8/8: puzzletron pipeline completed (multi-gpu)")


def run_mip_only(hydra_config_path: str):
    """Run only the MIP search and realize models phase.

    This assumes that pruning, replacement library building, NAS scoring, and subblock stats calculation
    have already been completed.

    Args:
        hydra_config_path: Path to the YAML configuration file
    """
    dist.setup(timeout=timedelta(10))

    # Register Hydra custom resolvers (needed for config resolution)
    register_hydra_resolvers()

    hydra_config_path = Path(hydra_config_path).resolve()
    hydra_config_dir = str(hydra_config_path.parent)
    hydra_config_name = hydra_config_path.stem

    # Load hydra config
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=hydra_config_dir,
        config_name=hydra_config_name,
        overrides=[],
    )

    # mip_and_realize_models (distributed processing)
    # TODO: How to make it part of mnt.search() api, similarly to run_full_puzzletron() API
    mprint("Puzzletron Progress 7/8: running MIP and realizing models (multi-gpu)")
    mip_and_realize_models.launch_mip_and_realize_model(hydra_cfg)

    dist.cleanup()
    mprint("Puzzletron Progress 8/8: puzzletron pipeline completed (multi-gpu)")


def main():
    args = parse_args()

    if args.mip_only:
        run_mip_only(hydra_config_path=args.config)
    else:
        run_full_puzzletron(hydra_config_path=args.config)


if __name__ == "__main__":
    main()
