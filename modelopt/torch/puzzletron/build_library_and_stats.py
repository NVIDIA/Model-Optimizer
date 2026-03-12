#!/usr/bin/env python3
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
Unified command that runs build_replacement_library followed by calc_subblock_stats.

This script combines the functionality of both commands into a single workflow:
1. First, it builds the replacement library for the puzzle
2. Then, it calculates subblock statistics

Usage:

    python modelopt.torch.puzzletron.build_library_and_stats.py --config-dir configs --config-name Llama-3_1-8B puzzle_dir=/path/to/puzzle/dir dataset_path=/path/to/dataset

The script uses the same Hydra configuration as the individual commands and supports
all the same configuration parameters for both build_replacement_library and calc_subblock_stats.
"""

import hydra
from omegaconf import DictConfig

from modelopt.torch.puzzletron.replacement_library.build_replacement_library import (
    launch_build_replacement_library,
)
from modelopt.torch.puzzletron.subblock_stats.calc_subblock_stats import launch_calc_subblock_stats
from modelopt.torch.puzzletron.tools.hydra_utils import register_hydra_resolvers
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.utils.parsing import format_global_config


def launch_build_library_and_stats(cfg: DictConfig) -> None:
    """
    Launch both build_replacement_library and calc_subblock_stats in sequence.

    Args:
        cfg: Hydra configuration containing settings for both commands
    """
    mprint("=" * 80)
    mprint("STARTING UNIFIED BUILD LIBRARY AND STATS WORKFLOW")
    mprint("=" * 80)

    # Step 1: Build replacement library
    mprint("=" * 50)
    mprint("STEP 1: Building Replacement Library")
    mprint("=" * 50)

    try:
        launch_build_replacement_library(cfg)
        mprint("✅ Replacement library built successfully!")
    except Exception as e:
        mprint(f"❌ Failed to build replacement library: {e}")
        raise

    # Step 2: Calculate subblock statistics
    mprint("=" * 50)
    mprint("STEP 2: Calculating Subblock Statistics")
    mprint("=" * 50)

    try:
        launch_calc_subblock_stats(cfg)
        mprint("✅ Subblock statistics calculated successfully!")
    except Exception as e:
        mprint(f"❌ Failed to calculate subblock statistics: {e}")
        raise

    mprint("=" * 80)
    mprint("UNIFIED WORKFLOW COMPLETED SUCCESSFULLY! 🎉")
    mprint("=" * 80)

    mprint("Generated files:")
    mprint(f"  - {cfg.puzzle_dir}/block_library.json")
    mprint(f"  - {cfg.puzzle_dir}/subblock_library.json")
    mprint(f"  - {cfg.puzzle_dir}/replacement_library.json")
    mprint(f"  - {cfg.puzzle_dir}/single_sequence_replacement_solutions.json")
    mprint(f"  - {cfg.puzzle_dir}/{cfg.calc_subblock_stats.subblock_stats_filename}")
    if hasattr(cfg.calc_subblock_stats, "moe_stats_filename"):
        mprint(f"  - {cfg.puzzle_dir}/{cfg.calc_subblock_stats.moe_stats_filename}")
