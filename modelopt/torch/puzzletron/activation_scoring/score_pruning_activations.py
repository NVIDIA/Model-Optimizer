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

from pathlib import Path

import torch
from omegaconf import DictConfig

import modelopt.torch.utils.distributed as dist
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.validate_model import validate_model


def has_checkpoint_support(activation_hooks_kwargs: dict) -> bool:
    """Determine if the activation hook method has proper checkpoint support implemented.

    Args:
        activation_hooks_kwargs: Hook configuration

    Returns:
        bool: True if the hook method has save_state/load_state implemented
    """
    method = activation_hooks_kwargs.get("method", "")

    # Methods with implemented checkpoint support
    supported_methods = {
        "iterative",  # IterativeChannelContributionHook: save_state/load_state implemented
        "independent",  # IndependentChannelContributionHook: save_state/load_state implemented
        "stats",  # RouterStatsHook: save_state/load_state implemented
        "ranked_choice_voting",  # RankedChoiceVotingHook: save_state/load_state implemented
    }

    return method in supported_methods


def check_scoring_completion(activations_log_dir: str, activation_hooks_kwargs=None) -> bool:
    """Check if scoring is already completed by looking for the expected output files.
    Also checks if the scoring method is safe for resume.

    Args:
        activations_log_dir: Directory where activation logs should be stored
        activation_hooks_kwargs: Hook configuration to check if resume is safe

    Returns:
        bool: True if scoring is completed (has rank files and args.json)
    """
    # Only check completion on main process
    if dist.is_master():
        log_dir = Path(activations_log_dir)

        # Check if directory exists
        if not log_dir.exists():
            return False

        # Check for rank files (at least rank_0.pth should exist)
        rank_files = list(log_dir.glob("rank_*.pth"))

        if not rank_files:
            return False

        # Check for args.json (created by main process)
        args_file = log_dir / "args.json"
        has_args_json = args_file.exists()

        # Check for completion: if we have rank files and args.json, scoring is complete
        if rank_files and has_args_json:
            # Add optional completion info for debugging
            mprint(f"Found completed scoring in {activations_log_dir}")
            mprint(f"  - Found {len(rank_files)} rank files")
            mprint(f"  - Found args.json: {has_args_json}")

            return True

    return False


def should_skip_scoring_completely(cfg: DictConfig) -> bool:
    """Determine if we should skip scoring entirely (only if 100% complete).
    Partial progress should proceed to validate_model for proper resume.

    Args:
        cfg: Configuration object

    Returns:
        bool: True if we should skip scoring (100% completed), False if we should run/resume it
    """
    # Check if activations_log_dir is specified
    if not hasattr(cfg.pruning, "activations_log_dir") or cfg.pruning.activations_log_dir is None:
        mprint("No activations_log_dir specified, running scoring")
        return False

    # Check for force restart flag
    force_restart = getattr(cfg.pruning, "force_restart_scoring", False)
    if force_restart:
        mprint("Force restart flag set, will restart scoring regardless of existing artifacts")
        return False

    # Get hook configuration to check if resume is mathematically safe
    activation_hooks_kwargs = getattr(cfg.pruning, "activation_hooks_kwargs", {})

    # Check if scoring is already completed
    is_completed = check_scoring_completion(
        cfg.pruning.activations_log_dir, activation_hooks_kwargs
    )

    # Broadcast the result to all processes in distributed mode
    if dist.size() > 1:
        should_skip = [is_completed]  # Use list for mutable object
        torch.distributed.broadcast_object_list(should_skip, src=0)
        is_completed = should_skip[0]

    if is_completed:
        mprint("Scoring 100% completed, skipping...")

    return is_completed


# Old progress tracking removed - checkpoint manager handles all progress tracking


def launch_score_activations(cfg: DictConfig):
    # Check if we should skip scoring entirely (only if 100% complete)
    if should_skip_scoring_completely(cfg):
        return

    mprint("Starting pruning activation scoring...")

    # The checkpoint manager inside validate_model handles all progress tracking
    validate_model(args=cfg.pruning, pipeline_parallel=True)
