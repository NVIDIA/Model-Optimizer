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

from ..tools.logger import mprint

__all__ = ["launch_score_activations"]


def _activation_pass_name(pass_cfg) -> str:
    return pass_cfg.get("name", None) or pass_cfg.activation_hooks_kwargs.method


def _filtered_activation_passes(cfg: DictConfig):
    passes = cfg.pruning.get("activation_passes", None)
    if not passes:
        return None

    pass_list = list(passes)
    filters = cfg.pruning.get("activation_pass_filter", None) or {}
    include = set(filters.get("include", []) or [])
    exclude = set(filters.get("exclude", []) or [])
    if not include and not exclude:
        return pass_list

    names = [_activation_pass_name(p) for p in pass_list]
    unknown = (include | exclude) - set(names)
    if unknown:
        raise ValueError(
            "pruning.activation_pass_filter references unknown pass name(s): "
            f"{sorted(unknown)}; available={names}"
        )

    filtered = [
        p
        for p, name in zip(pass_list, names)
        if (not include or name in include) and name not in exclude
    ]
    if not filtered:
        raise ValueError(
            "pruning.activation_pass_filter removed every activation pass; "
            f"include={sorted(include)} exclude={sorted(exclude)}"
        )
    filtered_names = [_activation_pass_name(p) for p in filtered]
    if filtered_names != names:
        mprint(f"[activation] filtered activation passes: {filtered_names} (from {names})")
    return filtered


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

    # When activation_passes is set each pass writes to a subdir of activations_log_dir.
    # The root dir itself never contains rank_*.pth, so checking it always returns False.
    # Instead, verify that every pass subdir is complete.
    passes = _filtered_activation_passes(cfg)
    if passes:
        from pathlib import Path as _Path
        parent = cfg.pruning.activations_log_dir
        is_completed = all(
            check_scoring_completion(
                str(_Path(parent) / _activation_pass_name(p)),
                activation_hooks_kwargs,
            )
            for p in passes
        )
    else:
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


def _run_activation_passes(cfg: DictConfig, num_nodes: int, node_index: int) -> bool:
    """Run multiple scoring passes (e.g. FFN + attention) into one parent ``activations_log_dir``.

    When ``pruning.activation_passes`` is set (a list of per-pass overrides: ``name`` +
    ``pruning_mixin`` / ``activation_hooks_kwargs`` / ``hook_class``), each pass writes its
    ``rank_*.pth`` to ``<activations_log_dir>/<name>/`` so the sorted-teacher builder (which
    ``rglob``s the parent) merges them by module name. Returns ``True`` when passes were run.

    **AutoModel backend**: ALL passes are registered as forward hooks simultaneously and the
    model runs ONE forward pass — O(1) data cost regardless of the number of pass types.
    The hooks attach to different modules (e.g. ``mlp.down_proj`` and ``self_attn.o_proj``)
    and are fully independent; combining them is exact.

    """
    all_passes = list(cfg.pruning.get("activation_passes", None) or [])
    passes = _filtered_activation_passes(cfg)
    if not passes:
        return False

    import json as _json
    from pathlib import Path as _Path

    parent = cfg.pruning.get("activations_log_dir", None) or str(
        _Path(cfg.puzzle_dir) / "pruning" / "pruning_scores"
    )
    pass_names = [_activation_pass_name(p) for p in passes]
    backend = cfg.pruning.get("backend", "automodel")
    if backend != "automodel":
        raise ValueError("Activation scoring only supports pruning.backend=automodel.")

    if should_skip_scoring_completely(cfg):
        mprint("[activation] all passes already complete, skipping multi-pass scoring")
        return True
    from ..plugins.automodel.launch import launch_score_activations_automodel_multipass

    launch_score_activations_automodel_multipass(
        cfg, passes, pass_names, parent, num_nodes, node_index
    )

    # Write a manifest so that sorted_teacher can verify the pass structure hasn't changed.
    # For filtered reruns, keep already-complete sibling pass dirs in the manifest; refreshing
    # attention scores should not make the canonical parent look attention-only.
    if dist.is_master() and pass_names:
        manifest_path = _Path(parent) / "activation_passes_manifest.json"
        manifest_names = [
            _activation_pass_name(pass_cfg)
            for pass_cfg in all_passes
            if check_scoring_completion(
                str(_Path(parent) / _activation_pass_name(pass_cfg)),
                pass_cfg.get("activation_hooks_kwargs", {}),
            )
        ]
        if not manifest_names:
            manifest_names = pass_names
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(_json.dumps({"passes": manifest_names}, indent=2))
        mprint(f"[activation] wrote pass manifest: {manifest_path}")
    return True


def launch_score_activations(cfg: DictConfig, num_nodes: int = 1, node_index: int = 0):
    """Score pruning activations.

    ``num_nodes``/``node_index`` mirror the bypass/scoring stages and allow
    splitting the work across nodes. Activation scoring is handled by the
    AutoModel backend (``pruning.backend: automodel``).

    When ``pruning.activation_passes`` is set, runs one scoring pass per entry (FFN + attention)
    into subdirs of ``activations_log_dir`` — see :func:`_run_activation_passes`.
    """
    if _run_activation_passes(cfg, num_nodes, node_index):
        return

    backend = cfg.pruning.get("backend", "automodel")
    if backend != "automodel":
        raise ValueError("Activation scoring only supports pruning.backend=automodel.")
    mprint(f"[activation] scoring backend = {backend!r}")

    if should_skip_scoring_completely(cfg):
        return

    from ..plugins.automodel.launch import launch_score_activations_automodel

    launch_score_activations_automodel(cfg, num_nodes=num_nodes, node_index=node_index)
