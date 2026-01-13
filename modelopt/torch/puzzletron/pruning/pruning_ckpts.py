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
"""Utilities for creating pruned model checkpoints.

This module provides functions to generate pruned checkpoints by modifying model architectures
(FFN intermediate sizes, attention head groups, hidden dimensions) and initializing child pruned models
from parent checkpoints.
"""

# mypy: ignore-errors
import json
import os
import time

from omegaconf import DictConfig

from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    GQAInitMode,
    HiddenSizeInitMode,
    LinearInitMode,
    MlpInitMode,
)
from modelopt.torch.puzzletron.tools.bypassed_training.init_child_from_parent import (
    init_child_from_parent,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config
from modelopt.torch.puzzletron.tools.logger import mprint


def launch_ffn_intermediates_prune_ckpt(
    cfg: DictConfig, max_save_workers: int | None = None, max_layer_workers: int | None = None
):
    for intermediate_size in cfg.pruning.intermediate_size_list:
        dirname = f"ffn_{intermediate_size}_attn_no_op"

        if os.path.exists(os.path.join(cfg.puzzle_dir, "ckpts", dirname)):
            mprint(f"Process intermediate_size {intermediate_size} has already been pruned & saved")
            continue

        mprint("Process intermediate_size {}".format(intermediate_size))

        model_config_overrides_json = {"ffn": [{"intermediate_size": intermediate_size}]}
        mlp_init_config_yaml = cfg.pruning.mlp_init_config_yaml

        output_dir = os.path.join(cfg.pruning.pruned_ckpts_outpt_dir, dirname)

        # Profile the overall init_child_from_parent call with optimizations
        mprint("Starting init_child_from_parent...")
        start_time = time.time()
        init_child_from_parent(
            parent_checkpoint_dir=cfg.teacher_dir,
            model_config_overrides_json=model_config_overrides_json,
            output_checkpoint_dir=output_dir,
            gqa_init_mode=GQAInitMode(cfg.pruning.gqa_init_mode),
            mlp_init_mode=MlpInitMode(cfg.pruning.mlp_init_mode),
            mlp_init_config_yaml=mlp_init_config_yaml,
            linear_init_mode=LinearInitMode.FromTeacher,  # dummy default value
            max_workers=max_save_workers,  # Will auto-calculate if None
            max_layer_workers=max_layer_workers,  # Will auto-calculate if None
        )
        init_child_from_parent_time = time.time() - start_time
        mprint(f"init_child_from_parent completed in {init_child_from_parent_time:.2f} seconds")

        # Create symlink in puzzle_dir/ckpts
        ckpt_path = os.path.join(cfg.puzzle_dir, "ckpts")
        os.makedirs(ckpt_path, exist_ok=True)
        os.symlink(output_dir, os.path.join(ckpt_path, dirname))

        mprint(f"=== COMPLETED FFN PRUNING FOR FFN INTERMEDIATE SIZE={intermediate_size} ===")
        mprint(f"Total processing time: {init_child_from_parent_time:.2f} seconds\n")


def launch_attn_groups_prune_ckpt(
    cfg: DictConfig, max_save_workers: int | None = None, max_layer_workers: int | None = None
):
    for n_heads_in_group in cfg.pruning.n_heads_in_group_list:
        dirname = f"n_heads_in_group{n_heads_in_group}"

        if os.path.exists(os.path.join(cfg.puzzle_dir, "ckpts", dirname)):
            mprint(f"Process n_heads_in_group {n_heads_in_group} has already been pruned & saved")
            continue

        mprint("Process n_heads_in_group {}".format(n_heads_in_group))
        mprint(f"=== STARTING ATTENTION PRUNING FOR n_heads_in_group={n_heads_in_group} ===")

        model_config_overrides_json = {"attention": [{"n_heads_in_group": n_heads_in_group}]}
        mlp_init_config_yaml = cfg.pruning.mlp_init_config_yaml

        output_dir = os.path.join(cfg.pruning.pruned_ckpts_outpt_dir, dirname)

        # Profile the overall init_child_from_parent call with optimizations
        mprint("Starting init_child_from_parent...")
        start_time = time.time()
        init_child_from_parent(
            parent_checkpoint_dir=cfg.teacher_dir,
            model_config_overrides_json=model_config_overrides_json,
            output_checkpoint_dir=output_dir,
            gqa_init_mode=GQAInitMode(cfg.pruning.gqa_init_mode),
            mlp_init_mode=MlpInitMode(cfg.pruning.mlp_init_mode),
            mlp_init_config_yaml=mlp_init_config_yaml,
            linear_init_mode=LinearInitMode.FromTeacher,  # dummy default value
            max_workers=max_save_workers,  # Will auto-calculate if None
            max_layer_workers=max_layer_workers,  # Will auto-calculate if None
        )
        init_child_from_parent_time = time.time() - start_time
        mprint(f"init_child_from_parent completed in {init_child_from_parent_time:.2f} seconds")

        # Create symlink in puzzle_dir/ckpts
        ckpt_path = os.path.join(cfg.puzzle_dir, "ckpts")
        os.makedirs(ckpt_path, exist_ok=True)
        os.symlink(output_dir, os.path.join(ckpt_path, dirname))

        mprint(f"=== COMPLETED ATTENTION PRUNING FOR n_heads_in_group={n_heads_in_group} ===")
        mprint(f"Total processing time: {init_child_from_parent_time:.2f} seconds\n")


def launch_hidden_dim_prune_ckpt(cfg: DictConfig):
    """Launch hidden dimension pruning using channel importance ranking."""
    # Get channel importance results from the activations log directory
    activations_log_dir = cfg.pruning.activations_log_dir
    channel_importance_path = os.path.join(activations_log_dir, "channel_importance_results.json")

    if not os.path.exists(channel_importance_path):
        raise FileNotFoundError(
            f"Channel importance results not found at {channel_importance_path}. "
            f"Make sure to run the activation collection step first."
        )

    # Load parent model config to get FFN configuration
    parent_model_config = load_model_config(cfg.pruning.model_name_or_path)
    parent_hidden_size = parent_model_config.hidden_size

    # Get teacher's FFN configuration
    intermediate_sizes = []
    for block_config in parent_model_config.block_configs:
        if block_config.ffn.intermediate_size is not None:
            intermediate_sizes.append(block_config.ffn.intermediate_size)
        else:
            intermediate_sizes.append(None)

    mprint("Teacher config:")
    mprint(f"  - hidden_size: {parent_hidden_size}")
    mprint(f"  - intermediate_sizes: {intermediate_sizes}")
    os.makedirs(os.path.join(cfg.puzzle_dir, "ckpts"), exist_ok=True)

    for hidden_size in cfg.pruning.hidden_size_list:
        mprint("\n######################################################################")
        mprint(f"Hidden Size = {hidden_size}")
        mprint("######################################################################\n")

        mprint("Child config:")
        mprint(f"  - hidden_size: {hidden_size}")

        # Create model config overrides with proper FFN configuration
        model_config_overrides_json = json.dumps(
            {
                "hidden_size": hidden_size,
                "ffn": [
                    {
                        "intermediate_size": intermediate_size,
                    }
                    for intermediate_size in intermediate_sizes
                ],
            }
        )

        mlp_init_config_yaml = cfg.pruning.mlp_init_config_yaml
        dirname = f"hidden_size_{hidden_size}"
        output_dir = os.path.join(cfg.pruning.pruned_ckpts_outpt_dir, dirname)

        mprint(f"Creating checkpoint with hidden_size={hidden_size}")
        mprint(f"Model config overrides: {model_config_overrides_json}")

        init_child_from_parent(
            parent_checkpoint_dir=cfg.pruning.model_name_or_path,
            model_config_overrides_json=model_config_overrides_json,
            output_checkpoint_dir=output_dir,
            gqa_init_mode=GQAInitMode(cfg.pruning.gqa_init_mode),
            mlp_init_mode=MlpInitMode(cfg.pruning.mlp_init_mode),
            mlp_init_config_yaml=mlp_init_config_yaml,
            linear_init_mode=LinearInitMode(cfg.pruning.linear_init_mode),
            hidden_size_init_mode=HiddenSizeInitMode(cfg.pruning.hidden_size_init_mode),
            channel_importance_path=channel_importance_path,
        )

        # Create symlink in puzzle_dir/ckpts
        ckpt_path = os.path.join(cfg.puzzle_dir, "ckpts")
        os.makedirs(ckpt_path, exist_ok=True)
        os.symlink(output_dir, os.path.join(ckpt_path, dirname))
        mprint(f"Created pruned checkpoint at: {output_dir}")


def launch_experts_prune_ckpt(
    cfg: DictConfig,
    max_save_workers: int | None = None,
    max_layer_workers: int | None = None,
    symlink_suffix: str | None = None,
):
    for num_experts in cfg.pruning.num_experts_to_keep_list:
        dirname = f"num_experts_{num_experts}"
        # Create symlink name with optional suffix
        symlink_name = f"{dirname}_{symlink_suffix}" if symlink_suffix else dirname
        if os.path.exists(os.path.join(cfg.puzzle_dir, "ckpts", symlink_name)):
            mprint(
                f"Process num_experts {num_experts} (symlink: {symlink_name}) has already been pruned & saved"
            )
            continue
        mprint(f"Process num_experts {num_experts}")
        mprint(f"=== STARTING EXPERT PRUNING FOR num_experts={num_experts} ===")
        model_config_overrides_json = {"ffn": [{"moe": {"num_local_experts": num_experts}}]}

        mlp_init_config_yaml = cfg.pruning.mlp_init_config_yaml

        output_dir = os.path.join(cfg.pruning.pruned_ckpts_outpt_dir, dirname)

        # Profile the overall init_child_from_parent call with optimizations
        mprint("Starting init_child_from_parent...")
        start_time = time.time()
        init_child_from_parent(
            parent_checkpoint_dir=cfg.teacher_dir,
            model_config_overrides_json=model_config_overrides_json,
            output_checkpoint_dir=output_dir,
            gqa_init_mode=GQAInitMode(cfg.pruning.gqa_init_mode),
            mlp_init_mode=MlpInitMode(cfg.pruning.mlp_init_mode),
            mlp_init_config_yaml=mlp_init_config_yaml,
            linear_init_mode=LinearInitMode.FromTeacher,  # dummy default value
            max_workers=max_save_workers,  # Will auto-calculate if None
            max_layer_workers=max_layer_workers,  # Will auto-calculate if None
        )
        init_child_from_parent_time = time.time() - start_time
        mprint(f"init_child_from_parent completed in {init_child_from_parent_time:.2f} seconds")

        # Create symlink in puzzle_dir/ckpts
        ckpt_path = os.path.join(cfg.puzzle_dir, "ckpts")
        os.makedirs(ckpt_path, exist_ok=True)
        os.symlink(output_dir, os.path.join(ckpt_path, symlink_name))

        mprint(f"=== COMPLETED EXPERT PRUNING FOR num_experts={num_experts} ===")
        mprint(f"Total processing time: {init_child_from_parent_time:.2f} seconds\n")


def launch_moe_ffn_intermediates_prune_ckpt(
    cfg: DictConfig, max_save_workers: int | None = None, max_layer_workers: int | None = None
):
    for intermediate_size in cfg.pruning.intermediate_size_list:
        dirname = f"moe_ffn_{intermediate_size}_attn_no_op"

        if os.path.exists(os.path.join(cfg.puzzle_dir, "ckpts", dirname)):
            mprint(f"Process intermediate_size {intermediate_size} has already been pruned & saved")
            continue

        mprint("Process intermediate_size {}".format(intermediate_size))

        model_config_overrides_json = {
            "attention": [{"no_op": True, "llama4": None}],
            "ffn": [{"moe": {"expert_intermediate_dim": intermediate_size}}],
        }
        mlp_init_config_yaml = cfg.pruning.mlp_init_config_yaml

        output_dir = os.path.join(cfg.pruning.pruned_ckpts_outpt_dir, dirname)

        # Profile the overall init_child_from_parent call with optimizations
        mprint("Starting init_child_from_parent...")
        start_time = time.time()
        init_child_from_parent(
            parent_checkpoint_dir=cfg.teacher_dir,
            model_config_overrides_json=model_config_overrides_json,
            output_checkpoint_dir=output_dir,
            gqa_init_mode=GQAInitMode(cfg.pruning.gqa_init_mode),
            mlp_init_mode=MlpInitMode(cfg.pruning.mlp_init_mode),
            mlp_init_config_yaml=mlp_init_config_yaml,
            linear_init_mode=LinearInitMode.FromTeacher,  # dummy default value
            max_workers=max_save_workers,  # Will auto-calculate if None
            max_layer_workers=max_layer_workers,  # Will auto-calculate if None
        )
        init_child_from_parent_time = time.time() - start_time
        mprint(f"init_child_from_parent completed in {init_child_from_parent_time:.2f} seconds")

        # Create symlink in puzzle_dir/ckpts
        os.symlink(output_dir, os.path.join(cfg.puzzle_dir, "ckpts", dirname))

        mprint(f"=== COMPLETED MOE FFN PRUNING FOR FFN INTERMEDIATE SIZE={intermediate_size} ===")
        mprint(f"Total processing time: {init_child_from_parent_time:.2f} seconds\n")


def launch_prune_ckpt(cfg: DictConfig):
    target_layer = cfg.pruning.activation_hooks_kwargs.target_layer
    # I/O optimization settings - same as FFN pruning
    max_save_workers = None  # Will auto-calculate as min(CPU count, num files)
    if "PRUNING_SAVE_WORKERS" in os.environ:
        max_save_workers = int(os.environ["PRUNING_SAVE_WORKERS"])

    # Layer workers now auto-calculate but can still be overridden
    max_layer_workers = None  # Will auto-calculate as min(CPU count, num layers)
    if "PRUNING_LAYER_WORKERS" in os.environ:
        max_layer_workers = int(os.environ["PRUNING_LAYER_WORKERS"])

    # Log optimization settings (extracted from individual pruning methods)
    mprint("Optimization Settings:")
    mprint(
        f"  - I/O workers (max_workers): {'auto-calculate' if max_save_workers is None else max_save_workers}"
    )
    mprint(
        f"  - Layer workers (max_layer_workers): {'auto-calculate' if max_layer_workers is None else max_layer_workers}"
    )
    mprint("  (Override with env vars: PRUNING_IO_WORKERS, PRUNING_LAYER_WORKERS)")

    if target_layer == "mlp.down_proj":
        launch_ffn_intermediates_prune_ckpt(cfg, max_save_workers, max_layer_workers)
    elif target_layer == "self_attn.o_proj":
        launch_attn_groups_prune_ckpt(cfg, max_save_workers, max_layer_workers)
    elif target_layer == "layernorm":
        launch_hidden_dim_prune_ckpt(cfg)
    elif target_layer == "router":
        # Check if we should use symlink suffix for chained pruning
        symlink_suffix = getattr(cfg.pruning, "symlink_suffix", None)
        launch_experts_prune_ckpt(cfg, max_save_workers, max_layer_workers, symlink_suffix)
    elif target_layer == r"regex:experts\.\d+\.down_proj$":
        launch_moe_ffn_intermediates_prune_ckpt(cfg, max_save_workers, max_layer_workers)
    else:
        raise NotImplementedError(
            f"checkpoint pruning is not currently supported for target layer: {target_layer}"
        )
