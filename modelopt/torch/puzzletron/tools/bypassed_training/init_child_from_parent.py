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
# mypy: ignore-errors

"""Initialize child models from parent models using AnyModel approach with deci_x_patcher."""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoModelForCausalLM

from modelopt.torch.puzzletron.anymodel.model_descriptor import (
    ModelDescriptor,
    ModelDescriptorFactory,
)
from modelopt.torch.puzzletron.anymodel.puzzformer import deci_x_patcher
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    GQAInitMode,
    HiddenSizeInitMode,
    LinearInitMode,
    MlpInitMode,
    create_child_state_dict,
    update_model_config,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import copy_tokenizer, load_state_dict
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import (
    _save_checkpoint,
    copy_deci_lm_hf_code,
    load_model_config,
)
from modelopt.torch.puzzletron.tools.logger import mprint
from modelopt.torch.puzzletron.tools.sharded_checkpoint_utils import _get_model_class_from_config


def init_child_from_parent(
    descriptor: ModelDescriptor,
    pruning_mixin,
    parent_checkpoint_dir: str,
    model_config_overrides_dict: dict,
    output_checkpoint_dir: str,
    gqa_init_mode: GQAInitMode,
    mlp_init_mode: MlpInitMode,
    mlp_init_config_yaml: Optional[str],
    linear_init_mode: LinearInitMode,
    hidden_size_init_mode: Optional[HiddenSizeInitMode] = None,
    channel_importance_path: Optional[str] = None,
    max_workers: Optional[int] = None,  # Auto-calculate optimal workers if None
    max_layer_workers: Optional[int] = None,  # Auto-calculate optimal workers if None
) -> None:
    """
    Init child models from parent models in the style of bypass training,
    but without having to run the entire bypass pipeline.

    Uses AnyModel approach with deci_x_patcher for heterogeneous layer configurations.

    I/O Optimization Parameters:
    - max_workers: Number of threads for parallel file I/O (default: auto-calculate min(CPU count, num files))
    - max_layer_workers: Number of threads for parallel layer processing (default: auto-calculate min(CPU count, num layers))
    """
    assert (
        gqa_init_mode != GQAInitMode.RandomKV
        and gqa_init_mode != GQAInitMode.RandomBlock
        and mlp_init_mode != MlpInitMode.Random
        and linear_init_mode != LinearInitMode.Random
    ), (
        "We do not support random init of any subblock in this script to avoid initializing the student model"
    )

    descriptor = ModelDescriptorFactory.get(descriptor)

    copy_tokenizer(parent_checkpoint_dir, output_checkpoint_dir)

    parent_model_config = load_model_config(parent_checkpoint_dir)
    parent_state_dict = load_state_dict(parent_checkpoint_dir)

    # Parse JSON if string
    if isinstance(model_config_overrides_dict, str):
        model_config_overrides_dict = json.loads(model_config_overrides_dict)

    # Separate global config overrides from block-level overrides
    global_config_overrides = {}
    block_config_overrides = {}

    for key, value in model_config_overrides_dict.items():
        if key in ["hidden_size"]:
            global_config_overrides[key] = value
        else:
            block_config_overrides[key] = value

    # Load child model config with global overrides
    child_model_config = load_model_config(
        parent_checkpoint_dir,
        model_config_overrides=global_config_overrides,
        ignore_unexpected_config_keys=True,
    )

    # Apply block-level overrides if any
    if block_config_overrides:
        child_model_config = update_model_config(
            model_config=child_model_config,
            model_config_overrides=block_config_overrides,
        )

    with torch.device("meta"):
        # Pass block_configs explicitly so patcher works for VL models where
        # decoder layers receive nested config (e.g., text_config) without block_configs
        with deci_x_patcher(
            model_descriptor=descriptor, block_configs=child_model_config.block_configs
        ):
            model_class = _get_model_class_from_config(child_model_config)
            # AutoModelForCausalLM uses from_config(); concrete model classes use _from_config()
            if model_class is AutoModelForCausalLM:
                child_model = model_class.from_config(child_model_config, trust_remote_code=True)
            else:
                child_model = model_class._from_config(child_model_config)

    child_state_dict_with_meta_tensors = child_model.state_dict()

    mlp_init_config = (
        yaml.safe_load(mlp_init_config_yaml)
        if isinstance(mlp_init_config_yaml, str)
        else mlp_init_config_yaml
    )

    # Profile create_child_state_dict with automatic layer parallelization
    mprint("Starting create_child_state_dict...")
    start_time = time.time()
    child_state_dict = create_child_state_dict(
        pruning_mixin=pruning_mixin,
        descriptor=descriptor,
        original_state_dict=parent_state_dict,
        new_state_dict=child_state_dict_with_meta_tensors,
        original_config=parent_model_config,
        new_config=child_model_config,
        gqa_init_mode=gqa_init_mode,
        mlp_init_mode=mlp_init_mode,
        mlp_init_config=mlp_init_config,
        linear_init_mode=linear_init_mode,
        hidden_size_init_mode=hidden_size_init_mode or HiddenSizeInitMode.CopyAsIs,
        channel_importance_path=channel_importance_path,
        max_layer_workers=max_layer_workers,
    )
    create_child_state_dict_time = time.time() - start_time
    mprint(f"create_child_state_dict completed in {create_child_state_dict_time:.2f} seconds")

    # Profile _save_checkpoint with automatic I/O worker calculation
    mprint("Starting _save_checkpoint...")
    actual_io_workers = max_workers if max_workers else "auto"
    mprint(f"I/O Settings: max_workers={actual_io_workers}")
    start_time = time.time()
    _save_checkpoint(
        child_model_config,
        child_state_dict,
        output_checkpoint_dir,
        descriptor,
        max_workers=max_workers,
    )
    save_checkpoint_time = time.time() - start_time
    mprint(f"_save_checkpoint completed in {save_checkpoint_time:.2f} seconds")

    copy_deci_lm_hf_code(output_checkpoint_dir)

    # Print profiling summary with actual worker counts used
    total_core_time = create_child_state_dict_time + save_checkpoint_time
    actual_layer_workers = max_layer_workers if max_layer_workers else "auto"
    actual_io_workers = max_workers if max_workers else "auto"
    mprint(f"\n=== PROFILING SUMMARY ===")
    mprint(
        f"create_child_state_dict: {create_child_state_dict_time:.2f}s ({create_child_state_dict_time / total_core_time * 100:.1f}%)"
    )
    mprint(
        f"_save_checkpoint: {save_checkpoint_time:.2f}s ({save_checkpoint_time / total_core_time * 100:.1f}%)"
    )
    mprint(f"Total core processing: {total_core_time:.2f}s")
    mprint(f"Optimizations: I/O workers={actual_io_workers}, Layer workers={actual_layer_workers}")
    mprint(f"=========================\n")
