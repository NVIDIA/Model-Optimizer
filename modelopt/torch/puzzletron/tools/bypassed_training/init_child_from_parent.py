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

"""TODO Add description"""

import json
import time

import torch
import yaml

from modelopt.torch.puzzletron.decilm.deci_lm_hf_code.modeling_decilm import DeciLMForCausalLM
from modelopt.torch.puzzletron.tools.bypassed_training.child_init import (
    GQAInitMode,
    HiddenSizeInitMode,
    LinearInitMode,
    MlpInitMode,
    create_child_state_dict,
    update_model_config,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import (
    copy_tokenizer,
    load_model_config,
    load_state_dict,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import (
    _save_checkpoint,
    copy_deci_lm_hf_code,
)
from modelopt.torch.puzzletron.tools.logger import mprint

"""

Usage example - remove all/some routed experts:
===============================================

PARENT_DIR=".../meta-llama/Llama-4-Scout-17B-16E-Instruct--deci-hf"

MLP_INIT_MODE="ConcatExpertsIntoDenseFFN"

## remove all routed experts, turn the shared expert into a dense FFN
# OUTPUT_DIR="/.../micro_scout/Scout-remove-routed-experts"
# MODEL_CONFIG_OVERRIDES_JSON='
# {
#     "ffn": [
#         {
#             "moe": null,
#             "intermediate_size": 14336,
#             "gated": true,
#             "hidden_act": "silu"
#         }
#     ]
# }
# '

## concat the shared expert with one routed expert into a dense FFN
OUTPUT_DIR=".../scratch/micro_scout/Scout-ConcatExpertsIntoDenseFFN-concat-shared-and-3-routed"
MODEL_CONFIG_OVERRIDES_JSON='
{
    "ffn": [
        {
            "moe": null,
            "intermediate_size": 14336,
            "gated": true,
            "hidden_act": "silu"
        }
    ]
}
'

echo ""
echo "MODEL_CONFIG_OVERRIDES_JSON:"
echo "${MODEL_CONFIG_OVERRIDES_JSON}"

python -m modelopt.torch.puzzletron.tools.bypassed_training.init_child_from_parent  \
    --parent_checkpoint_dir="$PARENT_DIR" \
    --model_config_overrides_json="$MODEL_CONFIG_OVERRIDES_JSON" \
    --output_checkpoint_dir="$OUTPUT_DIR" \
    --mlp_init_mode="$MLP_INIT_MODE" \
    --mlp_init_config_yaml="$MLP_INIT_CONFIG_YAML"
"""


def init_child_from_parent(
    parent_checkpoint_dir: str,
    model_config_overrides_json: str,
    output_checkpoint_dir: str,
    gqa_init_mode: GQAInitMode,
    mlp_init_mode: MlpInitMode,
    mlp_init_config_yaml: str | None,
    linear_init_mode: LinearInitMode,
    hidden_size_init_mode: HiddenSizeInitMode | None = None,
    channel_importance_path: str | None = None,
    max_workers: int | None = None,  # Auto-calculate optimal workers if None
    max_layer_workers: int | None = None,  # Auto-calculate optimal workers if None
) -> None:
    """Init child models from parent models in the style of bypass training,
    but without having to run the entire bypass pipeline.

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

    copy_tokenizer(parent_checkpoint_dir, output_checkpoint_dir)

    parent_model_config = load_model_config(parent_checkpoint_dir)
    parent_state_dict = load_state_dict(parent_checkpoint_dir)

    # Parse the model config overrides
    if isinstance(model_config_overrides_json, str):
        model_config_overrides_dict = json.loads(model_config_overrides_json)
    else:
        model_config_overrides_dict = model_config_overrides_json

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
        checkpoint_dir=parent_checkpoint_dir,
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
        child_model = DeciLMForCausalLM(child_model_config)
    child_state_dict_with_meta_tensors = child_model.state_dict()

    mlp_init_config = (
        yaml.safe_load(mlp_init_config_yaml)
        if isinstance(mlp_init_config_yaml, str) is None
        else mlp_init_config_yaml
    )

    # Profile create_child_state_dict with automatic layer parallelization
    mprint("Starting create_child_state_dict...")
    start_time = time.time()
    child_state_dict = create_child_state_dict(
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
        max_layer_workers=max_layer_workers,  # Will auto-calculate if None
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
        max_workers=max_workers,  # Will auto-calculate if None
    )
    save_checkpoint_time = time.time() - start_time
    mprint(f"_save_checkpoint completed in {save_checkpoint_time:.2f} seconds")

    copy_deci_lm_hf_code(output_checkpoint_dir)

    # Print profiling summary with actual worker counts used
    total_core_time = create_child_state_dict_time + save_checkpoint_time
    actual_layer_workers = max_layer_workers if max_layer_workers else "auto"
    actual_io_workers = max_workers if max_workers else "auto"
    mprint("\n=== PROFILING SUMMARY ===")
    mprint(
        f"create_child_state_dict: {create_child_state_dict_time:.2f}s ({create_child_state_dict_time / total_core_time * 100:.1f}%)"
    )
    mprint(
        f"_save_checkpoint: {save_checkpoint_time:.2f}s ({save_checkpoint_time / total_core_time * 100:.1f}%)"
    )
    mprint(f"Total core processing: {total_core_time:.2f}s")
    mprint(f"Optimizations: I/O workers={actual_io_workers}, Layer workers={actual_layer_workers}")
    mprint("=========================\n")
