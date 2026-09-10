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
Convert a TRTLLM eagle checkpoint to an VLLM compatible one-model checkpoint.
"""

import argparse
import json
import os
import shutil
from copy import deepcopy

VLLM_EAGLE3_ONE_CKPT_CFG_TEMPLATE = {
    "architectures": ["Eagle3Speculator"],
    "auto_map": {"": "eagle3.Eagle3SpeculatorConfig"},
    "draft_vocab_size": None,
    "has_no_defaults_at_init": False,
    "norm_before_residual": True,
    "speculators_config": {
        "algorithm": "eagle3",
        "default_proposal_method": "greedy",
        "proposal_methods": [
            {
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 3,
                "verifier_accept_k": 1,
            }
        ],
        "verifier": {"architectures": [""], "name_or_path": ""},
    },
    "speculators_model_type": "eagle3",
    "speculators_version": "0.1.0.dev14",
    "target_hidden_size": None,
    "torch_dtype": None,
    "transformer_layer_config": {
        "attention_bias": None,
        "attention_dropout": None,
        "head_dim": None,
        "hidden_act": None,
        "hidden_size": None,
        "initializer_range": None,
        "intermediate_size": None,
        "max_position_embeddings": None,
        "mlp_bias": None,
        "model_type": None,
        "num_attention_heads": None,
        "num_hidden_layers": None,
        "num_key_value_heads": None,
        "pretraining_tp": None,
        "rms_norm_eps": None,
        "rope_scaling": None,
        "rope_theta": None,
        "use_cache": True,
        "vocab_size": None,
    },
    "transformers_version": None,
}


def convert_to_eagle3_speculator_config(
    draft_cfg,
    verifier_name_or_path,
    template_cfg=VLLM_EAGLE3_ONE_CKPT_CFG_TEMPLATE,
):
    """
    Convert a draft model config and a verifier model config to an Eagle3Speculator config.
    """

    verifier_config_path = os.path.join(verifier_name_or_path, "config.json")
    with open(verifier_config_path, encoding="utf-8") as verifier_cfg_file:
        verifier_cfg = json.load(verifier_cfg_file)

    speculator_config = deepcopy(template_cfg)

    try:
        # Update speculators_config separately to avoid type conflicts
        speculator_config["speculators_config"].update(
            {
                "verifier": {
                    "architectures": verifier_cfg["architectures"],
                    "name_or_path": verifier_name_or_path,
                },
            }
        )

        # Update other fields
        speculator_config.update(
            {
                "draft_vocab_size": draft_cfg["draft_vocab_size"],
                "target_hidden_size": verifier_cfg["hidden_size"],
                "torch_dtype": draft_cfg["torch_dtype"],
                "transformer_layer_config": {
                    k: draft_cfg[k] for k in template_cfg["transformer_layer_config"]
                },
                "transformers_version": draft_cfg["transformers_version"],
            }
        )
    except Exception as e:
        raise Exception(f"Error converting draft config: {e}")

    return speculator_config


def main():
    parser = argparse.ArgumentParser(
        description="Convert TRTLLM eagle checkpoint to VLLM compatible one-model checkpoint."
    )
    parser.add_argument("--input", help="Path to TRTLLM eagle checkpoint.")
    parser.add_argument("--verifier", help="Name or path to the verifier model.")
    parser.add_argument("--output", help="Save path for converted vllm one-model checkpoint.")

    args = parser.parse_args()

    with open(os.path.join(args.input, "config.json"), encoding="utf-8") as f:
        original_draft_cfg = json.load(f)

    converted_cfg = convert_to_eagle3_speculator_config(
        original_draft_cfg,
        args.verifier,
    )

    # Write the converted config to the output directory
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, "config.json"), "w", encoding="utf-8") as f:
        json.dump(converted_cfg, f, indent=2, ensure_ascii=False)

    # Copy the model.safetensor file from input dir to output dir
    input_model_path = os.path.join(args.input, "model.safetensors")
    output_model_path = os.path.join(args.output, "model.safetensors")
    shutil.copyfile(input_model_path, output_model_path)


if __name__ == "__main__":
    main()
