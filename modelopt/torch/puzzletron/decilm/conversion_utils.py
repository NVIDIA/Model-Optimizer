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

import json
import os
import re
from collections import defaultdict

from safetensors.torch import load_file, save_file
from tqdm import tqdm


def convert_name(name):
    return name.replace("feed_forward", "mlp").replace("language_model.", "")


def convert_routed_experts_weight(llama_name, weight):
    assert ".experts." in llama_name, "Only use this func to convert weights of routed experts"
    llama_name_prefix = llama_name.split(".experts.")[0]
    deci_name_prefix = convert_name(llama_name_prefix)

    experts_state_dict = {}
    for i_expert, expert_weight in enumerate(weight.unbind(dim=0)):
        expert_prefix = f"{deci_name_prefix}.experts.{i_expert}"
        if "gate_up_proj" in llama_name:
            gate_weight, up_weight = expert_weight.transpose(0, 1).chunk(2, dim=0)
            experts_state_dict[f"{expert_prefix}.gate_proj.weight"] = gate_weight.contiguous()
            experts_state_dict[f"{expert_prefix}.up_proj.weight"] = up_weight.contiguous()
        elif "down_proj" in llama_name:
            down_weight = expert_weight.transpose(0, 1)
            experts_state_dict[f"{expert_prefix}.down_proj.weight"] = down_weight.contiguous()
        else:
            raise ValueError(f"Unknown expert weight: {llama_name}")

    return experts_state_dict


def get_layer_subblock(param):
    if param.startswith("model.embed_tokens."):
        return "embeddings"
    if param.startswith("lm_head.") or param == "model.norm.weight":
        return "lm_head"
    m = re.match(r"model\.layers\.(\d+)\.(.+)", param)
    if m:
        layer, suffix = m.groups()
        if suffix.startswith(("self_attn.", "input_layernorm.weight")):
            return f"block_{layer}_attention"
        elif suffix.startswith(("mlp.", "post_attention_layernorm.weight")):
            return f"block_{layer}_ffn"
    return None


def convert_model_weights_to_decilm(llama_hf_dir, output_dir, is_llama4=False):
    index_path = os.path.join(llama_hf_dir, "model.safetensors.index.json")
    single_file_path = os.path.join(llama_hf_dir, "model.safetensors")

    # Check if we have a sharded model (with index) or single file model
    if os.path.exists(index_path):
        # Sharded model - use existing logic
        with open(index_path) as f:
            index = json.load(f)
        param_to_file = index["weight_map"]
        all_param_names = list(param_to_file.keys())
    elif os.path.exists(single_file_path):
        # Single file model - create a synthetic index
        data = load_file(single_file_path)
        all_param_names = list(data.keys())
        param_to_file = dict.fromkeys(all_param_names, "model.safetensors")
    else:
        raise FileNotFoundError(
            f"Neither {index_path} nor {single_file_path} found. Cannot determine model format."
        )
    name_map = {
        name: convert_name(name)
        for name in all_param_names
        if name.startswith("language_model.") or not is_llama4
    }

    # Reverse map: file -> set of params
    file_to_params = defaultdict(set)
    for name, file in param_to_file.items():
        file_to_params[file].add(name)

    # Determine subblocks needed
    subblocks = defaultdict(list)
    for old_name, new_name in name_map.items():
        subblock = get_layer_subblock(new_name)
        if subblock:
            subblocks[subblock].append((old_name, new_name))

    # Output directory
    out_dir = os.path.join(output_dir, "subblocks_safetensors")
    os.makedirs(out_dir, exist_ok=True)

    # New weight index
    new_index = {"metadata": {"format": "pt"}, "weight_map": {}}

    # For single file models, load all data once
    if os.path.exists(single_file_path) and not os.path.exists(index_path):
        all_data = load_file(single_file_path)
    else:
        all_data = None

    for subblock, param_pairs in tqdm(subblocks.items(), desc="Processing subblocks"):
        tensors = {}

        if all_data is not None:
            # Single file model - get tensors from pre-loaded data
            for old_name, new_name in param_pairs:
                if old_name in all_data:
                    if ".experts." not in old_name:
                        tensors[new_name] = all_data[old_name]
                    else:
                        experts_state_dict = convert_routed_experts_weight(
                            old_name, all_data[old_name]
                        )
                        tensors.update(experts_state_dict)
        else:
            # Sharded model - load only needed files for this subblock
            param_files = {param_to_file[old] for old, _ in param_pairs}
            for file in param_files:
                data = load_file(os.path.join(llama_hf_dir, file))
                for old_name, new_name in param_pairs:
                    if param_to_file[old_name] == file and old_name in data:
                        if ".experts." not in old_name:
                            tensors[new_name] = data[old_name]
                        else:
                            experts_state_dict = convert_routed_experts_weight(
                                old_name, data[old_name]
                            )
                            tensors.update(experts_state_dict)

        # Save this subblock
        subblock_file = f"{subblock}.safetensors"
        save_file(tensors, os.path.join(out_dir, subblock_file))

        # Update index
        for new_name in tensors:
            new_index["weight_map"][new_name] = f"subblocks_safetensors/{subblock_file}"

    # Save new index file
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    print(f"✅ Finished saving subblocks and index to {output_dir}")
