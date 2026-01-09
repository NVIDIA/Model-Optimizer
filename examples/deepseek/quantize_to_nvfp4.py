# Adapted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/fp8_cast_bf16.py
# MIT License

# Copyright (c) 2023 DeepSeek

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import argparse
import glob
import json
import os
from typing import Any

import torch
from ds_kernel import weight_dequant
from safetensors.torch import load_file, save_file
from tqdm import tqdm

from modelopt.torch.quantization.qtensor import NVFP4QTensor


def _remap_key(key_dict: dict[str, Any]):
    # renaming the module to match HF modeling
    # The order matters here.
    mappig = {
        "ffn": "mlp",
        "w1": "gate_proj",
        "w2": "down_proj",
        "w3": "up_proj",
        "attn": "self_attn",
        "wq_a": "q_a_proj",
        "wq_b": "q_b_proj",
        "wq": "q_proj",
        "wkv_a": "kv_a_proj_with_mqa",
        "wkv_b": "kv_b_proj",
        "wo": "o_proj",
        "head": "lm_head",
    }

    new_dict = {}
    for k, v in key_dict.items():
        new_key = k.replace("layers", "model.layers")

        for original_pattern, replace_pattern in mappig.items():
            new_key = new_key.replace(original_pattern, replace_pattern)

        new_dict[new_key] = v

    key_dict.clear()
    key_dict.update(new_dict)


def remove_quantization_config_from_original_config(export_dir: str) -> None:
    """Remove `quantization_config` from exported HF `config.json`.

    Assumes the exported checkpoint directory has a `config.json` containing `quantization_config`.
    """
    config_path = os.path.join(export_dir, "config.json")
    with open(config_path) as f:
        cfg = json.load(f)
    del cfg["quantization_config"]
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
        f.write("\n")


def load_and_preprocess_state_dict(modelopt_state_root, world_size=8):
    state_dict_list = [
        torch.load(f"{modelopt_state_root}/amax_dict_rank{rank}-mp{world_size}.pt")
        for rank in range(world_size)
    ]

    # calculate the max across all TP
    merged_state_dict = state_dict_list[0]
    for rank in range(world_size):
        for key, amax in state_dict_list[rank].items():
            if key in merged_state_dict.items():
                amax = torch.max(amax, merged_state_dict[key].to(amax.device))
            merged_state_dict[key] = amax

    _remap_key(merged_state_dict)

    # set amax for modules to be fused and make sure they share the same input
    for key, amax in merged_state_dict.items():
        if "up_proj" in key:
            gate_proj_key = key.replace("up_proj", "gate_proj")
            if "weight_quantizer" in key:
                fused_amax = torch.max(amax, merged_state_dict[gate_proj_key])
                merged_state_dict[key] = fused_amax
                merged_state_dict[gate_proj_key] = fused_amax
            elif "input_quantizer" in key:
                assert amax == merged_state_dict[gate_proj_key]
            else:
                raise NotImplementedError

    return merged_state_dict


def process_quant_config(quant_config_path: str, save_path: str) -> dict[str, Any]:
    with open(quant_config_path) as f:
        quant_config = json.load(f)

    if "exclude_modules" in quant_config["quantization"]:
        exclude_dict = dict.fromkeys(quant_config["quantization"]["exclude_modules"])
        _remap_key(exclude_dict)
        quant_config["quantization"]["exclude_modules"] = list(exclude_dict.keys())

    per_layer_quant_config = {}
    if "quantized_layers" in quant_config["quantization"]:
        _remap_key(quant_config["quantization"]["quantized_layers"])
        per_layer_quant_config = quant_config["quantization"]["quantized_layers"]

    with open(save_path, "w") as f:
        json.dump(quant_config, f, indent=4)

    return per_layer_quant_config


def find_safetensors_files(directory: str):
    """Recursively finds all `.safetensors` files in the specified directory."""
    safetensors_files = sorted(
        glob.glob(os.path.join(directory, "**", "*.safetensors"), recursive=True)
    )

    return safetensors_files


# Adopted from https://github.com/deepseek-ai/DeepSeek-V3/blob/2f7b80eecebf3d1c84da5a0d465f6639ea175012/inference/fp8_cast_bf16.py
def convert_fp8_ckpt_to_nvfp4(
    renamed_state_dict,
    fp8_root,
    save_root,
    per_layer_quant_config,
):
    def amax_to_nvfp4_scaling_factor_2(amax):
        return amax.float() / (6.0 * 448.0)

    def amax_to_fp8_scaling_factor(amax):
        return amax.float() / 448.0

    torch.set_default_dtype(torch.bfloat16)
    model_index_file = os.path.join(fp8_root, "model.safetensors.index.json")
    os.makedirs(save_root, exist_ok=True)
    with open(model_index_file) as f:
        model_index = json.load(f)
    weight_map = model_index["weight_map"]

    # Cache for loaded safetensor files
    loaded_files = {}
    fp8_weight_names = []

    # Helper function to get tensor from the correct file
    def get_tensor(tensor_name):
        file_name = weight_map[tensor_name]
        if file_name not in loaded_files:
            file_path = os.path.join(fp8_root, file_name)
            loaded_files[file_name] = load_file(file_path, device="cuda")
        return loaded_files[file_name][tensor_name]

    all_safetensor_files = find_safetensors_files(fp8_root)
    for safetensor_file in tqdm(all_safetensor_files, desc="Quantizing"):
        # convert FP8 safetensor to BF16
        file_name = os.path.basename(safetensor_file)
        current_state_dict = load_file(safetensor_file, device="cuda")
        loaded_files[file_name] = current_state_dict

        bf16_state_dict = {}
        for key, item in current_state_dict.items():
            # insert inv
            if key.endswith("_scale_inv"):
                continue
            elif item.element_size() == 1:  # FP8 weight
                scale_inv_name = f"{key}_scale_inv"
                try:
                    # Get scale_inv from the correct file
                    scale_inv = get_tensor(scale_inv_name)
                    fp8_weight_names.append(key)
                    bf16_state_dict[key] = weight_dequant(item, scale_inv)
                except KeyError:
                    print(f"Warning: Missing scale_inv tensor for {key}, skipping conversion")
                    bf16_state_dict[key] = item
            else:
                bf16_state_dict[key] = item

        new_dict = {}
        for key, item in bf16_state_dict.items():
            if "weight" in key:
                weight = item
                amax_key = key + "_quantizer._amax"
                layer_name = key.replace(".weight", "")
                input_scale_key = layer_name + ".input_quantizer._amax"

                if amax_key in renamed_state_dict:
                    # default quant is NVFP4
                    is_nvfp4 = (
                        not per_layer_quant_config
                        or per_layer_quant_config.get(layer_name, {}).get("quant_algo", None)
                        == "NVFP4"
                    )

                    is_per_tensor_fp8 = (
                        per_layer_quant_config
                        and per_layer_quant_config.get(layer_name, {}).get("quant_algo", None)
                        == "FP8"
                    )

                    if is_nvfp4:
                        weight_scaling_factor_2 = amax_to_nvfp4_scaling_factor_2(
                            renamed_state_dict[amax_key]
                        ).to(weight.device)
                        input_scaling_factor = amax_to_nvfp4_scaling_factor_2(
                            renamed_state_dict[input_scale_key]
                        ).to(weight.device)
                        quantized_weight, weight_scaling_factor, weight_scaling_factor_2 = (
                            NVFP4QTensor.quantize(
                                weight, 16, None, weight_scaling_factor_2, try_tensorrt=False
                            )
                        )

                        # adding input_scale, weight_scaling_factor,
                        new_dict[key] = quantized_weight._quantized_data
                        new_dict[layer_name + ".input_scale"] = input_scaling_factor
                        new_dict[key + "_scale"] = weight_scaling_factor
                        new_dict[key + "_scale_2"] = weight_scaling_factor_2
                    elif is_per_tensor_fp8:
                        weight_scaling_factor = amax_to_fp8_scaling_factor(
                            renamed_state_dict[amax_key]
                        ).to(weight.device)
                        input_scaling_factor = amax_to_fp8_scaling_factor(
                            renamed_state_dict[input_scale_key]
                        ).to(weight.device)

                        quantized_weight = (weight / weight_scaling_factor).to(torch.float8_e4m3fn)
                        new_dict[key] = quantized_weight
                        new_dict[layer_name + ".input_scale"] = input_scaling_factor
                        new_dict[key + "_scale"] = weight_scaling_factor
                    else:
                        raise NotImplementedError("Quantization algorithm not supported")
                    continue

            new_dict[key] = item

        save_file(new_dict, os.path.join(save_root, file_name))

        # Memory management: keep only the 2 most recently used files
        while len(loaded_files) > 2:
            oldest_file = next(iter(loaded_files))
            del loaded_files[oldest_file]
            torch.cuda.empty_cache()

    # Update model index
    new_model_index_file = os.path.join(save_root, "model.safetensors.index.json")
    for weight_name in fp8_weight_names:
        scale_inv_name = f"{weight_name}_scale_inv"
        if scale_inv_name in weight_map:
            weight_map.pop(scale_inv_name)
    with open(new_model_index_file, "w") as f:
        json.dump({"metadata": {}, "weight_map": weight_map}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--amax_path", type=str, required=True, help="path to the expoted amax.")
    parser.add_argument(
        "--fp4_path", type=str, required=True, help="path to save the fp4 checkpoint."
    )
    parser.add_argument("--fp8_hf_path", type=str, required=True, help="fp8 hf ckpt.")
    parser.add_argument("--world_size", type=int, required=True, help="world size used by ptq.")
    args = parser.parse_args()

    per_layer_quant_config = process_quant_config(
        quant_config_path=os.path.join(args.amax_path, "hf_quant_config.json"),
        save_path=os.path.join(args.fp4_path, "hf_quant_config.json"),
    )

    renamed_state_dict = load_and_preprocess_state_dict(
        modelopt_state_root=args.amax_path,
        world_size=args.world_size,
    )
    convert_fp8_ckpt_to_nvfp4(
        renamed_state_dict,
        fp8_root=args.fp8_hf_path,
        save_root=args.fp4_path,
        per_layer_quant_config=per_layer_quant_config,
    )

    remove_quantization_config_from_original_config(args.fp4_path)
