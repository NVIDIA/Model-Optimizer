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
import gc
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from utils import get_original_huggingface_quant_method

from modelopt.torch.quantization.qtensor import MXFP4QTensor


def _to_oai_mxfp4_weight_only(model, block_size=32):
    new_state_dict = {}

    for name, param in model.state_dict().items():
        # Only convert experts weights, skip bias and other modules
        if "experts" in name and "bias" not in name:
            param = param.transpose(-1, -2).contiguous()
            quantized_tensors = []
            scales_tensors = []
            for expert in param:
                quantized, scales = MXFP4QTensor.quantize(expert, block_size=block_size)
                quantized_tensors.append(quantized._quantized_data)
                scales_tensors.append(scales)
            quantized = torch.stack(quantized_tensors)
            scales = torch.stack(scales_tensors)

            shape = quantized.shape
            # Add converted weights and scales to state_dict
            new_state_dict.update(
                {
                    f"{name}_blocks": quantized.view(shape[0], shape[1], -1, block_size // 2).cpu(),
                    f"{name}_scales": scales.view(shape[0], shape[1], -1).cpu(),
                }
            )
            # Free GPU memory immediately after processing each parameter
            del param, quantized, scales
            torch.cuda.empty_cache()
            gc.collect()
        else:
            new_state_dict[name] = param

    return new_state_dict


def convert_and_save(model, tokenizer, output_path: str):
    # Convert weights to mxfp4
    quantized_state_dict = _to_oai_mxfp4_weight_only(model)

    # Save converted weights
    model.save_pretrained(output_path, state_dict=quantized_state_dict)

    # Save config
    config_path = os.path.join(output_path, "config.json")
    config_data = {}

    with open(config_path) as file:
        config_data = json.load(file)

    config_data["quantization_config"] = {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "mxfp4",
    }

    config_data.pop("torch_dtype", None)

    with open(config_path, "w") as file:
        json.dump(config_data, file, indent=4)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)


def create_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", type=str, help="path to the fake-quantized model from QAT.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Set trust_remote_code for Huggingface models and tokenizers",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="path to the LoRA-QAT adapter weights. You can only specify lora_path or model_path, not both.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="path to the base model used for LoRA-QAT. Only used if lora_path is specified.",
    )
    parser.add_argument(
        "--output_path", type=str, required=True, help="location to save converted model."
    )

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    kwargs = {"device_map": "auto", "dtype": "auto", "trust_remote_code": args.trust_remote_code}
    if args.lora_path:
        assert args.model_path is None, "You can only specify lora_path or model_path, not both."
        model_path = args.base_path
        if get_original_huggingface_quant_method(args.base_path) == "mxfp4":
            kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
    else:
        model_path = args.model_path

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if args.lora_path:
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()  # Merge LoRA-QAT adapter weights to base model
        torch.cuda.empty_cache()
        gc.collect()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=args.trust_remote_code)

    # Quantize and save model
    convert_and_save(model, tokenizer, args.output_path)
