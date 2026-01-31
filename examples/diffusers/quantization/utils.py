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

import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from diffusers.models.attention_processor import Attention, AttnProcessor
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
from diffusers.utils import load_image

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.plugins.diffusion.diffusers import AttentionModuleMixin

USE_PEFT = True
try:
    from peft.tuners.lora.layer import Conv2d as PEFTLoRAConv2d
    from peft.tuners.lora.layer import Linear as PEFTLoRALinear
except ModuleNotFoundError:
    USE_PEFT = False


# Model-specific filter functions for quantization
def filter_func_default(name: str) -> bool:
    """Default filter function for general models."""
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding|pos_embed|time_text_embed|context_embedder|norm_out|x_embedder).*"
    )
    return pattern.match(name) is not None


def check_conv_and_mha(backbone, if_fp4, quantize_mha):
    for name, module in backbone.named_modules():
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) and if_fp4:
            module.weight_quantizer.disable()
            module.input_quantizer.disable()

            print(f"Disabled NVFP4 Conv layer quantization for layer {name}")

        elif isinstance(module, (Attention, AttentionModuleMixin)):
            head_size = int(module.inner_dim / module.heads)
            if not quantize_mha or head_size % 16 != 0:
                module.q_bmm_quantizer.disable()
                module.k_bmm_quantizer.disable()
                module.v_bmm_quantizer.disable()
                module.softmax_quantizer.disable()
                module.bmm2_output_quantizer.disable()
                setattr(module, "_disable_fp8_mha", True)

                print(f"Disabled Attention layer quantization for layer {name}")
            else:
                setattr(module, "_disable_fp8_mha", False)


def filter_func_ltx_video(name: str) -> bool:
    """Filter function specifically for LTX-Video models."""
    pattern = re.compile(
        r".*(proj_in|time_embed|caption_projection|proj_out|patchify_proj|adaln_single).*"
    )
    return pattern.match(name) is not None


def filter_func_flux_dev(name: str) -> bool:
    """Filter function specifically for Flux-dev models."""
    pattern = re.compile(r"(proj_out.*|.*(time_text_embed|context_embedder|x_embedder|norm_out).*)")
    return pattern.match(name) is not None


def filter_func_wan_video(name: str) -> bool:
    """Filter function specifically for LTX-Video models."""
    pattern = re.compile(r".*(patch_embedding|condition_embedder|proj_out).*")
    return pattern.match(name) is not None


def load_calib_prompts(
    batch_size,
    calib_data_path: str | Path = "Gustavosta/Stable-Diffusion-Prompts",
    split="train",
    column="Prompt",
) -> list[list[str]]:
    prompt_list: list[str] = []
    if isinstance(calib_data_path, Path):
        with open(calib_data_path) as f:
            prompt_list = f.readlines()
    else:
        dataset = load_dataset(calib_data_path)
        prompt_list = list(dataset[split][column])
    return [prompt_list[i : i + batch_size] for i in range(0, len(prompt_list), batch_size)]


def load_calib_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = load_image(img_path)
            if image is not None:
                images.append(image)
    return images


def set_fmha(unet):
    for name, module in unet.named_modules():
        if isinstance(module, Attention):
            module.set_processor(AttnProcessor())


def check_lora(unet):
    for name, module in unet.named_modules():
        if isinstance(module, (LoRACompatibleConv, LoRACompatibleLinear)):
            assert module.lora_layer is None, (
                f"To quantize {name}, LoRA layer should be fused/merged. Please"
                " fuse the LoRA layer before quantization."
            )
        elif USE_PEFT and isinstance(module, (PEFTLoRAConv2d, PEFTLoRALinear)):
            assert module.merged, (
                f"To quantize {name}, LoRA layer should be fused/merged. Please"
                " fuse the LoRA layer before quantization."
            )


def fp8_mha_disable(backbone, quantized_mha_output: bool = True):
    def mha_filter_func(name):
        pattern = re.compile(
            r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer).*"
            if quantized_mha_output
            else r".*(q_bmm_quantizer|k_bmm_quantizer|v_bmm_quantizer|softmax_quantizer|bmm2_output_quantizer).*"
        )
        return pattern.match(name) is not None

    if hasattr(F, "scaled_dot_product_attention"):
        mtq.disable_quantizer(backbone, mha_filter_func)
