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
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear
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


_MHA_QUANTIZER_NAMES = (
    "q_bmm_quantizer",
    "k_bmm_quantizer",
    "v_bmm_quantizer",
    "softmax_quantizer",
    "bmm2_output_quantizer",
)
_REQUIRED_MHA_QUANTIZER_NAMES = _MHA_QUANTIZER_NAMES[:4]
_SDXL_FP16_PROJECTION_NAMES = frozenset(("to_q", "to_k", "to_v"))


def check_conv_and_mha(backbone, if_fp4, quantize_mha):
    for name, module in backbone.named_modules():
        if isinstance(module, torch.nn.Conv1d | torch.nn.Conv2d) and if_fp4:
            nvfp4_quantizers = [
                quantizer
                for quantizer in (
                    getattr(module, "weight_quantizer", None),
                    getattr(module, "input_quantizer", None),
                )
                if isinstance(quantizer, TensorQuantizer) and quantizer.is_nvfp4_dynamic
            ]
            for quantizer in nvfp4_quantizers:
                quantizer.disable()
            if nvfp4_quantizers:
                print(f"Disabled NVFP4 Conv layer quantization for layer {name}")

        elif isinstance(module, Attention | AttentionModuleMixin):
            head_size = int(module.inner_dim / module.heads)
            if not quantize_mha or head_size % 16 != 0:
                for attr in _MHA_QUANTIZER_NAMES:
                    if hasattr(module, attr):
                        getattr(module, attr).disable()
                setattr(module, "_disable_fp8_mha", True)

                print(f"Disabled Attention layer quantization for layer {name}")
            else:
                setattr(module, "_disable_fp8_mha", False)


def _validate_finite_positive_amax(name, quantizer):
    amax = quantizer.amax
    if amax is None or amax.numel() != 1 or not torch.isfinite(amax).all() or not (amax > 0).all():
        raise ValueError(f"Quantizer '{name}' must have a finite positive calibrated amax.")


def _validate_finite_nonnegative_amax(name, quantizer):
    amax = quantizer.amax
    if amax is None or amax.numel() != 1 or not torch.isfinite(amax).all() or not (amax >= 0).all():
        raise ValueError(f"Quantizer '{name}' must have a finite nonnegative calibrated amax.")


def _validate_calibrated_fp8_quantizer(name, quantizer):
    if not quantizer.is_fp8:
        raise ValueError(f"Quantizer '{name}' must use per-tensor FP8.")
    _validate_finite_positive_amax(name, quantizer)


def validate_fp8_mha_quantizers(backbone, quantize_mha):
    """Validate that the restored or finalized FP8 MHA state matches its policy."""
    for name, module in backbone.named_modules():
        if not isinstance(module, Attention | AttentionModuleMixin):
            continue

        head_size = int(module.inner_dim / module.heads)
        mha_enabled = quantize_mha and head_size % 16 == 0
        for attr in _MHA_QUANTIZER_NAMES:
            quantizer = getattr(module, attr, None)
            if quantizer is None:
                if mha_enabled:
                    expected_state = (
                        "present and enabled"
                        if attr in _REQUIRED_MHA_QUANTIZER_NAMES
                        else "present and disabled"
                    )
                    raise ValueError(
                        f"FP8 MHA for attention '{name}' requires '{attr}' to be {expected_state}."
                    )
                continue
            if not isinstance(quantizer, TensorQuantizer):
                raise ValueError(
                    f"Attention '{name}.{attr}' must be a TensorQuantizer, got "
                    f"{type(quantizer).__name__}."
                )
            if not mha_enabled:
                if quantizer.is_enabled:
                    reason = "disabled by configuration" if not quantize_mha else "unsupported"
                    raise ValueError(
                        f"Attention '{name}.{attr}' must be disabled because FP8 MHA is {reason}."
                    )
                continue
            if attr in _REQUIRED_MHA_QUANTIZER_NAMES and not quantizer.is_enabled:
                raise ValueError(f"FP8 MHA for attention '{name}' requires '{attr}' to be enabled.")
            if attr == "bmm2_output_quantizer" and quantizer.is_enabled:
                raise ValueError(
                    f"FP8 MHA for attention '{name}' requires '{attr}' to be disabled."
                )
            if quantizer.is_enabled:
                qualified_name = f"{name}.{attr}"
                if attr == "softmax_quantizer":
                    if not quantizer.is_fp8:
                        raise ValueError(f"Quantizer '{qualified_name}' must use per-tensor FP8.")
                else:
                    _validate_calibrated_fp8_quantizer(qualified_name, quantizer)


def _validate_fp4_quantizer_placement(backbone, quantize_mha, allow_fp8_conv):
    for module_name, module in backbone.named_modules():
        for quantizer_name, quantizer in module.named_children():
            if not isinstance(quantizer, TensorQuantizer) or not quantizer.is_enabled:
                continue
            qualified_name = f"{module_name}.{quantizer_name}".lstrip(".")
            if quantizer.is_nvfp4_dynamic:
                if not isinstance(
                    module, torch.nn.Linear | RealQuantLinear
                ) or quantizer_name not in (
                    "input_quantizer",
                    "weight_quantizer",
                ):
                    raise ValueError(
                        f"Enabled NVFP4 quantizer '{qualified_name}' is only supported on Linear "
                        "input and weight quantizers."
                    )
                continue

            if quantizer.is_fp8:
                is_conv_quantizer = (
                    allow_fp8_conv
                    and isinstance(module, torch.nn.Conv2d)
                    and quantizer_name in ("input_quantizer", "weight_quantizer")
                )
                is_mha_quantizer = (
                    quantize_mha
                    and isinstance(module, Attention | AttentionModuleMixin)
                    and quantizer_name in _MHA_QUANTIZER_NAMES
                )
                if is_conv_quantizer or is_mha_quantizer:
                    continue
                raise ValueError(
                    f"Enabled FP8 quantizer '{qualified_name}' is only supported on SDXL Conv2d "
                    "input/weight quantizers or opt-in MHA quantizers."
                )

            raise ValueError(
                f"Enabled quantizer '{qualified_name}' has an unsupported format for FP4 export."
            )


def validate_nvfp4_quantizers(
    backbone, expected_block_size, quantize_mha, validate_sdxl_mixed_recipe=False
):
    """Validate the quantizer state required by NVFP4 ONNX export."""
    enabled_linear_pairs = 0
    for name, module in backbone.named_modules():
        if not isinstance(module, torch.nn.Linear | RealQuantLinear):
            continue

        input_quantizer = getattr(module, "input_quantizer", None)
        weight_quantizer = getattr(module, "weight_quantizer", None)
        if input_quantizer is None and weight_quantizer is None:
            continue
        if not isinstance(input_quantizer, TensorQuantizer) or not isinstance(
            weight_quantizer, TensorQuantizer
        ):
            raise ValueError(
                f"NVFP4 Linear '{name}' must use TensorQuantizer instances for both input and weight."
            )

        input_enabled = input_quantizer.is_enabled
        weight_enabled = weight_quantizer.is_enabled
        if validate_sdxl_mixed_recipe and name.rsplit(".", 1)[-1] in _SDXL_FP16_PROJECTION_NAMES:
            if input_enabled or weight_enabled:
                raise ValueError(
                    f"SDXL attention projection '{name}' must keep input and weight quantizers "
                    "disabled for the NVFP4 mixed recipe. Recalibrate with the current SDXL "
                    "FP4 recipe."
                )
            continue

        if not input_enabled and not weight_enabled:
            continue
        if input_enabled != weight_enabled:
            raise ValueError(
                f"NVFP4 Linear '{name}' must enable input and weight quantizers as a pair."
            )
        if not input_quantizer.is_nvfp4_dynamic or not weight_quantizer.is_nvfp4_dynamic:
            raise ValueError(
                f"NVFP4 Linear '{name}' must use dynamic E2M1 quantizers with FP8 block scales "
                "for both input and weight."
            )
        _validate_finite_nonnegative_amax(f"{name}.input_quantizer", input_quantizer)
        _validate_finite_nonnegative_amax(f"{name}.weight_quantizer", weight_quantizer)

        input_block_size = input_quantizer.block_sizes.get(-1)
        weight_block_size = weight_quantizer.block_sizes.get(-1)
        if input_block_size != expected_block_size or weight_block_size != expected_block_size:
            raise ValueError(
                f"NVFP4 Linear '{name}' requires block size {expected_block_size}; got "
                f"input={input_block_size}, weight={weight_block_size}."
            )
        enabled_linear_pairs += 1

    if enabled_linear_pairs == 0:
        raise ValueError(
            "NVFP4 quantization requires at least one enabled Linear input/weight pair."
        )

    if validate_sdxl_mixed_recipe:
        enabled_conv_pairs = 0
        for name, module in backbone.named_modules():
            if not isinstance(module, torch.nn.Conv2d):
                continue

            input_quantizer = getattr(module, "input_quantizer", None)
            weight_quantizer = getattr(module, "weight_quantizer", None)
            if input_quantizer is None and weight_quantizer is None:
                continue
            if not isinstance(input_quantizer, TensorQuantizer) or not isinstance(
                weight_quantizer, TensorQuantizer
            ):
                raise ValueError(
                    f"SDXL FP8 Conv2d '{name}' must use TensorQuantizer instances for both input "
                    "and weight."
                )

            input_enabled = input_quantizer.is_enabled
            weight_enabled = weight_quantizer.is_enabled
            if not input_enabled and not weight_enabled:
                continue
            if input_enabled != weight_enabled:
                raise ValueError(
                    f"SDXL FP8 Conv2d '{name}' must enable input and weight quantizers as a pair."
                )
            _validate_calibrated_fp8_quantizer(f"{name}.input_quantizer", input_quantizer)
            _validate_calibrated_fp8_quantizer(f"{name}.weight_quantizer", weight_quantizer)
            enabled_conv_pairs += 1

        if enabled_conv_pairs == 0:
            raise ValueError(
                "SDXL NVFP4 quantization requires at least one enabled calibrated FP8 Conv2d "
                "input/weight pair."
            )

    validate_fp8_mha_quantizers(backbone, quantize_mha)
    _validate_fp4_quantizer_placement(backbone, quantize_mha, validate_sdxl_mixed_recipe)


def filter_func_ltx_video(name: str) -> bool:
    """Filter function specifically for LTX-Video models."""
    pattern = re.compile(
        r".*(proj_in|time_embed|caption_projection|proj_out|patchify_proj|adaln_single|transformer_blocks\.(0|1|2|45|46|47)\.).*"
    )
    return pattern.match(name) is not None


def filter_func_flux_dev(name: str) -> bool:
    """Filter function specifically for Flux-dev models."""
    pattern = re.compile(
        r"(proj_out.*|.*(time_text_embed|context_embedder|x_embedder|norm_out|time_guidance_embed|stream_modulation).*)"
    )
    return pattern.match(name) is not None


def filter_func_ltx2_vae(name: str) -> bool:
    """Filter for LTX-2 VAE: keeps only conv1/conv2 in up_blocks resnets."""
    keep = re.compile(r".*up_blocks\.\d+\.resnets\.\d+\.conv[12](?:\.|$)")
    return not keep.match(name)


def filter_func_wan_vae(name: str) -> bool:
    """Filter for Wan 2.2 VAE: keeps only conv1/conv2 in resnet blocks."""
    keep = re.compile(
        r".*(down_blocks\.\d+\.(?:resnets\.\d+\.)?conv[12]"
        r"|mid_block\.resnets\.\d+\.conv[12]"
        r"|up_blocks\.\d+\.resnets\.\d+\.conv[12])(?:\.|$)"
    )
    return not keep.match(name)


def filter_func_wan_video(name: str) -> bool:
    """Filter function specifically for WAN-Video models."""
    pattern = re.compile(
        r".*(patch_embedding|condition_embedder|proj_out|blocks\.(0|1|2|37|38|39)\.).*"
    )
    return pattern.match(name) is not None


# Qwen-Image's transformer has 60 ``transformer_blocks``. The recipe quantizes
# only those blocks while keeping the first two and last two -- and everything
# outside ``transformer_blocks`` -- in original precision. The model-agnostic,
# config-driven form of this recipe (deriving the block count from the model)
# lives in quantize.py; this name-only filter covers the plain FP8/NVFP4 path
# for the full 60-block Qwen-Image transformer.
QWEN_IMAGE_NUM_TRANSFORMER_BLOCKS = 60
_QWEN_IMAGE_BLOCK_RE = re.compile(r"(?:^|\.)transformer_blocks\.(\d+)(?:\.|$)")


def filter_func_qwen_image(name: str) -> bool:
    """Filter function specifically for Qwen-Image models.

    Returns ``True`` for modules to keep in original precision (quantization
    disabled): everything outside ``transformer_blocks``, plus the first two and
    last two transformer blocks.
    """
    match = _QWEN_IMAGE_BLOCK_RE.search(name)
    if match is None:
        return True
    block_idx = int(match.group(1))
    return block_idx < 2 or block_idx >= QWEN_IMAGE_NUM_TRANSFORMER_BLOCKS - 2


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
