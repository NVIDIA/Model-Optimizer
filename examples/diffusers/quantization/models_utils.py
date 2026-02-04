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

import logging
from collections.abc import Callable
from enum import Enum
from typing import Any

from diffusers import (
    DiffusionPipeline,
    FluxPipeline,
    LTXConditionPipeline,
    StableDiffusion3Pipeline,
    WanPipeline,
)
from utils import (
    filter_func_default,
    filter_func_flux_dev,
    filter_func_ltx_video,
    filter_func_wan_video,
)


class ModelType(str, Enum):
    """Supported model types."""

    SDXL_BASE = "sdxl-1.0"
    SDXL_TURBO = "sdxl-turbo"
    SD3_MEDIUM = "sd3-medium"
    SD35_MEDIUM = "sd3.5-medium"
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    LTX_VIDEO_DEV = "ltx-video-dev"
    LTX2 = "ltx-2"
    WAN22_T2V_14b = "wan2.2-t2v-14b"
    WAN22_T2V_5b = "wan2.2-t2v-5b"


def get_model_filter_func(model_type: ModelType) -> Callable[[str], bool]:
    """
    Get the appropriate filter function for a given model type.

    Args:
        model_type: The model type enum

    Returns:
        A filter function appropriate for the model type
    """
    filter_func_map = {
        ModelType.FLUX_DEV: filter_func_flux_dev,
        ModelType.FLUX_SCHNELL: filter_func_default,
        ModelType.SDXL_BASE: filter_func_default,
        ModelType.SDXL_TURBO: filter_func_default,
        ModelType.SD3_MEDIUM: filter_func_default,
        ModelType.SD35_MEDIUM: filter_func_default,
        ModelType.LTX_VIDEO_DEV: filter_func_ltx_video,
        ModelType.LTX2: filter_func_ltx_video,
        ModelType.WAN22_T2V_14b: filter_func_wan_video,
        ModelType.WAN22_T2V_5b: filter_func_wan_video,
    }

    return filter_func_map.get(model_type, filter_func_default)


# Model registry with HuggingFace model IDs
MODEL_REGISTRY: dict[ModelType, str] = {
    ModelType.SDXL_BASE: "stabilityai/stable-diffusion-xl-base-1.0",
    ModelType.SDXL_TURBO: "stabilityai/sdxl-turbo",
    ModelType.SD3_MEDIUM: "stabilityai/stable-diffusion-3-medium-diffusers",
    ModelType.SD35_MEDIUM: "stabilityai/stable-diffusion-3.5-medium",
    ModelType.FLUX_DEV: "black-forest-labs/FLUX.1-dev",
    ModelType.FLUX_SCHNELL: "black-forest-labs/FLUX.1-schnell",
    ModelType.LTX_VIDEO_DEV: "Lightricks/LTX-Video-0.9.7-dev",
    ModelType.LTX2: "Lightricks/LTX-2",
    ModelType.WAN22_T2V_14b: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ModelType.WAN22_T2V_5b: "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
}

MODEL_PIPELINE: dict[ModelType, type[DiffusionPipeline] | None] = {
    ModelType.SDXL_BASE: DiffusionPipeline,
    ModelType.SDXL_TURBO: DiffusionPipeline,
    ModelType.SD3_MEDIUM: StableDiffusion3Pipeline,
    ModelType.SD35_MEDIUM: StableDiffusion3Pipeline,
    ModelType.FLUX_DEV: FluxPipeline,
    ModelType.FLUX_SCHNELL: FluxPipeline,
    ModelType.LTX_VIDEO_DEV: LTXConditionPipeline,
    ModelType.LTX2: None,
    ModelType.WAN22_T2V_14b: WanPipeline,
    ModelType.WAN22_T2V_5b: WanPipeline,
}

# Shared dataset configurations
_SD_PROMPTS_DATASET = {
    "name": "Gustavosta/Stable-Diffusion-Prompts",
    "split": "train",
    "column": "Prompt",
}

_OPENVID_DATASET = {
    "name": "nkp37/OpenVid-1M",
    "split": "train",
    "column": "caption",
}

# Model family base configurations
_SDXL_BASE_CONFIG: dict[str, Any] = {
    "backbone": "unet",
    "dataset": _SD_PROMPTS_DATASET,
}

_SD3_BASE_CONFIG: dict[str, Any] = {
    "backbone": "transformer",
    "dataset": _SD_PROMPTS_DATASET,
}

_FLUX_BASE_CONFIG: dict[str, Any] = {
    "backbone": "transformer",
    "dataset": _SD_PROMPTS_DATASET,
    "inference_extra_args": {
        "height": 1024,
        "width": 1024,
        "guidance_scale": 3.5,
        "max_sequence_length": 512,
    },
}

_WAN_BASE_CONFIG: dict[str, Any] = {
    "backbone": "transformer",
    "dataset": _OPENVID_DATASET,
}

# Model-specific default arguments for calibration
MODEL_DEFAULTS: dict[ModelType, dict[str, Any]] = {
    ModelType.SDXL_BASE: _SDXL_BASE_CONFIG,
    ModelType.SDXL_TURBO: _SDXL_BASE_CONFIG,
    ModelType.SD3_MEDIUM: _SD3_BASE_CONFIG,
    ModelType.SD35_MEDIUM: _SD3_BASE_CONFIG,
    ModelType.FLUX_DEV: _FLUX_BASE_CONFIG,
    ModelType.FLUX_SCHNELL: _FLUX_BASE_CONFIG,
    ModelType.LTX_VIDEO_DEV: {
        "backbone": "transformer",
        "dataset": _SD_PROMPTS_DATASET,
        "inference_extra_args": {
            "height": 512,
            "width": 704,
            "num_frames": 121,
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        },
    },
    ModelType.LTX2: {
        "backbone": "transformer",
        "dataset": _SD_PROMPTS_DATASET,
        "inference_extra_args": {
            "height": 1024,
            "width": 1536,
            "num_frames": 121,
            "frame_rate": 24.0,
            "cfg_guidance_scale": 4.0,
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        },
    },
    ModelType.WAN22_T2V_14b: {
        **_WAN_BASE_CONFIG,
        "from_pretrained_extra_args": {
            "boundary_ratio": 0.875,
        },
        "inference_extra_args": {
            "height": 720,
            "width": 1280,
            "num_frames": 81,
            "fps": 16,
            "guidance_scale": 4.0,
            "guidance_scale_2": 3.0,
            "negative_prompt": (
                "vivid colors, overexposed, static, blurry details, subtitles, style, "
                "work of art, painting, picture, still, overall grayish, worst quality, "
                "low quality, JPEG artifacts, ugly, deformed, extra fingers, poorly drawn hands, "
                "poorly drawn face, deformed, disfigured, deformed limbs, fused fingers, "
                "static image, cluttered background, three legs, many people in the background, "
                "walking backwards"
            ),
        },
    },
    ModelType.WAN22_T2V_5b: {
        **_WAN_BASE_CONFIG,
        "inference_extra_args": {
            "height": 512,
            "width": 768,
            "num_frames": 81,
            "fps": 16,
            "guidance_scale": 5.0,
            "negative_prompt": (
                "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留"  # noqa: RUF001
                "，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"  # noqa: RUF001
                "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"  # noqa: RUF001
            ),
        },
    },
}


def _coerce_extra_param_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def parse_extra_params(
    kv_args: list[str], unknown_args: list[str], logger: logging.Logger
) -> dict[str, Any]:
    extra_params: dict[str, Any] = {}
    for item in kv_args:
        if "=" not in item:
            raise ValueError(f"Invalid --extra-param value: '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        extra_params[key] = _coerce_extra_param_value(value)

    i = 0
    while i < len(unknown_args):
        token = unknown_args[i]
        if token.startswith("--extra_param."):
            key = token[len("--extra_param.") :]
            value = "true"
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
                i += 1
            extra_params[key] = _coerce_extra_param_value(value)
        elif token.startswith("--extra_param"):
            raise ValueError(
                "Use --extra_param.KEY VALUE or --extra-param KEY=VALUE for extra parameters."
            )
        else:
            logger.warning("Ignoring unknown argument: %s", token)
        i += 1

    return extra_params
