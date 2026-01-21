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

import argparse
import logging
import sys
import time as time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from config import (
    FP8_DEFAULT_CONFIG,
    INT8_DEFAULT_CONFIG,
    FLUX_DEV_NVFP4_MIXED_CONFIG,
    NVFP4_DEFAULT_CONFIG,
    NVFP4_FP8_MHA_CONFIG,
    reset_set_int8_config,
    set_quant_config_attr,
)

# This is a workaround for making the onnx export of models that use the torch RMSNorm work. We will
# need to move on to use dynamo based onnx export to properly fix the problem. The issue has been hit
# by both external users https://github.com/NVIDIA/Model-Optimizer/issues/262, and our
# internal users from MLPerf Inference.
#
if __name__ == "__main__":
    from diffusers.models.normalization import RMSNorm as DiffuserRMSNorm

    torch.nn.RMSNorm = DiffuserRMSNorm
    torch.nn.modules.normalization.RMSNorm = DiffuserRMSNorm

from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    FluxPipeline,
    FluxTransformer2DModel,
    LTXConditionPipeline,
    LTXLatentUpsamplePipeline,
    StableDiffusion3Pipeline,
    WanPipeline,
)
from diffusers.models import SD3Transformer2DModel as StableDiffusion3Transformer2DModel
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
)
from onnx_utils.export import generate_fp8_scales, modelopt_export_sd
from tqdm import tqdm
from utils import (
    check_conv_and_mha,
    check_lora,
    filter_func_default,
    filter_func_ltx_video,
    filter_func_wan_video,
    load_calib_prompts,
)
from save_quantized_safetensors import save_quantized_safetensors

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.export import export_hf_checkpoint


class ModelType(str, Enum):
    """Supported model types."""

    SDXL_BASE = "sdxl-1.0"
    SDXL_TURBO = "sdxl-turbo"
    SD3_MEDIUM = "sd3-medium"
    SD35_MEDIUM = "sd3.5-medium"
    FLUX_DEV = "flux-dev"
    FLUX_SCHNELL = "flux-schnell"
    LTX_VIDEO_DEV = "ltx-video-dev"
    WAN22_T2V = "wan2.2-t2v-14b"


class DataType(str, Enum):
    """Supported data types for model loading."""

    HALF = "Half"
    BFLOAT16 = "BFloat16"
    FLOAT = "Float"

    @property
    def torch_dtype(self) -> torch.dtype:
        return self._dtype_map[self.value]


DataType._dtype_map = {
    DataType.HALF: torch.float16,
    DataType.BFLOAT16: torch.bfloat16,
    DataType.FLOAT: torch.float32,
}


class QuantFormat(str, Enum):
    """Supported quantization formats."""

    INT8 = "int8"
    FP8 = "fp8"
    FP4 = "fp4"
    FP4_MIXED_DEV = "fp4_mixed_dev"


class QuantAlgo(str, Enum):
    """Supported quantization algorithms."""

    MAX = "max"
    SVDQUANT = "svdquant"
    SMOOTHQUANT = "smoothquant"


class CollectMethod(str, Enum):
    """Calibration collection methods."""

    GLOBAL_MIN = "global_min"
    MIN_MAX = "min-max"
    MIN_MEAN = "min-mean"
    MEAN_MAX = "mean-max"
    DEFAULT = "default"


def get_model_filter_func(model_type: ModelType) -> Callable[[str], bool]:
    """
    Get the appropriate filter function for a given model type.

    Args:
        model_type: The model type enum

    Returns:
        A filter function appropriate for the model type
    """
    filter_func_map = {
        ModelType.FLUX_DEV: filter_func_default,
        ModelType.FLUX_SCHNELL: filter_func_default,
        ModelType.SDXL_BASE: filter_func_default,
        ModelType.SDXL_TURBO: filter_func_default,
        ModelType.SD3_MEDIUM: filter_func_default,
        ModelType.SD35_MEDIUM: filter_func_default,
        ModelType.LTX_VIDEO_DEV: filter_func_ltx_video,
        ModelType.WAN22_T2V: filter_func_wan_video,
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
    ModelType.WAN22_T2V: "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
}

MODEL_PIPELINE: dict[ModelType, type[DiffusionPipeline]] = {
    ModelType.SDXL_BASE: DiffusionPipeline,
    ModelType.SDXL_TURBO: DiffusionPipeline,
    ModelType.SD3_MEDIUM: StableDiffusion3Pipeline,
    ModelType.SD35_MEDIUM: StableDiffusion3Pipeline,
    ModelType.FLUX_DEV: FluxPipeline,
    ModelType.FLUX_SCHNELL: FluxPipeline,
    ModelType.LTX_VIDEO_DEV: LTXConditionPipeline,
    ModelType.WAN22_T2V: WanPipeline,
}

# Model-specific default arguments for calibration
MODEL_DEFAULTS: dict[ModelType, dict[str, Any]] = {
    ModelType.SDXL_BASE: {
        "backbone": "unet",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
    },
    ModelType.SDXL_TURBO: {
        "backbone": "unet",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
    },
    ModelType.SD3_MEDIUM: {
        "backbone": "transformer",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
    },
    ModelType.SD35_MEDIUM: {
        "backbone": "transformer",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
    },
    ModelType.FLUX_DEV: {
        "backbone": "transformer",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
        "inference_extra_args": {
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "max_sequence_length": 512,
        },
    },
    ModelType.FLUX_SCHNELL: {
        "backbone": "transformer",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
        "inference_extra_args": {
            "height": 1024,
            "width": 1024,
            "guidance_scale": 3.5,
            "max_sequence_length": 512,
        },
    },
    ModelType.LTX_VIDEO_DEV: {
        "backbone": "transformer",
        "dataset": {
            "name": "Gustavosta/Stable-Diffusion-Prompts",
            "split": "train",
            "column": "Prompt",
        },
        "inference_extra_args": {
            "height": 512,
            "width": 704,
            "num_frames": 121,
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
        },
    },
    ModelType.WAN22_T2V: {
        "backbone": "transformer",
        "dataset": {"name": "nkp37/OpenVid-1M", "split": "train", "column": "caption"},
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
}


@dataclass
class QuantizationConfig:
    """Configuration for model quantization."""

    format: QuantFormat = QuantFormat.INT8
    algo: QuantAlgo = QuantAlgo.MAX
    percentile: float = 1.0
    collect_method: CollectMethod = CollectMethod.DEFAULT
    alpha: float = 1.0  # SmoothQuant alpha
    lowrank: int = 32  # SVDQuant lowrank
    quantize_mha: bool = False
    compress: bool = False

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.format == QuantFormat.FP8 and self.collect_method != CollectMethod.DEFAULT:
            raise NotImplementedError("Only 'default' collect method is implemented for FP8.")
        if self.quantize_mha and self.format == QuantFormat.INT8:
            raise ValueError("MHA quantization is only supported for FP8, not INT8.")
        if self.compress and self.format == QuantFormat.INT8:
            raise ValueError("Compression is only supported for FP8 and FP4, not INT8.")


@dataclass
class CalibrationConfig:
    """Configuration for calibration process."""

    prompts_dataset: dict | Path
    batch_size: int = 2
    calib_size: int = 128
    n_steps: int = 30

    def validate(self) -> None:
        """Validate calibration configuration."""
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive.")
        if self.calib_size <= 0:
            raise ValueError("Calibration size must be positive.")
        if self.n_steps <= 0:
            raise ValueError("Number of steps must be positive.")

    @property
    def num_batches(self) -> int:
        """Calculate number of calibration batches."""
        return self.calib_size // self.batch_size


@dataclass
class ModelConfig:
    """Configuration for model loading and inference."""

    model_type: ModelType = ModelType.FLUX_DEV
    model_dtype: dict[str, torch.dtype] = field(default_factory=lambda: {"default": torch.float16})
    backbone: str = ""
    trt_high_precision_dtype: DataType = DataType.HALF
    override_model_path: Path | None = None
    vae_path: Path | None = None  # Local VAE path (optional)
    text_encoder_path: Path | None = None  # Local text encoder path (optional)
    text_encoder_2_path: Path | None = None  # Local text encoder 2 path (optional, for FLUX/SD3)
    cpu_offloading: bool = False
    ltx_skip_upsampler: bool = False  # Skip upsampler for LTX-Video (faster calibration)

    @property
    def model_path(self) -> str:
        """Get the model path (override or default)."""
        if self.override_model_path:
            return str(self.override_model_path)
        return MODEL_REGISTRY[self.model_type]


@dataclass
class ExportConfig:
    """Configuration for model export."""

    quantized_torch_ckpt_path: Path | None = None
    onnx_dir: Path | None = None
    hf_ckpt_dir: Path | None = None
    restore_from: Path | None = None
    save_safetensors: bool = True  # NEW: Save SafeTensors directly

    def validate(self) -> None:
        """Validate export configuration."""
        if self.restore_from and not self.restore_from.exists():
            raise FileNotFoundError(f"Restore checkpoint not found: {self.restore_from}")

        if self.quantized_torch_ckpt_path:
            parent_dir = self.quantized_torch_ckpt_path.parent
            if not parent_dir.exists():
                parent_dir.mkdir(parents=True, exist_ok=True)

        if self.onnx_dir and not self.onnx_dir.exists():
            self.onnx_dir.mkdir(parents=True, exist_ok=True)

        if self.hf_ckpt_dir and not self.hf_ckpt_dir.exists():
            self.hf_ckpt_dir.mkdir(parents=True, exist_ok=True)


def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        verbose: Enable verbose logging

    Returns:
        Configured logger instance
    """
    log_level = logging.DEBUG if verbose else logging.INFO

    # Create custom formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Set up console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    logger.addHandler(console_handler)

    # Optionally reduce noise from other libraries
    logging.getLogger("diffusers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)

    return logger


class PipelineManager:
    """Manages diffusion pipeline creation and configuration."""

    def __init__(self, config: ModelConfig, logger: logging.Logger):
        """
        Initialize pipeline manager.

        Args:
            config: Model configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.pipe: DiffusionPipeline | None = None
        self.pipe_upsample: LTXLatentUpsamplePipeline | None = None  # For LTX-Video upsampling

    @staticmethod
    def create_pipeline_from(
        model_type: ModelType,
        torch_dtype: torch.dtype | dict[str, str | torch.dtype] = torch.bfloat16,
        override_model_path: str | None = None,
    ) -> DiffusionPipeline:
        """
        Create and return an appropriate pipeline based on configuration.

        Returns:
            Configured diffusion pipeline

        Raises:
            ValueError: If model type is unsupported
        """
        try:
            model_id = (
                MODEL_REGISTRY[model_type] if override_model_path is None else override_model_path
            )
            is_safetensors_file = model_id.lower().endswith('.safetensors')

            # If the path is a .safetensors file, we need to handle it differently
            if is_safetensors_file:
                # Load base model from HuggingFace
                base_model_id = MODEL_REGISTRY[model_type]
                pipe = MODEL_PIPELINE[model_type].from_pretrained(
                    base_model_id,
                    torch_dtype=torch_dtype,
                    **MODEL_DEFAULTS[model_type].get("from_pretrained_extra_args", {}),
                )

                # Load custom transformer from safetensors file
                if model_type in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL]:
                    dtype = torch_dtype.get("transformer", torch_dtype.get("default")) if isinstance(torch_dtype, dict) else torch_dtype
                    # Create transformer with hardcoded config to avoid downloading from HuggingFace
                    # Use meta device to reduce RAM usage during initialization
                    with torch.device('meta'):
                        transformer = FluxTransformer2DModel(
                            attention_head_dim=128,
                            guidance_embeds=True,
                            in_channels=64,
                            joint_attention_dim=4096,
                            num_attention_heads=24,
                            num_layers=19,
                            num_single_layers=38,
                            patch_size=1,
                            pooled_projection_dim=768,
                        )
                    
                    # Load weights from safetensors file
                    from safetensors.torch import load_file
                    state_dict = load_file(model_id)
                    transformer = transformer.to_empty(device='cpu')
                    transformer.load_state_dict(state_dict, strict=False, assign=True)
                    del state_dict
                    transformer = transformer.to(dtype)
                    
                    pipe.transformer = transformer
                elif model_type in [ModelType.SD3_MEDIUM, ModelType.SD35_MEDIUM]:
                    dtype = torch_dtype.get("transformer", torch_dtype.get("default")) if isinstance(torch_dtype, dict) else torch_dtype
                    transformer = StableDiffusion3Transformer2DModel.from_single_file(
                        model_id,
                        torch_dtype=dtype,
                    )
                    pipe.transformer = transformer
                else:
                    raise ValueError(
                        f"Loading from single .safetensors file is not supported for {model_type.value}. "
                        "Please provide a directory with the full model or use a HuggingFace model ID."
                    )
            else:
                # Normal loading from directory or HuggingFace repo
                pipe = MODEL_PIPELINE[model_type].from_pretrained(
                    model_id,
                    torch_dtype=torch_dtype,
                    use_safetensors=True,
                    **MODEL_DEFAULTS[model_type].get("from_pretrained_extra_args", {}),
                )

            pipe.set_progress_bar_config(disable=True)
            return pipe
        except Exception as e:
            raise e

    def create_pipeline(self) -> DiffusionPipeline:
        """
        Create and return an appropriate pipeline based on configuration.

        Returns:
            Configured diffusion pipeline

        Raises:
            ValueError: If model type is unsupported
        """
        self.logger.info(f"Creating pipeline for {self.config.model_type.value}")
        self.logger.info(f"Model path: {self.config.model_path}")
        self.logger.info(f"Data type: {self.config.model_dtype}")

        try:
            model_path = self.config.model_path
            is_safetensors_file = model_path.lower().endswith('.safetensors')
            has_local_components = (
                self.config.vae_path or
                self.config.text_encoder_path or
                self.config.text_encoder_2_path
            )

            # If the path is a .safetensors file, we need to handle it differently
            if is_safetensors_file:
                self.logger.info(f"Detected single .safetensors file: {model_path}")

                # Check if user provided local VAE/text encoders
                if has_local_components:
                    self.logger.info("Loading pipeline components from local paths...")
                    self.pipe = self._load_pipeline_from_components(model_path)
                else:
                    self.logger.info("Loading base model from HuggingFace and replacing transformer...")
                    # Load base model from HuggingFace
                    base_model_id = MODEL_REGISTRY[self.config.model_type]
                    self.logger.info(f"Loading base pipeline from: {base_model_id}")

                    self.pipe = MODEL_PIPELINE[self.config.model_type].from_pretrained(
                        base_model_id,
                        torch_dtype=self.config.model_dtype,
                        **MODEL_DEFAULTS[self.config.model_type].get("from_pretrained_extra_args", {}),
                    )

                    # Load custom transformer from safetensors file
                    self.logger.info(f"Loading custom transformer from: {model_path}")
                    if self.config.model_type in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL]:
                        # Create transformer with hardcoded config to avoid downloading from HuggingFace
                        # Use meta device to reduce RAM usage during initialization
                        with torch.device('meta'):
                            transformer = FluxTransformer2DModel(
                                attention_head_dim=128,
                                guidance_embeds=True,
                                in_channels=64,
                                joint_attention_dim=4096,
                                num_attention_heads=24,
                                num_layers=19,
                                num_single_layers=38,
                                patch_size=1,
                                pooled_projection_dim=768,
                            )
                        
                        # Load weights from safetensors file
                        from safetensors.torch import load_file
                        state_dict = load_file(model_path)
                        transformer = transformer.to_empty(device='cpu')
                        transformer.load_state_dict(state_dict, strict=False, assign=True)
                        del state_dict
                        transformer = transformer.to(self.config.model_dtype.get("transformer", self.config.model_dtype.get("default")))
                        
                        self.pipe.transformer = transformer
                        self.logger.info("Successfully replaced FLUX transformer with custom weights")
                    elif self.config.model_type in [ModelType.SD3_MEDIUM, ModelType.SD35_MEDIUM]:
                        transformer = StableDiffusion3Transformer2DModel.from_single_file(
                            model_path,
                            torch_dtype=self.config.model_dtype.get("transformer", self.config.model_dtype.get("default")),
                        )
                        self.pipe.transformer = transformer
                        self.logger.info("Successfully replaced SD3 transformer with custom weights")
                    else:
                        raise ValueError(
                            f"Loading from single .safetensors file is not supported for {self.config.model_type.value}. "
                            "Please provide a directory with the full model or use a HuggingFace model ID."
                        )
            else:
                # Normal loading from directory or HuggingFace repo
                self.pipe = MODEL_PIPELINE[self.config.model_type].from_pretrained(
                    model_path,
                    torch_dtype=self.config.model_dtype,
                    use_safetensors=True,
                    **MODEL_DEFAULTS[self.config.model_type].get("from_pretrained_extra_args", {}),
                )

            if self.config.model_type == ModelType.LTX_VIDEO_DEV:
                # Optionally load the upsampler pipeline for LTX-Video
                if not self.config.ltx_skip_upsampler:
                    self.logger.info("Loading LTX-Video upsampler pipeline...")
                    self.pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(
                        "Lightricks/ltxv-spatial-upscaler-0.9.7",
                        vae=self.pipe.vae,
                        torch_dtype=self.config.model_dtype,
                    )
                    self.pipe_upsample.set_progress_bar_config(disable=True)
                else:
                    self.logger.info("Skipping upsampler pipeline for faster calibration")
            self.pipe.set_progress_bar_config(disable=True)

            self.logger.info("Pipeline created successfully")
            return self.pipe

        except Exception as e:
            self.logger.error(f"Failed to create pipeline: {e}")
            raise

    def _load_pipeline_from_components(self, transformer_path: str) -> DiffusionPipeline:
        """
        Load pipeline from individual component files (transformer, VAE, text encoders).

        Args:
            transformer_path: Path to transformer .safetensors file

        Returns:
            Configured pipeline with local components
        """
        dtype_default = self.config.model_dtype.get("default", torch.bfloat16)

        if self.config.model_type in [ModelType.FLUX_DEV, ModelType.FLUX_SCHNELL]:
            self.logger.info("Loading FLUX pipeline from local components...")

            # Load transformer
            self.logger.info(f"Loading transformer from: {transformer_path}")
            # Create transformer with hardcoded config to avoid downloading from HuggingFace
            # Use meta device to reduce RAM usage during initialization
            with torch.device('meta'):
                transformer = FluxTransformer2DModel(
                    attention_head_dim=128,
                    guidance_embeds=True,
                    in_channels=64,
                    joint_attention_dim=4096,
                    num_attention_heads=24,
                    num_layers=19,
                    num_single_layers=38,
                    patch_size=1,
                    pooled_projection_dim=768,
                )
            
            # Load weights from safetensors file directly onto model
            from safetensors.torch import load_file
            state_dict = load_file(transformer_path)
            
            # Move from meta to CPU using to_empty()
            transformer = transformer.to_empty(device='cpu')
            transformer.load_state_dict(state_dict, strict=False, assign=True)
            del state_dict  # Free memory immediately
            
            # Convert to desired dtype
            transformer = transformer.to(self.config.model_dtype.get("transformer", dtype_default))
            self.logger.info(f"Transformer loaded successfully (dtype: {transformer.dtype})")

            # Load VAE
            if self.config.vae_path:
                self.logger.info(f"Loading FLUX VAE from: {self.config.vae_path}")
                # FLUX VAE has 16 latent channels (official configuration)
                with torch.device('meta'):
                    vae = AutoencoderKL(
                        in_channels=3,
                        out_channels=3,
                        down_block_types=["DownEncoderBlock2D"] * 4,
                        up_block_types=["UpDecoderBlock2D"] * 4,
                        block_out_channels=[128, 256, 512, 512],
                        layers_per_block=2,
                        act_fn="silu",
                        latent_channels=16,  # FLUX uses 16 latent channels
                        norm_num_groups=32,
                        sample_size=1024,
                        scaling_factor=0.3611,
                        shift_factor=0.1159,
                    )
                
                vae_state_dict = load_file(str(self.config.vae_path))
                vae = vae.to_empty(device='cpu')
                vae.load_state_dict(vae_state_dict, strict=False, assign=True)
                del vae_state_dict
                vae = vae.to(self.config.model_dtype.get("vae", dtype_default))
                self.logger.info(f"FLUX VAE loaded successfully (dtype: {vae.dtype})")
            else:
                self.logger.info("Loading VAE from HuggingFace (black-forest-labs/FLUX.1-dev)...")
                vae = AutoencoderKL.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="vae",
                    torch_dtype=self.config.model_dtype.get("vae", dtype_default),
                )

            # Load text encoders
            if self.config.text_encoder_path:
                self.logger.info(f"Loading CLIP text encoder from: {self.config.text_encoder_path}")
                from transformers import CLIPTextConfig
                
                # Load config and create model
                try:
                    clip_config = CLIPTextConfig.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        local_files_only=True,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not load CLIP config from cache: {e}. Downloading...")
                    clip_config = CLIPTextConfig.from_pretrained("openai/clip-vit-large-patch14")
                
                # Create model and load weights from safetensors
                # Use CLIPTextModel to keep pooler_output available for Flux pipelines.
                text_encoder = CLIPTextModel(clip_config)
                clip_state_dict = load_file(str(self.config.text_encoder_path))
                text_encoder.load_state_dict(clip_state_dict, strict=False)
                del clip_state_dict
                text_encoder = text_encoder.to(self.config.model_dtype.get("text_encoder", dtype_default))
                
                # Load tokenizer from cached HF model
                try:
                    tokenizer = CLIPTokenizer.from_pretrained(
                        "openai/clip-vit-large-patch14",
                        local_files_only=True,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not load tokenizer from cache, downloading: {e}")
                    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            else:
                self.logger.info("Loading CLIP from HuggingFace (openai/clip-vit-large-patch14)...")
                text_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    torch_dtype=self.config.model_dtype.get("text_encoder", dtype_default),
                )
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

            if self.config.text_encoder_2_path:
                self.logger.info(f"Loading T5 text encoder from: {self.config.text_encoder_2_path}")
                # Load T5 from single file
                from safetensors.torch import load_file
                state_dict = load_file(str(self.config.text_encoder_2_path))
                
                # Try to load config and create model from cached HF model
                try:
                    from transformers import T5Config
                    config_t5 = T5Config.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        subfolder="text_encoder_2",
                        local_files_only=True,
                    )
                    text_encoder_2 = T5EncoderModel(config_t5)
                    text_encoder_2.load_state_dict(state_dict, strict=False)
                    del state_dict  # Free memory
                    text_encoder_2 = text_encoder_2.to(self.config.model_dtype.get("text_encoder_2", dtype_default))
                except Exception as e:
                    self.logger.warning(f"Could not load T5 config from cache: {e}. Downloading...")
                    config_t5 = T5Config.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        subfolder="text_encoder_2",
                    )
                    text_encoder_2 = T5EncoderModel(config_t5)
                    text_encoder_2.load_state_dict(state_dict, strict=False)
                    del state_dict  # Free memory
                    text_encoder_2 = text_encoder_2.to(self.config.model_dtype.get("text_encoder_2", dtype_default))
                
                
                # Load tokenizer from cached HF model
                try:
                    tokenizer_2 = T5TokenizerFast.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        subfolder="tokenizer_2",
                        local_files_only=True,
                    )
                except Exception as e:
                    self.logger.warning(f"Could not load T5 tokenizer from cache: {e}. Downloading...")
                    tokenizer_2 = T5TokenizerFast.from_pretrained(
                        "black-forest-labs/FLUX.1-dev",
                        subfolder="tokenizer_2",
                    )
            else:
                self.logger.info("Loading T5 from HuggingFace (black-forest-labs/FLUX.1-dev)...")
                text_encoder_2 = T5EncoderModel.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="text_encoder_2",
                    torch_dtype=self.config.model_dtype.get("text_encoder_2", dtype_default),
                )
                tokenizer_2 = T5TokenizerFast.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="tokenizer_2",
                )

            # Load scheduler from base model
            from diffusers import FlowMatchEulerDiscreteScheduler
            try:
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="scheduler",
                    local_files_only=True,
                )
            except Exception as e:
                self.logger.warning(f"Could not load scheduler from cache: {e}. Downloading...")
                scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                    "black-forest-labs/FLUX.1-dev",
                    subfolder="scheduler",
                )

            # Construct pipeline
            pipe = FluxPipeline(
                scheduler=scheduler,
                text_encoder=text_encoder,
                tokenizer=tokenizer,
                text_encoder_2=text_encoder_2,
                tokenizer_2=tokenizer_2,
                vae=vae,
                transformer=transformer,
            )

            self.logger.info("Successfully created FLUX pipeline from local components")
            return pipe

        elif self.config.model_type in [ModelType.SD3_MEDIUM, ModelType.SD35_MEDIUM]:
            self.logger.info("Loading SD3 pipeline from local components...")

            # Load transformer
            self.logger.info(f"Loading transformer from: {transformer_path}")
            transformer = StableDiffusion3Transformer2DModel.from_single_file(
                transformer_path,
                torch_dtype=self.config.model_dtype.get("transformer", dtype_default),
            )

            # Load VAE
            if self.config.vae_path:
                self.logger.info(f"Loading VAE from: {self.config.vae_path}")
                vae = AutoencoderKL.from_single_file(
                    str(self.config.vae_path),
                    torch_dtype=self.config.model_dtype.get("vae", dtype_default),
                )
            else:
                base_model = "stabilityai/stable-diffusion-3-medium" if self.config.model_type == ModelType.SD3_MEDIUM else "stabilityai/stable-diffusion-3.5-medium"
                self.logger.info(f"Loading VAE from HuggingFace ({base_model})...")
                vae = AutoencoderKL.from_pretrained(
                    base_model,
                    subfolder="vae",
                    torch_dtype=self.config.model_dtype.get("vae", dtype_default),
                )

            # For SD3, we need 3 text encoders - for now, fall back to HF if not all provided
            # This is more complex and users typically keep the full model together
            if self.config.text_encoder_path and self.config.text_encoder_2_path:
                self.logger.warning("SD3 requires 3 text encoders. Falling back to HuggingFace for text encoders...")

            base_model = "stabilityai/stable-diffusion-3-medium" if self.config.model_type == ModelType.SD3_MEDIUM else "stabilityai/stable-diffusion-3.5-medium"

            # Load full pipeline and replace transformer
            pipe = StableDiffusion3Pipeline.from_pretrained(
                base_model,
                transformer=transformer,
                vae=vae,
                torch_dtype=self.config.model_dtype,
            )

            self.logger.info("Successfully created SD3 pipeline with local components")
            return pipe

        else:
            raise ValueError(
                f"Loading from components is not supported for {self.config.model_type.value}"
            )

    def setup_device(self) -> None:
        """Configure pipeline device placement."""
        if not self.pipe:
            raise RuntimeError("Pipeline not created. Call create_pipeline() first.")

        if self.config.cpu_offloading:
            self.logger.info("Enabling CPU offloading for memory efficiency")
            self.pipe.enable_model_cpu_offload()
            if self.pipe_upsample:
                self.pipe_upsample.enable_model_cpu_offload()
        else:
            self.logger.info("Moving pipeline to CUDA")
            self.pipe.to("cuda")
            if self.pipe_upsample:
                self.logger.info("Moving upsampler pipeline to CUDA")
                self.pipe_upsample.to("cuda")
        # Enable VAE tiling for LTX-Video to save memory
        if self.config.model_type == ModelType.LTX_VIDEO_DEV:
            if hasattr(self.pipe, "vae") and hasattr(self.pipe.vae, "enable_tiling"):
                self.logger.info("Enabling VAE tiling for LTX-Video")
                self.pipe.vae.enable_tiling()

    def get_backbone(self) -> torch.nn.Module:
        """
        Get the backbone model (transformer or UNet).

        Returns:
            Backbone model module
        """
        if not self.pipe:
            raise RuntimeError("Pipeline not created. Call create_pipeline() first.")

        return getattr(self.pipe, self.config.backbone)


class Calibrator:
    """Handles model calibration for quantization."""

    def __init__(
        self,
        pipeline_manager: PipelineManager,
        config: CalibrationConfig,
        model_type: ModelType,
        logger: logging.Logger,
    ):
        """
        Initialize calibrator.

        Args:
            pipeline_manager: Pipeline manager with main and upsampler pipelines
            config: Calibration configuration
            model_type: Type of model being calibrated
            logger: Logger instance
        """
        self.pipeline_manager = pipeline_manager
        self.pipe = pipeline_manager.pipe
        self.pipe_upsample = pipeline_manager.pipe_upsample
        self.config = config
        self.model_type = model_type
        self.logger = logger

    def load_and_batch_prompts(self) -> list[list[str]]:
        """
        Load calibration prompts from file.

        Returns:
            List of batched calibration prompts
        """
        self.logger.info(f"Loading calibration prompts from {self.config.prompts_dataset}")
        if isinstance(self.config.prompts_dataset, Path):
            return load_calib_prompts(
                self.config.batch_size,
                self.config.prompts_dataset,
            )

        return load_calib_prompts(
            self.config.batch_size,
            self.config.prompts_dataset["name"],
            self.config.prompts_dataset["split"],
            self.config.prompts_dataset["column"],
        )

    def run_calibration(self, batched_prompts: list[list[str]]) -> None:
        """
        Run calibration steps on the pipeline.

        Args:
            batched_prompts: List of batched calibration prompts
        """
        self.logger.info(f"Starting calibration with {self.config.num_batches} batches")
        extra_args = MODEL_DEFAULTS.get(self.model_type, {}).get("inference_extra_args", {})

        with tqdm(total=self.config.num_batches, desc="Calibration", unit="batch") as pbar:
            for i, prompt_batch in enumerate(batched_prompts):
                if i >= self.config.num_batches:
                    break

                if self.model_type == ModelType.LTX_VIDEO_DEV:
                    # Special handling for LTX-Video
                    self._run_ltx_video_calibration(prompt_batch, extra_args)
                elif self.model_type == ModelType.WAN22_T2V:
                    # Special handling for LTX-Video
                    self._run_wan_video_calibration(prompt_batch, extra_args)
                else:
                    common_args = {
                        "prompt": prompt_batch,
                        "num_inference_steps": self.config.n_steps,
                    }
                    self.pipe(**common_args, **extra_args).images  # type: ignore[misc]
                pbar.update(1)
                self.logger.debug(f"Completed calibration batch {i + 1}/{self.config.num_batches}")
        self.logger.info("Calibration completed successfully")

    def _run_wan_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        negative_prompt = extra_args["negative_prompt"]
        height = extra_args["height"]
        width = extra_args["width"]
        num_frames = extra_args["num_frames"]
        guidance_scale = extra_args["guidance_scale"]
        guidance_scale_2 = extra_args["guidance_scale_2"]

        self.pipe(
            prompt=prompt_batch,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            guidance_scale_2=guidance_scale_2,
            num_inference_steps=self.config.n_steps,
        ).frames  # type: ignore[misc]

    def _run_ltx_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        """
        Run calibration for LTX-Video model using the full multi-stage pipeline.

        Args:
            prompt_batch: Batch of prompts
            extra_args: Model-specific arguments
        """
        # Extract specific args for LTX-Video
        expected_height = extra_args.get("height", 512)
        expected_width = extra_args.get("width", 704)
        num_frames = extra_args.get("num_frames", 121)
        negative_prompt = extra_args.get(
            "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
        )

        def round_to_nearest_resolution_acceptable_by_vae(height, width):
            height = height - (height % self.pipe.vae_spatial_compression_ratio)  # type: ignore[union-attr]
            width = width - (width % self.pipe.vae_spatial_compression_ratio)  # type: ignore[union-attr]
            return height, width

        downscale_factor = 2 / 3
        # Part 1: Generate video at smaller resolution
        downscaled_height, downscaled_width = (
            int(expected_height * downscale_factor),
            int(expected_width * downscale_factor),
        )
        downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
            downscaled_height, downscaled_width
        )

        # Generate initial latents at lower resolution
        latents = self.pipe(  # type: ignore[misc]
            conditions=None,
            prompt=prompt_batch,
            negative_prompt=negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=self.config.n_steps,
            output_type="latent",
        ).frames

        # Part 2: Upscale generated video using latent upsampler (if available)
        if self.pipe_upsample is not None:
            _ = self.pipe_upsample(latents=latents, output_type="latent").frames

            # Part 3: Denoise the upscaled video with few steps to improve texture
            # However, in this example code, we will omit the upscale step since its optional.


class Quantizer:
    """Handles model quantization operations."""

    def __init__(
        self, config: QuantizationConfig, model_config: ModelConfig, logger: logging.Logger
    ):
        """
        Initialize quantizer.

        Args:
            config: Quantization configuration
            model_config: Model configuration
            logger: Logger instance
        """
        self.config = config
        self.model_config = model_config
        self.logger = logger

    def get_quant_config(self, n_steps: int, backbone: torch.nn.Module) -> Any:
        """
        Build quantization configuration based on format.

        Args:
            n_steps: Number of denoising steps

        Returns:
            Quantization configuration object
        """
        self.logger.info(f"Building quantization config for {self.config.format.value}")

        if self.config.format == QuantFormat.INT8:
            if self.config.algo == QuantAlgo.SMOOTHQUANT:
                quant_config = mtq.INT8_SMOOTHQUANT_CFG
            else:
                quant_config = INT8_DEFAULT_CONFIG
            if self.config.collect_method != CollectMethod.DEFAULT:
                reset_set_int8_config(
                    quant_config,
                    self.config.percentile,
                    n_steps,
                    collect_method=self.config.collect_method.value,
                    backbone=backbone,
                )
        elif self.config.format == QuantFormat.FP8:
            quant_config = FP8_DEFAULT_CONFIG
        elif self.config.format == QuantFormat.FP4:
            if self.model_config.model_type.value.startswith("flux"):
                quant_config = NVFP4_FP8_MHA_CONFIG
            else:
                quant_config = NVFP4_DEFAULT_CONFIG
        elif self.config.format == QuantFormat.FP4_MIXED_DEV:
            if self.model_config.model_type.value.startswith("flux"):
                quant_config = FLUX_DEV_NVFP4_MIXED_CONFIG
            else:
                self.logger.warning(
                    "FP4 mixed-dev format is only supported for FLUX models; using NVFP4 default config."
                )
                quant_config = NVFP4_DEFAULT_CONFIG
        else:
            raise NotImplementedError(f"Unknown format {self.config.format}")
        set_quant_config_attr(
            quant_config,
            self.model_config.trt_high_precision_dtype.value,
            self.config.algo.value,
            alpha=self.config.alpha,
            lowrank=self.config.lowrank,
        )

        return quant_config

    def quantize_model(
        self,
        backbone: torch.nn.Module,
        quant_config: Any,
        forward_loop: callable,  # type: ignore[valid-type]
    ) -> None:
        """
        Apply quantization to the model.

        Args:
            backbone: Model backbone to quantize
            quant_config: Quantization configuration
            forward_loop: Forward pass function for calibration
        """
        self.logger.info("Checking for LoRA layers...")
        check_lora(backbone)

        self.logger.info("Starting model quantization...")
        mtq.quantize(backbone, quant_config, forward_loop)
        # Get model-specific filter function
        model_filter_func = get_model_filter_func(self.model_config.model_type)
        self.logger.info(f"Using filter function for {self.model_config.model_type.value}")

        self.logger.info("Disabling specific quantizers...")
        mtq.disable_quantizer(backbone, model_filter_func)

        self.logger.info("Quantization completed successfully")


class ExportManager:
    """Handles model export operations."""

    def __init__(self, config: ExportConfig, logger: logging.Logger, quant_format: str = "fp4", quant_algo: str = "max"):
        """
        Initialize export manager.

        Args:
            config: Export configuration
            logger: Logger instance
            quant_format: Quantization format for metadata
            quant_algo: Quantization algorithm for metadata
        """
        self.config = config
        self.logger = logger
        self.quant_format = quant_format
        self.quant_algo = quant_algo

    def _has_conv_layers(self, model: torch.nn.Module) -> bool:
        """
        Check if the model contains any convolutional layers.

        Args:
            model: Model to check

        Returns:
            True if model contains Conv layers, False otherwise
        """
        for module in model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)) and (
                module.input_quantizer.is_enabled or module.weight_quantizer.is_enabled
            ):
                return True
        return False

    def save_checkpoint(self, backbone: torch.nn.Module) -> None:
        """
        Save quantized model checkpoint with ComfyUI-compatible metadata.

        Args:
            backbone: Model backbone to save
        """
        if not self.config.quantized_torch_ckpt_path:
            self.logger.warning("⚠️ quantized_torch_ckpt_path is None or empty! Skipping save.")
            self.logger.warning(f"⚠️ Config: {self.config}")
            return

        self.logger.info(f"💾 Saving quantized checkpoint to {self.config.quantized_torch_ckpt_path}")
        self.logger.info(f"💾 Path type: {type(self.config.quantized_torch_ckpt_path)}")
        self.logger.info(f"💾 Path exists (parent): {self.config.quantized_torch_ckpt_path.parent.exists()}")

        try:
            # Save ModelOpt .pt checkpoint
            mto.save(backbone, str(self.config.quantized_torch_ckpt_path))
            self.logger.info("✅ ModelOpt checkpoint saved successfully")

            # Verify the file was actually created
            import os
            if os.path.exists(str(self.config.quantized_torch_ckpt_path)):
                file_size = os.path.getsize(str(self.config.quantized_torch_ckpt_path)) / (1024**3)
                self.logger.info(f"✅ File verified: {file_size:.2f} GB")
            else:
                self.logger.error(f"❌ File was not created: {self.config.quantized_torch_ckpt_path}")
                return

            # Also save as SafeTensors with proper metadata for ComfyUI
            if self.config.save_safetensors:
                safetensors_path = self.config.quantized_torch_ckpt_path.with_suffix('.safetensors')
                self.logger.info(f"")
                self.logger.info("=" * 80)
                self.logger.info("🚀 CLAUDE'S FIX: Starting SafeTensors export with metadata!")
                self.logger.info(f"   Fix applied: 2026-01-14 @ 14:20 UTC+3")
                self.logger.info(f"   Target file: {safetensors_path.name}")
                self.logger.info("=" * 80)
                self.logger.info(f"📦 Saving SafeTensors format for ComfyUI compatibility...")

                # Map quant format
                if self.quant_format in ("fp4", "fp4_mixed_dev"):
                    quant_format_str = "nvfp4"
                else:
                    quant_format_str = "float8_e4m3fn"

                try:
                    save_quantized_safetensors(
                        backbone,
                        safetensors_path,
                        quant_format=quant_format_str,
                        quant_algo=self.quant_algo,
                        logger=self.logger
                    )
                except Exception as e:
                    self.logger.error(f"❌ SafeTensors save failed: {e}")
                    self.logger.warning("⚠️ ModelOpt .pt file is still available")

        except Exception as e:
            self.logger.error(f"❌ Save failed with error: {e}")
            raise

    def export_onnx(
        self,
        pipe: DiffusionPipeline,
        backbone: torch.nn.Module,
        model_type: ModelType,
        quant_format: QuantFormat,
    ) -> None:
        """
        Export model to ONNX format.

        Args:
            pipe: Diffusion pipeline
            backbone: Model backbone
            model_type: Type of model
            quant_format: Quantization format
        """
        if not self.config.onnx_dir:
            return

        self.logger.info(f"Starting ONNX export to {self.config.onnx_dir}")

        if quant_format == QuantFormat.FP8 and self._has_conv_layers(backbone):
            self.logger.info(
                "Detected quantizing conv layers in backbone. Generating FP8 scales..."
            )
            generate_fp8_scales(backbone)
        onnx_format = QuantFormat.FP4 if quant_format == QuantFormat.FP4_MIXED_DEV else quant_format
        self.logger.info("Preparing models for export...")
        pipe.to("cpu")
        torch.cuda.empty_cache()
        backbone.to("cuda")
        # Export to ONNX
        backbone.eval()
        with torch.no_grad():
            self.logger.info("Exporting to ONNX...")
            modelopt_export_sd(
                backbone, str(self.config.onnx_dir), model_type.value, onnx_format.value
            )

        self.logger.info("ONNX export completed successfully")

    def restore_checkpoint(self, backbone: nn.Module) -> None:
        """
        Restore a previously quantized model.

        Args:
            backbone: Model backbone to restore into
        """
        if not self.config.restore_from:
            return

        self.logger.info(f"Restoring model from {self.config.restore_from}")
        mto.restore(backbone, str(self.config.restore_from))
        self.logger.info("Model restored successfully")

    def export_hf_ckpt(self, pipe: DiffusionPipeline) -> None:
        """
        Export quantized model to HuggingFace checkpoint format.

        Args:
            pipe: Diffusion pipeline containing the quantized model
        """
        if not self.config.hf_ckpt_dir:
            return

        self.logger.info(f"Exporting HuggingFace checkpoint to {self.config.hf_ckpt_dir}")
        export_hf_checkpoint(pipe, export_dir=self.config.hf_ckpt_dir)
        self.logger.info("HuggingFace checkpoint export completed successfully")


def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Enhanced Diffusion Model Quantization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Basic INT8 quantization with SmoothQuant
            %(prog)s --model flux-dev --format int8 --quant-algo smoothquant --collect-method global_min

            # FP8 quantization with ONNX export
            %(prog)s --model sd3-medium --format fp8 --onnx-dir ./onnx_models/

            # FP8 quantization with weight compression (reduces memory footprint)
            %(prog)s --model flux-dev --format fp8 --compress

            # Quantize LTX-Video model with full multi-stage pipeline
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32

            # Faster LTX-Video quantization (skip upsampler)
            %(prog)s --model ltx-video-dev --format fp8 --batch-size 1 --calib-size 32 --ltx-skip-upsampler

            # Restore and export a previously quantized model
            %(prog)s --model flux-schnell --restore-from checkpoint.pt --onnx-dir ./exports/
        """,
    )
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument(
        "--model",
        type=str,
        default="flux-dev",
        choices=[m.value for m in ModelType],
        help="Model to load and quantize",
    )
    model_group.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="model backbone in the DiffusionPipeline to work on, if not provided use default based on model type",
    )
    model_group.add_argument(
        "--model-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for loading the pipeline. If you want different dtypes for separate components, "
        "please specify using --component-dtype",
    )
    model_group.add_argument(
        "--component-dtype",
        action="append",
        default=[],
        help="Precision for loading each component of the model by format of name:dtype. "
        "You can specify multiple components. "
        "Example: --component-dtype vae:Half --component-dtype transformer:BFloat16",
    )
    model_group.add_argument(
        "--override-model-path", type=str, help="Custom path to model (overrides default)"
    )
    model_group.add_argument(
        "--vae-path", type=str, help="Local path to VAE model (optional, for use with .safetensors transformer files)"
    )
    model_group.add_argument(
        "--text-encoder-path", type=str, help="Local path to text encoder (CLIP for FLUX, optional)"
    )
    model_group.add_argument(
        "--text-encoder-2-path", type=str, help="Local path to second text encoder (T5 for FLUX, optional)"
    )
    model_group.add_argument(
        "--cpu-offloading", action="store_true", help="Enable CPU offloading for limited VRAM"
    )
    model_group.add_argument(
        "--ltx-skip-upsampler",
        action="store_true",
        help="Skip upsampler pipeline for LTX-Video (faster calibration, only quantizes main transformer)",
    )
    quant_group = parser.add_argument_group("Quantization Configuration")
    quant_group.add_argument(
        "--format",
        type=str,
        default="int8",
        choices=[f.value for f in QuantFormat],
        help="Quantization format",
    )
    quant_group.add_argument(
        "--quant-algo",
        type=str,
        default="max",
        choices=[a.value for a in QuantAlgo],
        help="Quantization algorithm",
    )
    quant_group.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Percentile for calibration, works for INT8, not including smoothquant",
    )
    quant_group.add_argument(
        "--collect-method",
        type=str,
        default="default",
        choices=[c.value for c in CollectMethod],
        help="Calibration collection method, works for INT8, not including smoothquant",
    )
    quant_group.add_argument("--alpha", type=float, default=1.0, help="SmoothQuant alpha parameter")
    quant_group.add_argument("--lowrank", type=int, default=32, help="SVDQuant lowrank parameter")
    quant_group.add_argument(
        "--quantize-mha", action="store_true", help="Quantizing MHA into FP8 if its True"
    )
    quant_group.add_argument(
        "--compress",
        action="store_true",
        help="Compress quantized weights to reduce memory footprint (FP8/FP4 only)",
    )

    calib_group = parser.add_argument_group("Calibration Configuration")
    calib_group.add_argument("--batch-size", type=int, default=2, help="Batch size for calibration")
    calib_group.add_argument(
        "--calib-size", type=int, default=128, help="Total number of calibration samples"
    )
    calib_group.add_argument("--n-steps", type=int, default=30, help="Number of denoising steps")
    calib_group.add_argument(
        "--prompts-file",
        type=str,
        default=None,
        help="Calibrate using prompts in the file instead of the default dataset.",
    )

    export_group = parser.add_argument_group("Export Configuration")
    export_group.add_argument(
        "--quantized-torch-ckpt-save-path",
        type=str,
        help="Path to save quantized PyTorch checkpoint",
    )
    export_group.add_argument("--onnx-dir", type=str, help="Directory for ONNX export")
    export_group.add_argument(
        "--hf-ckpt-dir",
        type=str,
        help="Directory for HuggingFace checkpoint export",
    )
    export_group.add_argument(
        "--restore-from", type=str, help="Path to restore from previous checkpoint"
    )
    export_group.add_argument(
        "--trt-high-precision-dtype",
        type=str,
        default="Half",
        choices=[d.value for d in DataType],
        help="Precision for TensorRT high-precision layers",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    return parser


def main() -> None:
    parser = create_argument_parser()
    args = parser.parse_args()

    model_type = ModelType(args.model)
    if args.backbone is None:
        args.backbone = MODEL_DEFAULTS[model_type]["backbone"]
    s = time.time()

    model_dtype = {"default": DataType(args.model_dtype).torch_dtype}
    for component_dtype in args.component_dtype:
        component, dtype = component_dtype.split(":")
        model_dtype[component] = DataType(dtype).torch_dtype

    logger = setup_logging(args.verbose)
    logger.info("Starting Enhanced Diffusion Model Quantization")

    try:
        model_config = ModelConfig(
            model_type=model_type,
            model_dtype=model_dtype,
            backbone=args.backbone,
            trt_high_precision_dtype=DataType(args.trt_high_precision_dtype),
            override_model_path=Path(args.override_model_path)
            if args.override_model_path
            else None,
            vae_path=Path(args.vae_path) if args.vae_path else None,
            text_encoder_path=Path(args.text_encoder_path) if args.text_encoder_path else None,
            text_encoder_2_path=Path(args.text_encoder_2_path) if args.text_encoder_2_path else None,
            cpu_offloading=args.cpu_offloading,
            ltx_skip_upsampler=args.ltx_skip_upsampler,
        )

        quant_config = QuantizationConfig(
            format=QuantFormat(args.format),
            algo=QuantAlgo(args.quant_algo),
            percentile=args.percentile,
            collect_method=CollectMethod(args.collect_method),
            alpha=args.alpha,
            lowrank=args.lowrank,
            quantize_mha=args.quantize_mha,
            compress=args.compress,
        )

        if args.prompts_file is not None:
            prompts_file = Path(args.prompts_file)
            assert prompts_file.exists(), (
                f"User specified prompts file {prompts_file} does not exist."
            )
            prompts_dataset = prompts_file
        else:
            prompts_dataset = MODEL_DEFAULTS[model_type]["dataset"]
        calib_config = CalibrationConfig(
            prompts_dataset=prompts_dataset,
            batch_size=args.batch_size,
            calib_size=args.calib_size,
            n_steps=args.n_steps,
        )

        # DEBUG: Log the parsed arguments
        logger.info("=" * 80)
        logger.info("🔍 EXPORT CONFIGURATION DEBUG")
        logger.info(f"   args.quantized_torch_ckpt_save_path = {args.quantized_torch_ckpt_save_path}")
        logger.info(f"   Type: {type(args.quantized_torch_ckpt_save_path)}")
        logger.info(f"   Truthy: {bool(args.quantized_torch_ckpt_save_path)}")
        logger.info("=" * 80)

        export_config = ExportConfig(
            quantized_torch_ckpt_path=Path(args.quantized_torch_ckpt_save_path)
            if args.quantized_torch_ckpt_save_path
            else None,
            onnx_dir=Path(args.onnx_dir) if args.onnx_dir else None,
            hf_ckpt_dir=Path(args.hf_ckpt_dir) if args.hf_ckpt_dir else None,
            restore_from=Path(args.restore_from) if args.restore_from else None,
        )

        # DEBUG: Log the created config
        logger.info("=" * 80)
        logger.info("🔍 CREATED EXPORT CONFIG")
        logger.info(f"   export_config.quantized_torch_ckpt_path = {export_config.quantized_torch_ckpt_path}")
        logger.info(f"   Type: {type(export_config.quantized_torch_ckpt_path)}")
        logger.info("=" * 80)

        logger.info("Validating configurations...")
        quant_config.validate()
        export_config.validate()
        if not export_config.restore_from:
            calib_config.validate()

        pipeline_manager = PipelineManager(model_config, logger)
        pipe = pipeline_manager.create_pipeline()
        pipeline_manager.setup_device()

        backbone = pipeline_manager.get_backbone()
        export_manager = ExportManager(
            export_config,
            logger,
            quant_format=quant_config.format.value,
            quant_algo=quant_config.algo.value
        )

        if export_config.restore_from and export_config.restore_from.exists():
            export_manager.restore_checkpoint(backbone)

            if export_config.quantized_torch_ckpt_path and not export_config.restore_from.samefile(
                export_config.restore_from
            ):
                export_manager.save_checkpoint(backbone)
        else:
            logger.info("Initializing calibration...")
            calibrator = Calibrator(pipeline_manager, calib_config, model_config.model_type, logger)
            batched_prompts = calibrator.load_and_batch_prompts()

            quantizer = Quantizer(quant_config, model_config, logger)
            backbone_quant_config = quantizer.get_quant_config(calib_config.n_steps, backbone)

            def forward_loop(mod):
                calibrator.run_calibration(batched_prompts)

            quantizer.quantize_model(backbone, backbone_quant_config, forward_loop)

            # Compress model weights if requested (only for FP8/FP4)
            if quant_config.compress:
                logger.info("Compressing model weights to reduce memory footprint...")
                mtq.compress(backbone)
                logger.info("Model compression completed")

            export_manager.save_checkpoint(backbone)

        is_fp4 = quant_config.format in (QuantFormat.FP4, QuantFormat.FP4_MIXED_DEV)
        check_conv_and_mha(backbone, is_fp4, quant_config.quantize_mha)
        mtq.print_quant_summary(backbone)

        export_manager.export_onnx(
            pipe,
            backbone,
            model_config.model_type,
            quant_config.format,
        )

        export_manager.export_hf_ckpt(pipe)

        logger.info(
            f"Quantization process completed successfully! Time taken = {time.time() - s} seconds"
        )

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
