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
from typing import Any

import torch
from diffusers import DiffusionPipeline, LTXLatentUpsamplePipeline
from models_utils import MODEL_DEFAULTS, MODEL_PIPELINE, MODEL_REGISTRY, ModelType
from quantize_config import ModelConfig


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
        self.pipe: Any | None = None
        self.pipe_upsample: LTXLatentUpsamplePipeline | None = None  # For LTX-Video upsampling
        self._transformer: torch.nn.Module | None = None

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
            pipeline_cls = MODEL_PIPELINE[model_type]
            if pipeline_cls is None:
                raise ValueError(f"Model type {model_type.value} does not use diffusers pipelines.")
            model_id = (
                MODEL_REGISTRY[model_type] if override_model_path is None else override_model_path
            )
            pipe = pipeline_cls.from_pretrained(
                model_id,
                torch_dtype=torch_dtype,
                use_safetensors=True,
                **MODEL_DEFAULTS[model_type].get("from_pretrained_extra_args", {}),
            )
            pipe.set_progress_bar_config(disable=True)
            return pipe
        except Exception as e:
            raise e

    def create_pipeline(self) -> Any:
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
            if self.config.model_type == ModelType.LTX2:
                from modelopt.torch.quantization.plugins.diffusion import ltx2 as ltx2_plugin

                ltx2_plugin.register_ltx2_quant_linear()
                self.pipe = self._create_ltx2_pipeline()
                self.logger.info("LTX-2 pipeline created successfully")
                return self.pipe

            pipeline_cls = MODEL_PIPELINE[self.config.model_type]
            if pipeline_cls is None:
                raise ValueError(
                    f"Model type {self.config.model_type.value} does not use diffusers pipelines."
                )
            self.pipe = pipeline_cls.from_pretrained(
                self.config.model_path,
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

    def setup_device(self) -> None:
        """Configure pipeline device placement."""
        if not self.pipe:
            raise RuntimeError("Pipeline not created. Call create_pipeline() first.")

        if self.config.model_type == ModelType.LTX2:
            self.logger.info("Skipping device setup for LTX-2 pipeline (handled internally)")
            return

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

        if self.config.model_type == ModelType.LTX2:
            self._ensure_ltx2_transformer_cached()
            return self._transformer
        return getattr(self.pipe, self.config.backbone)

    def _ensure_ltx2_transformer_cached(self) -> None:
        if not self.pipe:
            raise RuntimeError("Pipeline not created. Call create_pipeline() first.")
        if self._transformer is None:
            transformer = self.pipe.stage_1_model_ledger.transformer()
            self.pipe.stage_1_model_ledger.transformer = lambda: transformer
            self._transformer = transformer

    def _create_ltx2_pipeline(self) -> Any:
        params = dict(self.config.extra_params)
        checkpoint_path = params.pop("checkpoint_path", None)
        distilled_lora_path = params.pop("distilled_lora_path", None)
        distilled_lora_strength = params.pop("distilled_lora_strength", 0.8)
        spatial_upsampler_path = params.pop("spatial_upsampler_path", None)
        gemma_root = params.pop("gemma_root", None)
        fp8transformer = params.pop("fp8transformer", False)

        if not checkpoint_path:
            raise ValueError("Missing required extra_param: checkpoint_path.")
        if not distilled_lora_path:
            raise ValueError("Missing required extra_param: distilled_lora_path.")
        if not spatial_upsampler_path:
            raise ValueError("Missing required extra_param: spatial_upsampler_path.")
        if not gemma_root:
            raise ValueError("Missing required extra_param: gemma_root.")

        from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                str(distilled_lora_path),
                float(distilled_lora_strength),
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]
        pipeline_kwargs = {
            "checkpoint_path": str(checkpoint_path),
            "distilled_lora": distilled_lora,
            "spatial_upsampler_path": str(spatial_upsampler_path),
            "gemma_root": str(gemma_root),
            "loras": [],
            "fp8transformer": bool(fp8transformer),
        }
        pipeline_kwargs.update(params)
        return TI2VidTwoStagesPipeline(**pipeline_kwargs)
