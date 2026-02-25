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
from pathlib import Path
from typing import Any

from models_utils import MODEL_DEFAULTS, ModelType
from pipeline_manager import PipelineManager
from quantize_config import CalibrationConfig
from tqdm import tqdm
from utils import load_calib_prompts


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

                if self.model_type == ModelType.LTX2:
                    self._run_ltx2_calibration(prompt_batch, extra_args)
                elif self.model_type == ModelType.LTX_VIDEO_DEV:
                    # Special handling for LTX-Video
                    self._run_ltx_video_calibration(prompt_batch, extra_args)
                elif self.model_type in [ModelType.WAN22_T2V_14b, ModelType.WAN22_T2V_5b]:
                    # Special handling for WAN video models
                    self._run_wan_video_calibration(prompt_batch, extra_args)
                else:
                    common_args = {
                        "prompt": prompt_batch,
                        "num_inference_steps": self.config.n_steps,
                    }
                    self.pipe(**common_args, **extra_args).images
                pbar.update(1)
                self.logger.debug(f"Completed calibration batch {i + 1}/{self.config.num_batches}")
        self.logger.info("Calibration completed successfully")

    def _run_wan_video_calibration(
        self, prompt_batch: list[str], extra_args: dict[str, Any]
    ) -> None:
        kwargs = {}
        kwargs["negative_prompt"] = extra_args["negative_prompt"]
        kwargs["height"] = extra_args["height"]
        kwargs["width"] = extra_args["width"]
        kwargs["num_frames"] = extra_args["num_frames"]
        kwargs["guidance_scale"] = extra_args["guidance_scale"]
        if "guidance_scale_2" in extra_args:
            kwargs["guidance_scale_2"] = extra_args["guidance_scale_2"]
        kwargs["num_inference_steps"] = self.config.n_steps

        self.pipe(prompt=prompt_batch, **kwargs).frames

    def _run_ltx2_calibration(self, prompt_batch: list[str], extra_args: dict[str, Any]) -> None:
        from ltx_core.model.video_vae import TilingConfig

        prompt = prompt_batch[0]
        extra_params = self.pipeline_manager.config.extra_params
        kwargs = {
            "negative_prompt": extra_args.get(
                "negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"
            ),
            "seed": extra_params.get("seed", 0),
            "height": extra_params.get("height", extra_args.get("height", 1024)),
            "width": extra_params.get("width", extra_args.get("width", 1536)),
            "num_frames": extra_params.get("num_frames", extra_args.get("num_frames", 121)),
            "frame_rate": extra_params.get("frame_rate", extra_args.get("frame_rate", 24.0)),
            "num_inference_steps": self.config.n_steps,
            "cfg_guidance_scale": extra_params.get(
                "cfg_guidance_scale", extra_args.get("cfg_guidance_scale", 4.0)
            ),
            "images": extra_params.get("images", []),
            "tiling_config": extra_params.get("tiling_config", TilingConfig.default()),
        }
        self.pipe(prompt=prompt, **kwargs)

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
            height = height - (height % self.pipe.vae_spatial_compression_ratio)
            width = width - (width % self.pipe.vae_spatial_compression_ratio)
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
        latents = self.pipe(
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
