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
Distillation Trainer for LTX-2 DiT Model with ModelOpt Quantization

This module implements sparsity/quantization-aware distillation training where:
- Teacher: Original unsparsified/unquantized model (inference only)
- Student: Quantized model using ModelOpt's fake quantization (trainable)

The distillation loss combines:
- L_task: Standard flow matching MSE loss (student_pred vs velocity_target)
- L_distill: Distillation MSE loss (student_pred vs teacher_pred)

Usage:
    python distillation_trainer.py --config configs/distillation_example.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Literal

import torch
import torch.distributed as dist
from ltx_trainer import logger
from ltx_trainer.config import ConfigBaseModel, LtxTrainerConfig
from ltx_trainer.model_loader import load_transformer
from ltx_trainer.trainer import IS_MAIN_PROCESS, LtxvTrainer
from omegaconf import OmegaConf
from pydantic import Field
from torch import Tensor

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq

# Custom quantization configs. Checked before mtq built-in configs.
# Add your own configs here; they take precedence over mtq.* attributes.
CUSTOM_QUANT_CONFIGS: dict[str, dict] = {
    # Example: override NVFP4 with a different algorithm
    # "MY_NVFP4_CFG": {
    #     "quant_cfg": mtq.NVFP4_DEFAULT_CFG["quant_cfg"],
    #     "algorithm": "max",
    # },
}


# IS_MAIN_PROCESS (from ltx_trainer) checks LOCAL_RANK == 0, which is True on
# every node in multi-node training.  For file writes on a shared filesystem
# (Lustre) we need a global-rank-0 check so that only a single process writes.
def is_global_rank0() -> bool:
    """Check if this is global rank 0. Safe to call before or after dist init."""
    if dist.is_initialized():
        return dist.get_rank() == 0
    return os.environ.get("RANK", "0") == "0"


def get_quant_config(quant_cfg_name: str) -> dict:
    """
    Resolve a quantization config by name.

    Lookup order:
    1. CUSTOM_QUANT_CONFIGS (user-defined overrides in this file)
    2. mtq.<name> (built-in ModelOpt configs, e.g. FP8_DEFAULT_CFG, INT8_DEFAULT_CFG)

    Args:
        quant_cfg_name: Name of the quantization config, e.g. "FP8_DEFAULT_CFG".

    Returns:
        A copy of the quantization config dict.
    """
    # Check custom configs first
    if quant_cfg_name in CUSTOM_QUANT_CONFIGS:
        logger.info(f"Using custom quant config: {quant_cfg_name}")
        return CUSTOM_QUANT_CONFIGS[quant_cfg_name].copy()

    # Fall back to mtq built-in configs
    cfg = getattr(mtq, quant_cfg_name, None)
    if cfg is None:
        available_custom = list(CUSTOM_QUANT_CONFIGS.keys())
        available_mtq = [
            attr for attr in dir(mtq) if attr.endswith("_CFG") and not attr.startswith("_")
        ]
        raise ValueError(
            f"Unknown quant_cfg: '{quant_cfg_name}'. "
            f"Available custom: {available_custom}. "
            f"Available mtq built-in: {available_mtq}"
        )
    logger.info(f"Using mtq built-in quant config: {quant_cfg_name}")
    return cfg.copy()


class MockDataset(torch.utils.data.Dataset):
    """
    Mock dataset that produces random data matching the expected training format.

    This is useful for testing the training pipeline without preparing real data.
    The output format matches what PrecomputedDataset produces, with keys:
    - "latents": video latent tensors and metadata
    - "conditions": text embeddings and attention masks

    Note: prompt_embed_dim should be 3840 (the connector's inner_dim = 30 heads * 128 dim),
    NOT 4096 (Gemma's raw hidden size). The PrecomputedDataset stores embeddings that have
    already been projected through the feature_extractor_linear layer.
    """

    def __init__(
        self,
        width: int = 512,
        height: int = 320,
        num_frames: int = 33,
        dataset_length: int = 100,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 3840,  # Connector inner_dim, not Gemma's 4096
        prompt_sequence_length: int = 256,
        fps: int = 25,
        dtype: torch.dtype = torch.bfloat16,  # Must match model dtype
    ):
        """
        Initialize mock dataset.

        Args:
            width: Video width in pixels (must be divisible by 32)
            height: Video height in pixels (must be divisible by 32)
            num_frames: Number of video frames (should be 8k+1 for proper compression)
            dataset_length: Number of samples in the dataset
            latent_dim: Latent channel dimension (128 for LTX-2)
            latent_spatial_compression_ratio: Spatial compression ratio (32 for LTX-2)
            latent_temporal_compression_ratio: Temporal compression ratio (8 for LTX-2)
            prompt_embed_dim: Text embedding dimension after projection (3840 for LTX-2,
                which is connector's inner_dim = 30 heads * 128 dim_head)
            prompt_sequence_length: Max text sequence length
            fps: Frames per second
            dtype: Data type for floating point tensors (must match model dtype, default bfloat16)
        """
        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length
        self.fps = fps
        self.dtype = dtype

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict:
        """
        Get a mock sample.

        Returns format expected by training strategy:
        - latents: dict with "latents" tensor [C, F, H, W] and metadata
        - conditions: dict with "prompt_embeds" and "prompt_attention_mask"
        """
        return {
            # Video latents (key: "latents" to match PrecomputedDataset)
            "latents": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                    dtype=self.dtype,  # Must match model dtype
                ),
                "num_frames": torch.tensor(self.num_latent_frames),
                "height": torch.tensor(self.latent_height),
                "width": torch.tensor(self.latent_width),
                "fps": torch.tensor(self.fps),
            },
            # Text conditions (key: "conditions" to match PrecomputedDataset)
            "conditions": {
                "prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                    dtype=self.dtype,  # Must match model dtype
                ),
                # Attention mask must be numeric (not bool) for _run_connectors
                # Using int8 to save memory (1 byte vs 8 bytes for long)
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.int8,
                ),
            },
            "idx": idx,
        }


class DistillationConfig(ConfigBaseModel):
    """Configuration for distillation-specific parameters."""

    teacher_model_path: str | Path | None = Field(
        default=None,
        description="Path to the teacher model checkpoint. If None, uses the same as model.model_path "
        "(teacher is loaded without quantization).",
    )

    distillation_alpha: float = Field(
        default=0.5,
        description="Weight for the task loss. Distillation loss weight is (1 - alpha). "
        "alpha=1.0 means no distillation (pure task loss), "
        "alpha=0.0 means pure distillation loss.",
        ge=0.0,
        le=1.0,
    )

    distillation_loss_type: Literal["mse", "cosine"] = Field(
        default="mse",
        description="Type of distillation loss. 'mse' is recommended since transformer outputs "
        "are continuous velocity predictions in latent space (not probabilities). "
        "'cosine' matches direction only, ignoring magnitude.",
    )

    teacher_dtype: Literal["bfloat16", "float16", "float32"] = Field(
        default="bfloat16",
        description="Data type for teacher model. BFloat16 is recommended for memory efficiency.",
    )

    # ModelOpt Quantization Settings
    quant_cfg: str | None = Field(
        default=None,
        description="Name of the ModelOpt quantization config to apply to the student model. "
        "Looked up first in CUSTOM_QUANT_CONFIGS (distillation_trainer.py), then as mtq.<name>. "
        "Examples: 'FP8_DEFAULT_CFG', 'INT8_DEFAULT_CFG', 'NVFP4_DEFAULT_CFG'. "
        "Set to None to disable quantization.",
    )

    # Calibration settings (full-inference calibration, matching PTQ workflow)
    calibration_prompts_file: str | Path | None = Field(
        default=None,
        description="Path to a text file with one calibration prompt per line. "
        "If None, uses the HuggingFace dataset 'Gustavosta/Stable-Diffusion-Prompts' ",
    )

    calibration_size: int = Field(
        default=128,
        description="Total number of calibration prompts to use. Each prompt runs a full "
        "denoising inference through the DiT, covering all noise levels. ",
        ge=0,
    )

    calibration_n_steps: int = Field(
        default=30,
        description="Number of denoising steps per calibration prompt. Each step calls the "
        "transformer at a different noise level.",
        ge=1,
    )

    calibration_guidance_scale: float = Field(
        default=4.0,
        description="CFG guidance scale during calibration. Default 4.0.",
        ge=1.0,
    )

    restore_quantized_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to restore a previously quantized model from mto.save().",
    )

    save_quantized_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to save the final quantized model checkpoint.",
    )

    # Checkpoint resume settings
    resume_from_checkpoint: str | Path | None = Field(
        default=None,
        description="Path to a training state checkpoint directory (from save_training_state) to resume "
        "training from. Restores model weights, optimizer, LR scheduler, RNG states, and step counter. "
        "Set to 'latest' to auto-find the latest checkpoint in output_dir/checkpoints/.",
    )

    must_save_by: float | None = Field(
        default=None,
        description="Minutes after which training must save a checkpoint and exit. "
        "Use this when running under a Slurm time limit — set to a value slightly less "
        "than the time limit (e.g., time_limit=30min → must_save_by=25) to ensure "
        "a checkpoint is saved before the job is killed. Timer starts at train() entry. "
        "Set to None to disable.",
        gt=0,
    )

    # Debug/Test options
    use_mock_data: bool = Field(
        default=False,
        description="Use mock data instead of real preprocessed data for testing.",
    )

    mock_data_samples: int = Field(
        default=100,
        description="Number of mock samples to generate when use_mock_data is True.",
        ge=1,
    )


class DistillationTrainerConfig(LtxTrainerConfig):
    """Extended trainer config with distillation settings."""

    distillation: DistillationConfig = Field(
        default_factory=DistillationConfig,
        description="Distillation-specific configuration.",
    )


class DistillationTrainer(LtxvTrainer):
    """
    Distillation trainer that extends LtxvTrainer with:
    - Teacher model loading and inference
    - ModelOpt quantization for student
    - Combined task + distillation loss
    """

    def __init__(self, config: DistillationTrainerConfig) -> None:
        # Store distillation config before parent init (needed by overrides called during super().__init__)
        self._distillation_config = config.distillation
        # Will be populated by _load_text_encoder_and_cache_embeddings() during super().__init__
        self._cached_calibration_embeddings: list | None = None

        # Create base trainer config (without distillation section)
        trainer_config = LtxTrainerConfig(
            **{k: v for k, v in config.model_dump().items() if k != "distillation"}
        )

        # Initialize parent (loads student model, sets up accelerator)
        # Note: _prepare_models_for_training() is overridden to NOT call
        # accelerator.prepare() on the student — we defer that to _init_optimizer()
        # so model+optimizer can be prepared together (required by FSDP2).
        super().__init__(trainer_config)

        # Load teacher model (after parent init so we have accelerator)
        # Teacher is loaded, frozen, and prepared with a dummy optimizer.
        self._load_teacher_model()

        logger.info(
            f"Distillation training initialized with alpha={self._distillation_config.distillation_alpha:.2f}"
        )

    def _prepare_models_for_training(self) -> None:
        """
        Override parent to defer accelerator.prepare() for the student model.

        The parent calls accelerator.prepare(transformer) here, but FSDP2 requires
        model and optimizer to be prepared together. So we do everything the parent
        does EXCEPT the accelerator.prepare() call — that happens in _init_optimizer()
        where we can call prepare(model, optimizer, scheduler) together.
        """
        from accelerate.utils import DistributedType

        # For FSDP + LoRA: Cast entire model to FP32 for uniform dtype
        if (
            self._accelerator.distributed_type == DistributedType.FSDP
            and self._config.model.training_mode == "lora"
        ):
            logger.debug("FSDP: casting transformer to FP32 for uniform dtype")
            self._transformer = self._transformer.to(dtype=torch.float32)

        # Enable gradient checkpointing if requested
        transformer = (
            self._transformer.get_base_model()
            if hasattr(self._transformer, "get_base_model")
            else self._transformer
        )
        transformer.set_gradient_checkpointing(
            self._config.optimization.enable_gradient_checkpointing
        )

        # Keep frozen models on CPU for memory efficiency
        self._vae_decoder = self._vae_decoder.to("cpu")
        if self._vae_encoder is not None:
            self._vae_encoder = self._vae_encoder.to("cpu")

        # NOTE: We intentionally do NOT call accelerator.prepare(self._transformer) here.
        # It will be called in _init_optimizer() together with the optimizer, which is
        # required for FSDP2 compatibility. This also works fine with FSDP1.

        # Log GPU memory usage
        vram_usage_gb = torch.cuda.memory_allocated() / 1024**3
        logger.debug(f"GPU memory usage after models preparation: {vram_usage_gb:.2f} GB")

    def _load_text_encoder_and_cache_embeddings(self):
        """
        Override parent to also cache calibration prompt embeddings before Gemma is unloaded.

        The parent method loads the full Gemma text encoder, caches validation prompt embeddings,
        then UNLOADS the heavy Gemma model (sets model/tokenizer/feature_extractor_linear to None)
        to free VRAM. Only the lightweight embedding connectors remain.

        We hook in here to also cache calibration prompt embeddings while the full text encoder
        is still available. These cached embeddings are later used by _run_inference_calibration()
        via the ValidationSampler's CachedPromptEmbeddings mechanism.
        """
        from ltx_trainer.model_loader import load_text_encoder
        from ltx_trainer.validation_sampler import CachedPromptEmbeddings

        # Call parent to load text encoder, cache validation embeddings, and unload Gemma.
        # But we need to intercept BEFORE the unload. We re-implement the parent logic
        # with our addition in the middle.

        logger.debug("Loading text encoder...")
        self._text_encoder = load_text_encoder(
            checkpoint_path=self._config.model.model_path,
            gemma_model_path=self._config.model.text_encoder_path,
            device="cuda",
            dtype=torch.bfloat16,
            load_in_8bit=self._config.acceleration.load_text_encoder_in_8bit,
        )

        # Cache validation embeddings (same as parent)
        cached_validation = None
        if self._config.validation.prompts:
            logger.info(
                f"Pre-computing embeddings for {len(self._config.validation.prompts)} validation prompts..."
            )
            cached_validation = []
            with torch.inference_mode():
                for prompt in self._config.validation.prompts:
                    v_ctx_pos, a_ctx_pos, _ = self._text_encoder(prompt)
                    v_ctx_neg, a_ctx_neg, _ = self._text_encoder(
                        self._config.validation.negative_prompt
                    )
                    cached_validation.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx_pos.cpu(),
                            audio_context_positive=a_ctx_pos.cpu(),
                            video_context_negative=v_ctx_neg.cpu()
                            if v_ctx_neg is not None
                            else None,
                            audio_context_negative=a_ctx_neg.cpu()
                            if a_ctx_neg is not None
                            else None,
                        )
                    )

        # Cache calibration prompt embeddings while the heavy text encoder is still loaded.
        # Only needed if we'll actually run fresh calibration (Path C). Skip if a
        # resumable checkpoint, user-specified checkpoint, or step 0 checkpoint exists.
        calib_cfg = self._distillation_config
        if (
            calib_cfg.quant_cfg is not None
            and calib_cfg.calibration_size > 0
            and self._needs_fresh_calibration()
        ):
            prompts = self._load_calibration_prompts()
            negative_prompt = getattr(
                self._config.validation,
                "negative_prompt",
                "worst quality, inconsistent motion, blurry, jittery, distorted",
            )
            logger.info(
                f"Pre-computing embeddings for {len(prompts)} calibration prompts "
                f"(guidance_scale={calib_cfg.calibration_guidance_scale})..."
            )
            self._cached_calibration_embeddings = []
            use_cfg = calib_cfg.calibration_guidance_scale != 1.0
            with torch.inference_mode():
                for prompt in prompts:
                    v_ctx_pos, a_ctx_pos, _ = self._text_encoder(prompt)
                    v_ctx_neg, a_ctx_neg = None, None
                    if use_cfg:
                        v_ctx_neg, a_ctx_neg, _ = self._text_encoder(negative_prompt)
                    self._cached_calibration_embeddings.append(
                        CachedPromptEmbeddings(
                            video_context_positive=v_ctx_pos.cpu(),
                            audio_context_positive=a_ctx_pos.cpu(),
                            video_context_negative=v_ctx_neg.cpu()
                            if v_ctx_neg is not None
                            else None,
                            audio_context_negative=a_ctx_neg.cpu()
                            if a_ctx_neg is not None
                            else None,
                        )
                    )
            logger.info(f"Cached {len(self._cached_calibration_embeddings)} calibration embeddings")

        # Unload heavy components to free VRAM, keeping only the embedding connectors
        self._text_encoder.model = None
        self._text_encoder.tokenizer = None
        self._text_encoder.feature_extractor_linear = None
        gc.collect()
        torch.cuda.empty_cache()
        logger.debug("Validation/calibration prompt embeddings cached. Gemma model unloaded")

        return cached_validation

    def _load_models(self) -> None:
        """
        Load the LTX-2 model components with ModelOpt quantization for student.

        This overrides the parent method to:
        1. Load models as usual (without ltx-trainer's quantization)
        2. Apply ModelOpt fake quantization to the student transformer
        """
        # Call parent to load all models normally
        super()._load_models()

        # Apply ModelOpt quantization to student if configured
        if self._distillation_config.quant_cfg is not None:
            self._apply_modelopt_quantization()
            gc.collect()
            torch.cuda.empty_cache()
            logger.info(f"Quantized model: {self._transformer}")

    def _needs_fresh_calibration(self) -> bool:
        """Check whether fresh quantization calibration will be needed.

        Returns False if an existing checkpoint can be restored instead
        (Path A, B, or B2 in _apply_modelopt_quantization), meaning we can
        skip the expensive calibration embedding caching.
        """
        cfg = self._distillation_config

        # Path A: resume checkpoint with modelopt_state.pt
        if cfg.resume_from_checkpoint is not None:
            checkpoint_dir = self._find_resume_checkpoint(cfg.resume_from_checkpoint)
            if checkpoint_dir is not None:
                if (checkpoint_dir / "modelopt_state.pt").exists():
                    return False

        # Path B: user-specified quantized checkpoint
        if cfg.restore_quantized_checkpoint is not None:
            return False

        # Path B2: auto-detected step 0 checkpoint
        step0_path = self._get_checkpoints_dir() / "step_000000_quantized" / "backbone.pt"
        return not step0_path.exists()

    def _apply_modelopt_quantization(self) -> None:
        """
        Apply ModelOpt fake quantization to the student transformer.

        Four paths are supported (checked in order):

        Path A - Resume from training checkpoint:
            If resume_from_checkpoint is set, restore only the quantization module
            architecture (fake quantizer modules) from the saved modelopt_state.pt.
            The actual trained weights (including quantizer scales) will be loaded
            later by accelerator.load_state() in _load_training_state().

        Path B - Restore from a user-specified quantized checkpoint:
            If restore_quantized_checkpoint is set, restore both architecture and
            weights from a complete mto.save() checkpoint.

        Path B2 - Auto-detect step 0 quantized checkpoint:
            If a previous run already completed calibration and saved the step 0
            checkpoint (step_000000_quantized/backbone.pt), restore from it
            automatically. This avoids re-running the expensive calibration.

        Path C - Fresh quantization with full-inference calibration:
            Apply mtq.quantize() with a forward_loop that runs full denoising
            inference (like the PTQ workflow), covering all noise levels.
            After calibration, saves the result as step 0 checkpoint for future runs.
        """
        quant_cfg_name = self._distillation_config.quant_cfg
        if not quant_cfg_name:
            logger.info("No quant_cfg specified, skipping quantization")
            return

        # Path A: Resume from training checkpoint — restore architecture only.
        # The trained weights (including quantizer scales) are loaded later by
        # accelerator.load_state() in _load_training_state().
        resume_path = self._distillation_config.resume_from_checkpoint
        if resume_path is not None:
            checkpoint_dir = self._find_resume_checkpoint(resume_path)
            if checkpoint_dir is not None:
                modelopt_state_path = checkpoint_dir / "modelopt_state.pt"
                if modelopt_state_path.exists():
                    logger.info(
                        f"Resuming: restoring quantization architecture from "
                        f"{modelopt_state_path} (weights loaded later by accelerator)"
                    )
                    self._transformer = mto.restore_from_modelopt_state(
                        self._transformer, modelopt_state_path=modelopt_state_path
                    )
                    logger.info("Quantization architecture restored for resume")
                    return
                else:
                    logger.warning(
                        f"modelopt_state.pt not found in {checkpoint_dir}, "
                        "falling through to fresh quantization"
                    )

        # Path B: Restore from a standalone quantized checkpoint (architecture + weights).
        if self._distillation_config.restore_quantized_checkpoint is not None:
            restore_path = str(self._distillation_config.restore_quantized_checkpoint)
            logger.info(f"Restoring quantized model from {restore_path}")
            mto.restore(self._transformer, restore_path)
            return

        # Path B2: Auto-detect step 0 quantized checkpoint from a previous run.
        # If calibration was already completed and saved, reuse it instead of
        # re-running the expensive calibration process.
        step0_path = self._get_checkpoints_dir() / "step_000000_quantized" / "backbone.pt"
        if step0_path.exists():
            logger.info(
                f"Found existing step 0 quantized checkpoint at {step0_path}, "
                "restoring instead of re-running calibration"
            )
            try:
                mto.restore(self._transformer, str(step0_path))
                return
            except Exception as e:
                logger.warning(
                    f"Failed to restore step 0 checkpoint (file may be corrupted): {e}. "
                    "Falling through to fresh quantization."
                )

        # Path C: Fresh quantization with full-inference calibration.
        logger.info(f"Applying ModelOpt quantization ({quant_cfg_name}) to student transformer...")

        quant_config = get_quant_config(quant_cfg_name)

        def forward_loop(model):
            """Run full-inference calibration covering all noise levels."""
            self._run_inference_calibration(model)

        mtq.quantize(self._transformer, quant_config, forward_loop=forward_loop)

        # Free cached calibration embeddings — no longer needed after quantization
        self._cached_calibration_embeddings = None

        logger.info(f"ModelOpt quantization ({quant_cfg_name}) applied successfully")

        # Save the freshly quantized+calibrated model as "step 0" checkpoint.
        # This avoids re-running calibration if training is interrupted before the
        # first regular checkpoint. On resume, Path B2 auto-detects and loads this.
        # Only model + quantizer scales are saved (no optimizer/scheduler state at step 0).
        # We use atomic save (write to tmp, then rename) to prevent corrupt checkpoints.
        step0_dir = self._get_checkpoints_dir() / "step_000000_quantized"
        step0_path = step0_dir / "backbone.pt"
        # Only global rank 0 saves (all ranks have identical models pre-FSDP);
        # others wait at the barrier. Atomic save (tmp + rename) prevents corruption.
        if is_global_rank0():
            step0_dir.mkdir(parents=True, exist_ok=True)
            step0_tmp_path = step0_dir / "backbone.pt.tmp"
            logger.info(f"Saving quantized model (step 0) to {step0_path}")
            mto.save(self._transformer, str(step0_tmp_path))
            step0_tmp_path.rename(step0_path)
            logger.info("Step 0 quantized checkpoint saved")
        if dist.is_initialized():
            dist.barrier()

    def _create_mock_dataset(self) -> MockDataset:
        """Create a mock dataset for testing without real data."""
        # Get video dimensions from validation config or use defaults
        video_dims = getattr(self._config.validation, "video_dims", [512, 320, 33])
        width, height, num_frames = video_dims

        logger.info(
            f"Creating mock dataset with {self._distillation_config.mock_data_samples} samples "
            f"(video: {width}x{height}x{num_frames})"
        )

        return MockDataset(
            width=width,
            height=height,
            num_frames=num_frames,
            dataset_length=self._distillation_config.mock_data_samples,
        )

    def _load_calibration_prompts(self) -> list[str]:
        """
        Load calibration prompts for full-inference quantization calibration.

        Follows the same pattern as the PTQ workflow (examples/diffusers/quantization/):
        - If calibration_prompts_file is set: reads a text file with one prompt per line
        - Otherwise: loads from the HuggingFace dataset 'Gustavosta/Stable-Diffusion-Prompts'

        Returns:
            List of calibration prompts, truncated to calibration_size.
        """
        calib_size = self._distillation_config.calibration_size
        prompts_file = self._distillation_config.calibration_prompts_file

        if prompts_file is not None:
            prompts_path = Path(prompts_file)
            if not prompts_path.exists():
                raise FileNotFoundError(f"Calibration prompts file not found: {prompts_path}")
            logger.info(f"Loading calibration prompts from {prompts_path}")
            with open(prompts_path) as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            logger.info(
                "Loading calibration prompts from HuggingFace dataset "
                "'Gustavosta/Stable-Diffusion-Prompts'..."
            )
            from datasets import load_dataset

            dataset = load_dataset("Gustavosta/Stable-Diffusion-Prompts")
            prompts = list(dataset["train"]["Prompt"])

        # Truncate to requested size
        prompts = prompts[:calib_size]
        logger.info(f"Loaded {len(prompts)} calibration prompts")
        return prompts

    def _run_inference_calibration(self, model: torch.nn.Module) -> None:
        """
        Run full-inference calibration through the DiT, covering all noise levels.

        This replaces the old training-style calibration with full denoising inference,
        matching the PTQ workflow. For each calibration prompt, a complete denoising loop
        is run (e.g. 30 steps), so the transformer sees activations at every noise level.

        With CFG guidance_scale > 1.0 (default 4.0), each denoising step calls the
        transformer twice (positive + negative prompt), matching real inference patterns.

        Note: Text embeddings were pre-computed and cached in
        _load_text_encoder_and_cache_embeddings() BEFORE the Gemma model was unloaded.
        We pass these cached embeddings to the ValidationSampler via GenerationConfig.

        Args:
            model: The transformer model being calibrated (same reference as self._transformer,
                   with statistics collection enabled by mtq.quantize).
        """
        from ltx_trainer.validation_sampler import GenerationConfig, ValidationSampler

        calib_cfg = self._distillation_config
        if calib_cfg.calibration_size == 0:
            logger.info("Skipping calibration (calibration_size=0)")
            return

        if not self._cached_calibration_embeddings:
            raise RuntimeError(
                "No cached calibration embeddings available! "
                "Probably the saved checkpoint has no modelopt_state.pt or corrupted."
            )

        # Get video dimensions from validation config
        video_dims = getattr(self._config.validation, "video_dims", [512, 320, 33])
        width, height, num_frames = video_dims
        negative_prompt = getattr(
            self._config.validation,
            "negative_prompt",
            "worst quality, inconsistent motion, blurry, jittery, distorted",
        )
        num_prompts = len(self._cached_calibration_embeddings)

        logger.info(
            f"Running full-inference calibration: {num_prompts} prompts, "
            f"{calib_cfg.calibration_n_steps} steps/prompt, "
            f"guidance_scale={calib_cfg.calibration_guidance_scale}, "
            f"video={width}x{height}x{num_frames}"
        )

        # Create a ValidationSampler with the model being calibrated.
        # The exact model reference matters: mtq.quantize() sets up statistics
        # collection on this instance, so all forward passes must go through it.
        # text_encoder=None because we use pre-cached embeddings (Gemma is unloaded).
        sampler = ValidationSampler(
            transformer=model,
            vae_decoder=self._vae_decoder,
            vae_encoder=self._vae_encoder,
            text_encoder=None,  # Gemma unloaded; using cached embeddings
            audio_decoder=None,  # Skip audio for calibration
            vocoder=None,
        )

        device = "cuda"
        model.eval()

        with torch.no_grad():
            for i, cached_emb in enumerate(self._cached_calibration_embeddings):
                gen_config = GenerationConfig(
                    prompt="",  # Not used when cached_embeddings is provided
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_frames=num_frames,
                    frame_rate=getattr(self._config.validation, "frame_rate", 25.0),
                    num_inference_steps=calib_cfg.calibration_n_steps,
                    guidance_scale=calib_cfg.calibration_guidance_scale,
                    seed=42 + i,  # Vary seed per prompt for diverse activations
                    generate_audio=False,
                    tiled_decoding=False,  # Skip tiling overhead
                    cached_embeddings=cached_emb,  # Pre-computed text embeddings
                )

                try:
                    sampler.generate(config=gen_config, device=device)
                except Exception as e:
                    logger.warning(f"Calibration prompt {i} failed: {e}")
                    continue

                if (i + 1) % 10 == 0 or (i + 1) == len(self._cached_calibration_embeddings):
                    logger.info(f"Calibration progress: {i + 1}/{num_prompts} prompts")

        model.train()
        logger.info("Full-inference calibration complete")

    def _init_optimizer(self) -> None:
        """
        Override parent to prepare student model + optimizer + scheduler together.

        FSDP2 requires model and optimizer to be passed to accelerator.prepare()
        in a single call. This override:
        1. Creates the optimizer (pointing at self._transformer parameters)
        2. Creates the LR scheduler
        3. Calls accelerator.prepare(model, optimizer, scheduler) together

        This is compatible with both FSDP1 and FSDP2.
        """
        from torch.optim import AdamW

        opt_cfg = self._config.optimization

        lr = opt_cfg.learning_rate
        if opt_cfg.optimizer_type == "adamw":
            optimizer = AdamW(self._trainable_params, lr=lr)
        elif opt_cfg.optimizer_type == "adamw8bit":
            from bitsandbytes.optim import AdamW8bit

            optimizer = AdamW8bit(self._trainable_params, lr=lr)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_cfg.optimizer_type}")

        lr_scheduler = self._create_scheduler(optimizer)

        # Prepare student model + optimizer + scheduler together (FSDP2 requirement)
        logger.info("Preparing student model + optimizer + scheduler with accelerator...")
        if lr_scheduler is not None:
            self._transformer, self._optimizer, self._lr_scheduler = self._accelerator.prepare(
                self._transformer, optimizer, lr_scheduler
            )
        else:
            self._transformer, self._optimizer = self._accelerator.prepare(
                self._transformer, optimizer
            )
            self._lr_scheduler = None

        # Log memory after preparation
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after model+optimizer preparation: {mem_gb:.2f} GB")

    def _init_dataloader(self) -> None:
        """Override to support mock data for training."""
        if self._distillation_config.use_mock_data:
            from torch.utils.data import DataLoader

            self._dataset = self._create_mock_dataset()
            self._dataloader = DataLoader(
                self._dataset,
                batch_size=self._config.optimization.batch_size,
                shuffle=True,
                num_workers=self._config.data.num_dataloader_workers,
                pin_memory=True,
                drop_last=True,
            )
            # Wrap with accelerator
            self._dataloader = self._accelerator.prepare(self._dataloader)
        else:
            # Use parent implementation for real data
            super()._init_dataloader()

    def _load_teacher_model(self) -> None:
        """
        Load the teacher transformer model for distillation.

        The teacher is loaded, frozen, and prepared with the accelerator using a
        dummy SGD optimizer (lr=0, never stepped). The dummy optimizer is needed
        because FSDP2 requires model+optimizer together in prepare(). For FSDP1,
        this also works fine (prepare just wraps the model).
        """
        from torch.optim import SGD

        teacher_path = self._distillation_config.teacher_model_path
        if teacher_path is None:
            teacher_path = self._config.model.model_path

        # Map dtype string to torch dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        teacher_dtype = dtype_map[self._distillation_config.teacher_dtype]

        logger.info(
            f"Loading teacher model from {teacher_path} with dtype={self._distillation_config.teacher_dtype}"
        )

        # Load teacher transformer to CPU first
        self._teacher_transformer = load_transformer(
            checkpoint_path=str(teacher_path),
            device="cpu",
            dtype=teacher_dtype,
        )

        # Teacher is inference-only, freeze it
        self._teacher_transformer.requires_grad_(False)
        self._teacher_transformer.eval()

        # Prepare teacher with accelerator using a dummy optimizer.
        # FSDP2 requires model+optimizer together in prepare(). We use a minimal
        # SGD with lr=0 that will never be stepped — just to satisfy the API.
        logger.info(
            f"Preparing teacher model with accelerator (distributed_type={self._accelerator.distributed_type})"
        )
        teacher_params = list(self._teacher_transformer.parameters())
        dummy_optimizer = SGD(teacher_params, lr=0.0)

        self._teacher_transformer, wrapped_dummy_optimizer = self._accelerator.prepare(
            self._teacher_transformer, dummy_optimizer
        )

        # Remove the teacher model and dummy optimizer from accelerator's internal
        # tracking lists. This prevents save_state()/load_state() from saving/loading
        # the teacher (which is frozen and loaded fresh from the original checkpoint
        # on each run). The FSDP wrapping is already done at this point, so the
        # teacher doesn't need to stay registered.
        # Note: _models and _optimizers must stay 1:1 aligned for FSDP optimizer
        # save/load (load_fsdp_optimizer uses _models[i] to pair with _optimizers[i]).
        # We use the wrapped objects returned by prepare() since _optimizers stores
        # AcceleratedOptimizer wrappers, not raw optimizers.
        self._accelerator._models.remove(self._teacher_transformer)
        self._accelerator._optimizers.remove(wrapped_dummy_optimizer)

        # Re-freeze teacher after prepare (FSDP wrapping may reset requires_grad)
        self._teacher_transformer.requires_grad_(False)
        self._teacher_transformer.eval()

        # Log memory after teacher loading
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory after teacher preparation: {mem_gb:.2f} GB")

        logger.info(
            "Teacher model loaded and prepared (unregistered from accelerator state tracking)"
        )

    def _training_step(self, batch: dict[str, dict[str, Tensor]]) -> Tensor:
        """
        Perform a single distillation training step.

        Computes combined loss:
        L_total = alpha * L_task + (1 - alpha) * L_distill

        where:
        - L_task: MSE between student prediction and flow matching target
        - L_distill: MSE between student prediction and teacher prediction
        """
        alpha = self._distillation_config.distillation_alpha

        # Apply embedding connectors to transform pre-computed text embeddings
        conditions = batch["conditions"]
        video_embeds, audio_embeds, attention_mask = self._text_encoder._run_connectors(
            conditions["prompt_embeds"], conditions["prompt_attention_mask"]
        )
        conditions["video_prompt_embeds"] = video_embeds
        conditions["audio_prompt_embeds"] = audio_embeds
        conditions["prompt_attention_mask"] = attention_mask

        # Use strategy to prepare training inputs
        model_inputs = self._training_strategy.prepare_training_inputs(
            batch, self._timestep_sampler
        )

        # Run student forward pass
        student_video_pred, student_audio_pred = self._transformer(
            video=model_inputs.video,
            audio=model_inputs.audio,
            perturbations=None,
        )

        # Compute task loss only if alpha > 0
        if alpha > 0:
            task_loss = self._training_strategy.compute_loss(
                student_video_pred, student_audio_pred, model_inputs
            )
        else:
            task_loss = torch.tensor(0.0, device=student_video_pred.device)

        # Compute distillation loss only if alpha < 1
        if alpha < 1.0:
            # Run teacher forward pass (no gradients)
            with torch.no_grad():
                teacher_video_pred, _teacher_audio_pred = self._teacher_transformer(
                    video=model_inputs.video,
                    audio=model_inputs.audio,
                    perturbations=None,
                )

            # Compute distillation loss
            distill_loss = self._compute_distillation_loss(
                student_video_pred,
                teacher_video_pred,
                loss_mask=model_inputs.video_loss_mask,
            )
        else:
            distill_loss = torch.tensor(0.0, device=student_video_pred.device)

        # Combine losses
        total_loss = alpha * task_loss + (1.0 - alpha) * distill_loss

        # Log individual losses using parent's _log_metrics pattern (no explicit step)
        # This avoids step conflicts with wandb's auto-incrementing step counter
        if hasattr(self, "_accelerator") and self._accelerator.is_main_process:
            self._log_metrics(
                {
                    "loss/task": task_loss.item(),
                    "loss/distillation": distill_loss.item(),
                    "loss/total": total_loss.item(),
                }
            )

        return total_loss

    def _compute_distillation_loss(
        self,
        student_pred: Tensor,
        teacher_pred: Tensor,
        loss_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute distillation loss between student and teacher predictions."""
        loss_type = self._distillation_config.distillation_loss_type

        if loss_type == "mse":
            loss = torch.nn.functional.mse_loss(student_pred, teacher_pred, reduction="none")
        elif loss_type == "cosine":
            student_flat = student_pred.flatten(start_dim=2)
            teacher_flat = teacher_pred.flatten(start_dim=2)
            cos_sim = torch.nn.functional.cosine_similarity(student_flat, teacher_flat, dim=-1)
            loss = 1.0 - cos_sim.mean()
        else:
            raise ValueError(f"Unknown distillation loss type: {loss_type}")

        # Apply loss mask if provided
        # loss_mask is [B, seq_len], need to unsqueeze to [B, seq_len, 1] for broadcasting
        # with loss shape [B, seq_len, C]
        if loss_mask is not None:
            # Unsqueeze and convert to float for multiplication
            loss_mask = loss_mask.unsqueeze(-1).float()
            # Apply mask and normalize (same as original trainer)
            loss = loss.mul(loss_mask).div(loss_mask.mean())
            loss = loss.mean()
        else:
            loss = loss.mean()

        return loss

    def save_quantized_model(self, path: str | Path | None = None) -> None:
        """Save the quantized model using ModelOpt (global rank 0 only)."""
        if not is_global_rank0():
            return
        if path is None:
            path = self._distillation_config.save_quantized_checkpoint
        if path is None:
            path = Path(self._config.output_dir) / "quantized_model"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving quantized model to {path}")
        mto.save(self._transformer, str(path))
        logger.info("Quantized model saved successfully")

    # ── Overrides to fix multi-node shared-FS writes ──────────────────────
    # The parent trainer guards file writes with IS_MAIN_PROCESS (LOCAL_RANK==0),
    # which is True on every node.  We override to use is_global_rank0() so that
    # only a single process writes on a shared filesystem.

    def _save_checkpoint(self) -> Path | None:
        """Save model weights (override: use global rank 0 for file writes)."""
        from accelerate.utils import DistributedType
        from safetensors.torch import save_file

        is_lora = self._config.model.training_mode == "lora"
        is_fsdp = self._accelerator.distributed_type == DistributedType.FSDP

        save_dir = Path(self._config.output_dir) / "checkpoints"
        prefix = "lora" if is_lora else "model"
        filename = f"{prefix}_weights_step_{self._global_step:05d}.safetensors"
        saved_weights_path = save_dir / filename

        # Collective operation — all ranks must participate
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._transformer)

        if not is_global_rank0():
            return None

        save_dir.mkdir(exist_ok=True, parents=True)
        save_dtype = (
            torch.bfloat16 if self._config.checkpoints.precision == "bfloat16" else torch.float32
        )

        if is_lora:
            from peft import get_peft_model_state_dict

            unwrapped = self._accelerator.unwrap_model(self._transformer, keep_torch_compile=False)
            state_dict = get_peft_model_state_dict(
                unwrapped, state_dict=full_state_dict if is_fsdp else None
            )
            state_dict = {k.replace("base_model.model.", "", 1): v for k, v in state_dict.items()}
            state_dict = {f"diffusion_model.{k}": v for k, v in state_dict.items()}
            state_dict = {
                k: v.to(save_dtype) if isinstance(v, Tensor) else v for k, v in state_dict.items()
            }
            metadata = self._build_checkpoint_metadata()
            save_file(state_dict, saved_weights_path, metadata=metadata)
        else:
            full_state_dict = {
                k: v.to(save_dtype) if isinstance(v, Tensor) else v
                for k, v in full_state_dict.items()
            }
            self._accelerator.save(full_state_dict, saved_weights_path)

        rel_path = saved_weights_path.relative_to(self._config.output_dir)
        logger.info(f"Model weights for step {self._global_step} saved in {rel_path}")

        self._checkpoint_paths.append(saved_weights_path)
        return saved_weights_path

    def _save_config(self) -> None:
        """Save training config (override: use global rank 0 for file writes)."""
        if not is_global_rank0():
            return
        import yaml

        config_path = Path(self._config.output_dir) / "training_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(self._config.model_dump(), f, default_flow_style=False, indent=2)
        logger.info(
            f"Training configuration saved to: {config_path.relative_to(self._config.output_dir)}"
        )

    def _init_wandb(self) -> None:
        """Initialize W&B (override: use global rank 0 to avoid duplicate runs)."""
        if not self._config.wandb.enabled or not is_global_rank0():
            self._wandb_run = None
            return
        # Delegate to parent's implementation on global rank 0
        super()._init_wandb()

    def _get_checkpoints_dir(self) -> Path:
        """Return the directory used for full training state checkpoints."""
        return Path(self._config.output_dir) / "checkpoints"

    def _save_training_state(self) -> Path | None:
        """
        Save the full training state using accelerator.save_state().

        This saves everything needed to resume training exactly:
        - Student model weights (FSDP-sharded)
        - Optimizer state
        - LR scheduler state
        - RNG states (Python, NumPy, PyTorch CPU/CUDA per device)
        - Gradient scaler state (if using mixed precision)
        - ModelOpt state (quantization architecture for restore on resume)
        - Custom metadata (global_step, distillation config)

        Atomic save strategy:
            1. Save everything into step_XXXXXX_tmp/
            2. After all writes complete, rename to step_XXXXXX/
            Directory rename is atomic on the same filesystem, so either
            the final directory exists (complete) or it doesn't. If the
            process is killed mid-save, only the _tmp directory remains,
            which is cleaned up on the next run.

        Note: The teacher model is NOT saved here — it was unregistered from
        the accelerator's tracking lists after prepare() (see _load_teacher_model).
        On resume, the teacher is loaded fresh from the original pretrained checkpoint.

        Returns:
            Path to the saved state directory, or None on non-main processes.
        """
        final_dir = self._get_checkpoints_dir() / f"step_{self._global_step:06d}"
        tmp_dir = self._get_checkpoints_dir() / f"step_{self._global_step:06d}_tmp"

        logger.info(f"Saving full training state at step {self._global_step}...")

        # Ensure the checkpoints directory exists before save_state.
        if is_global_rank0():
            tmp_dir.mkdir(parents=True, exist_ok=True)
        self._accelerator.wait_for_everyone()

        # Save into the _tmp directory first (all ranks participate for FSDP).
        self._accelerator.save_state(str(tmp_dir))

        # Additional saves only on global rank 0 to avoid file write races.
        if is_global_rank0():
            # Save modelopt state for quantization architecture restoration on resume.
            if self._distillation_config.quant_cfg is not None:
                try:
                    modelopt_state_dict = mto.modelopt_state(self._transformer)
                    torch.save(modelopt_state_dict, tmp_dir / "modelopt_state.pt")
                    logger.debug("Saved modelopt_state.pt for resume")
                except Exception as e:
                    logger.warning(f"Failed to save modelopt_state: {e}")

            # Save custom metadata.
            metadata = {
                "global_step": self._global_step,
                "distillation_alpha": self._distillation_config.distillation_alpha,
                "quant_cfg": self._distillation_config.quant_cfg,
            }
            metadata_path = tmp_dir / "distillation_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Barrier: ensure all ranks finished writing before rename
        self._accelerator.wait_for_everyone()

        # Atomic rename _tmp → final (only global rank 0)
        if is_global_rank0():
            if tmp_dir.exists():
                tmp_dir.rename(final_dir)
                logger.info(f"Training state saved to {final_dir} (step={self._global_step})")
            else:
                logger.error(f"Save directory {tmp_dir} not found after save_state — skipping")

        # Cleanup old / incomplete checkpoints
        self._accelerator.wait_for_everyone()
        self._cleanup_checkpoints()

        self._accelerator.wait_for_everyone()
        return final_dir if is_global_rank0() else None

    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints, keeping only the last N.

        Also removes any *_tmp directories left behind by interrupted saves.
        """
        if not is_global_rank0():
            return

        import shutil

        ckpt_dir = self._get_checkpoints_dir()
        if not ckpt_dir.exists():
            return

        # Remove leftover _tmp directories from interrupted saves
        for tmp_dir in ckpt_dir.glob("step_*_tmp"):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            logger.info(f"Removed incomplete checkpoint: {tmp_dir.name}")

        # Keep only last N complete training checkpoints.
        # Exclude _tmp (incomplete) and _quantized (calibration-only, not training state).
        keep_n = self._config.checkpoints.keep_last_n
        if keep_n <= 0:
            return

        step_dirs = sorted(ckpt_dir.glob("step_[0-9]*"), key=lambda p: p.name)
        step_dirs = [
            d
            for d in step_dirs
            if not d.name.endswith("_tmp") and not d.name.endswith("_quantized")
        ]
        if len(step_dirs) <= keep_n:
            return

        dirs_to_remove = step_dirs[:-keep_n]
        for old_dir in dirs_to_remove:
            shutil.rmtree(old_dir, ignore_errors=True)
            logger.info(f"Removed old checkpoint: {old_dir.name}")

    def _find_resume_checkpoint(self, path_or_keyword: str | Path) -> Path | None:
        """
        Find the checkpoint directory to resume from.

        Only considers fully saved checkpoints (step_XXXXXX, not step_*_tmp).
        Incomplete _tmp checkpoints are ignored and cleaned up.

        Args:
            path_or_keyword: Either "latest" to auto-find, or an explicit path.

        Returns:
            Path to the checkpoint directory, or None if not found.
        """
        if str(path_or_keyword).lower() == "latest":
            ckpt_dir = self._get_checkpoints_dir()
            if not ckpt_dir.exists():
                logger.warning(f"No checkpoints directory found at {ckpt_dir}")
                return None

            # Only match step_XXXXXX (6 digits), excluding _tmp (incomplete saves)
            # and _quantized (step 0 calibration-only checkpoint, no training state).
            step_dirs = sorted(ckpt_dir.glob("step_[0-9]*"), key=lambda p: p.name)
            step_dirs = [
                d
                for d in step_dirs
                if not d.name.endswith("_tmp") and not d.name.endswith("_quantized")
            ]
            if not step_dirs:
                logger.warning(f"No complete checkpoints found in {ckpt_dir}")
                return None

            latest = step_dirs[-1]
            logger.info(f"Auto-detected latest checkpoint: {latest}")
            return latest
        else:
            path = Path(path_or_keyword)
            if not path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {path}")
            return path

    def _load_training_state(self, checkpoint_dir: Path) -> int:
        """
        Load full training state from a checkpoint directory.

        Note: The quantization architecture (fake quantizer modules) must already be
        restored BEFORE this method is called. This happens in _apply_modelopt_quantization()
        (Path A) which uses mto.restore_from_modelopt_state() to set up the module structure.
        This method then loads the trained weights (including quantizer scales) into that
        structure via accelerator.load_state().

        This restores (all via accelerator.load_state()):
        - Model weights (student, FSDP-sharded, including quantizer scales)
        - Optimizer state
        - LR scheduler state
        - Dataloader iteration position (auto-skips consumed batches)
        - RNG states (Python, NumPy, PyTorch CPU/CUDA per device)
        - Gradient scaler (mixed precision)
        - global_step (from custom metadata file)

        Args:
            checkpoint_dir: Path to the training state checkpoint directory.

        Returns:
            The global_step to resume from.
        """
        logger.info(f"Resuming training state from {checkpoint_dir}...")

        # accelerator.load_state() is a collective op — all ranks must call it.
        # It restores all objects registered via accelerator.prepare() in order:
        #   1. Student model weights (self._transformer) — including quantizer scales
        #   2. Optimizer state (self._optimizer)
        #   3. LR scheduler state (self._lr_scheduler)
        #   4. Dataloader iteration position (via skip_first_batches internally)
        #   5. RNG states (Python, NumPy, PyTorch CPU/CUDA per device)
        #   6. Gradient scaler (mixed precision)
        # Note: Teacher model was unregistered from accelerator (see _load_teacher_model),
        # so it is NOT loaded here — it is loaded fresh from pretrained on each run.
        self._accelerator.load_state(str(checkpoint_dir))
        logger.info(
            "Restored: student model (with quantizer scales), optimizer, LR scheduler, "
            "dataloader position, RNG states, and gradient scaler via accelerator.load_state()"
        )

        # Load custom metadata to get global_step
        metadata_path = checkpoint_dir / "distillation_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            resumed_step = metadata.get("global_step", 0)
            logger.info(f"Restored global_step={resumed_step} from metadata")
        else:
            # Fallback: try to parse step from directory name
            try:
                resumed_step = int(checkpoint_dir.name.split("_")[-1])
                logger.warning(
                    f"Metadata file not found, parsed step from dir name: {resumed_step}"
                )
            except (ValueError, IndexError):
                resumed_step = 0
                logger.warning("Could not determine step from checkpoint, resuming from step 0")

        return resumed_step

    def train(
        self,
        disable_progress_bars: bool = False,
        step_callback=None,
    ) -> tuple[Path | None, dict]:
        """
        Override parent train() to add full checkpoint resume support.

        When `distillation.resume_from_checkpoint` is set, this:
        1. Initializes optimizer/dataloader/scheduler as normal
        2. Loads full training state (model, optimizer, scheduler, RNG)
        3. Skips already-completed steps
        4. Saves full training state at checkpoint intervals
        """
        from accelerate.utils import DistributedType, set_seed
        from ltx_trainer.gpu_utils import get_gpu_memory_gb
        from ltx_trainer.hf_hub_utils import push_to_hub
        from ltx_trainer.progress import TrainingProgress
        from ltx_trainer.trainer import TrainingStats

        MEMORY_CHECK_INTERVAL = 200  # noqa: N806

        device = self._accelerator.device
        cfg = self._config
        start_mem = get_gpu_memory_gb(device)

        train_start_time = time.time()

        # Use the same seed for all processes and ensure deterministic operations
        set_seed(cfg.seed)
        logger.debug(f"Process {self._accelerator.process_index} using seed: {cfg.seed}")

        self._init_optimizer()
        self._init_dataloader()
        self._init_timestep_sampler()

        # Synchronize all processes after initialization
        self._accelerator.wait_for_everyone()

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

        # Save the training configuration as YAML
        self._save_config()

        # =====================================================================
        # Resume from checkpoint if configured
        # =====================================================================
        resume_step = 0
        resume_path = self._distillation_config.resume_from_checkpoint
        if resume_path is not None:
            checkpoint_dir = self._find_resume_checkpoint(resume_path)
            if checkpoint_dir is not None:
                resume_step = self._load_training_state(checkpoint_dir)
                logger.info(f"Resuming training from step {resume_step}")
            else:
                logger.warning("No checkpoint found to resume from, starting from scratch")

        # Create the dataloader iterator AFTER load_state() so it picks up the
        # resumed dataloader state. accelerator.load_state() automatically replaces
        # self._dataloader with a version that skips already-consumed batches
        # (via skip_first_batches), so iter() here starts at the correct position.
        data_iter = iter(self._dataloader)

        # Timer for Slurm time-limit-aware checkpointing
        must_save_by_minutes = self._distillation_config.must_save_by
        if must_save_by_minutes is not None:
            save_deadline = train_start_time + must_save_by_minutes * 60
            logger.info(
                f"Time-limit save enabled: will save and exit after "
                f"{must_save_by_minutes:.1f} minutes"
            )
        else:
            save_deadline = None

        logger.info("Starting training...")
        config_msg = (
            f"Config: steps={cfg.optimization.steps}, "
            f"grad_accum={cfg.optimization.gradient_accumulation_steps}, "
            f"checkpoints.interval={cfg.checkpoints.interval}, "
            f"checkpoints.keep_last_n={cfg.checkpoints.keep_last_n}, "
            f"output_dir={cfg.output_dir}, "
            f"must_save_by={must_save_by_minutes}"
        )
        logger.info(config_msg)
        # Also print to stdout (logger goes to stderr via RichHandler,
        # which lands in .err files in Slurm)
        if IS_MAIN_PROCESS:
            print(f"[distillation_trainer] {config_msg}", flush=True)

        # Create progress tracking
        progress_enabled = IS_MAIN_PROCESS and not disable_progress_bars
        progress = TrainingProgress(
            enabled=progress_enabled,
            total_steps=cfg.optimization.steps,
        )

        if IS_MAIN_PROCESS and disable_progress_bars:
            logger.warning(
                "Progress bars disabled. Intermediate status messages will be logged instead."
            )

        self._transformer.train()
        self._global_step = resume_step

        peak_mem_during_training = start_mem

        sampled_videos_paths = None

        # Calculate how many raw steps to skip and how many to run
        total_raw_steps = cfg.optimization.steps * cfg.optimization.gradient_accumulation_steps
        skip_raw_steps = resume_step * cfg.optimization.gradient_accumulation_steps

        with progress:
            # Initial validation before training starts (skip if resuming)
            if (
                resume_step == 0
                and cfg.validation.interval
                and not cfg.validation.skip_initial_validation
            ):
                sampled_videos_paths = self._sample_videos(progress)
                if (
                    IS_MAIN_PROCESS
                    and sampled_videos_paths
                    and self._config.wandb.log_validation_videos
                ):
                    self._log_validation_samples(sampled_videos_paths, cfg.validation.prompts)

            self._accelerator.wait_for_everyone()

            # Accumulators for averaging metrics across gradient accumulation steps
            grad_accum_steps = cfg.optimization.gradient_accumulation_steps
            accum_loss = 0.0
            accum_step_time = 0.0

            for step in range(skip_raw_steps, total_raw_steps):
                # Get next batch, reset the dataloader if needed
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self._dataloader)
                    batch = next(data_iter)

                step_start_time = time.time()
                with self._accelerator.accumulate(self._transformer):
                    is_optimization_step = (step + 1) % grad_accum_steps == 0
                    if is_optimization_step:
                        self._global_step += 1

                    loss = self._training_step(batch)
                    self._accelerator.backward(loss)

                    # Accumulate metrics for this micro-batch
                    accum_loss += loss.item()
                    accum_step_time += time.time() - step_start_time

                    if self._accelerator.sync_gradients and cfg.optimization.max_grad_norm > 0:
                        self._accelerator.clip_grad_norm_(
                            self._trainable_params,
                            cfg.optimization.max_grad_norm,
                        )

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    if self._lr_scheduler is not None:
                        self._lr_scheduler.step()

                    # Run validation if needed
                    if (
                        cfg.validation.interval
                        and self._global_step > 0
                        and self._global_step % cfg.validation.interval == 0
                        and is_optimization_step
                    ):
                        if self._accelerator.distributed_type == DistributedType.FSDP:
                            sampled_videos_paths = self._sample_videos(progress)
                            if (
                                IS_MAIN_PROCESS
                                and sampled_videos_paths
                                and self._config.wandb.log_validation_videos
                            ):
                                self._log_validation_samples(
                                    sampled_videos_paths, cfg.validation.prompts
                                )
                        elif IS_MAIN_PROCESS:
                            sampled_videos_paths = self._sample_videos(progress)
                            if sampled_videos_paths and self._config.wandb.log_validation_videos:
                                self._log_validation_samples(
                                    sampled_videos_paths, cfg.validation.prompts
                                )

                    # Save training state for resuming (model, optimizer, scheduler,
                    # dataloader position, RNG states — all handled by accelerator)
                    saved_this_step = False
                    ckpt_interval = cfg.checkpoints.interval
                    if (
                        ckpt_interval
                        and self._global_step > 0
                        and self._global_step % ckpt_interval == 0
                        and is_optimization_step
                    ):
                        logger.info(
                            f"Saving checkpoint at step {self._global_step} "
                            f"(interval={ckpt_interval})..."
                        )
                        self._save_training_state()
                        saved_this_step = True

                    # Time-limit save: if we're approaching the Slurm time limit,
                    # save a checkpoint and exit gracefully.
                    if (
                        save_deadline is not None
                        and is_optimization_step
                        and time.time() >= save_deadline
                    ):
                        elapsed_min = (time.time() - train_start_time) / 60
                        logger.info(
                            f"Time limit reached ({elapsed_min:.1f} min >= "
                            f"{must_save_by_minutes:.1f} min). "
                            f"Saving checkpoint at step {self._global_step} and exiting..."
                        )
                        if not saved_this_step:
                            self._save_training_state()
                        # Break out of the training loop; post-loop code
                        # will collect stats and return.
                        break

                    self._accelerator.wait_for_everyone()

                    # Call step callback if provided
                    if step_callback and is_optimization_step:
                        step_callback(
                            self._global_step, cfg.optimization.steps, sampled_videos_paths
                        )

                    self._accelerator.wait_for_everyone()

                    # On optimization steps: compute averaged metrics, log, then reset
                    if is_optimization_step:
                        avg_loss = accum_loss / grad_accum_steps
                        total_step_time = accum_step_time

                        current_lr = self._optimizer.param_groups[0]["lr"]

                        progress.update_training(
                            loss=avg_loss,
                            lr=current_lr,
                            step_time=total_step_time,
                            advance=True,
                        )

                        # Log averaged metrics to W&B
                        if IS_MAIN_PROCESS:
                            self._log_metrics(
                                {
                                    "train/loss": avg_loss,
                                    "train/learning_rate": current_lr,
                                    "train/step_time": total_step_time,
                                    "train/global_step": self._global_step,
                                }
                            )

                        # Periodic step logging to console/Slurm logs
                        if IS_MAIN_PROCESS and self._global_step % 10 == 0:
                            elapsed = time.time() - train_start_time
                            progress_pct = self._global_step / cfg.optimization.steps
                            if progress_pct > 0:
                                eta = (elapsed / progress_pct) - elapsed
                                eta_str = f"{eta // 3600:.0f}h {(eta % 3600) // 60:.0f}m"
                            else:
                                eta_str = "calculating..."
                            logger.info(
                                f"Step {self._global_step}/{cfg.optimization.steps} | "
                                f"Loss: {avg_loss:.4f} | LR: {current_lr:.2e} | "
                                f"Time/Step: {total_step_time:.2f}s | ETA: {eta_str}",
                            )

                        # Reset accumulators
                        accum_loss = 0.0
                        accum_step_time = 0.0

                    # Sample GPU memory periodically
                    if step % MEMORY_CHECK_INTERVAL == 0:
                        current_mem = get_gpu_memory_gb(device)
                        peak_mem_during_training = max(peak_mem_during_training, current_mem)

        # Collect final stats
        train_end_time = time.time()
        end_mem = get_gpu_memory_gb(device)
        peak_mem = max(start_mem, end_mem, peak_mem_during_training)

        total_time_seconds = train_end_time - train_start_time
        actual_steps = self._global_step - resume_step
        steps_per_second = actual_steps / total_time_seconds if total_time_seconds > 0 else 0
        samples_per_second = (
            steps_per_second * self._accelerator.num_processes * cfg.optimization.batch_size
        )

        stats = TrainingStats(
            total_time_seconds=total_time_seconds,
            steps_per_second=steps_per_second,
            samples_per_second=samples_per_second,
            peak_gpu_memory_gb=peak_mem,
            num_processes=self._accelerator.num_processes,
            global_batch_size=cfg.optimization.batch_size * self._accelerator.num_processes,
        )

        # Save final training state (for potential resume)
        self._save_training_state()

        # Save inference-ready model weights (standalone safetensors file)
        saved_path = self._save_checkpoint()

        if is_global_rank0():
            self._log_training_stats(stats)

            if cfg.hub.push_to_hub:
                push_to_hub(saved_path, sampled_videos_paths, self._config)

            if self._wandb_run is not None:
                self._log_metrics(
                    {
                        "stats/total_time_minutes": stats.total_time_seconds / 60,
                        "stats/steps_per_second": stats.steps_per_second,
                        "stats/samples_per_second": stats.samples_per_second,
                        "stats/peak_gpu_memory_gb": stats.peak_gpu_memory_gb,
                    }
                )
                self._wandb_run.finish()

        self._accelerator.wait_for_everyone()
        self._accelerator.end_training()

        return saved_path, stats


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LTX-2 Distillation Training with ModelOpt Quantization",
        # Allow OmegaConf-style overrides to pass through
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )

    # Parse known args to allow for OmegaConf overrides
    args, overrides = parser.parse_known_args()
    return args, overrides


def main():
    """Main entry point for distillation training."""
    # CRITICAL: Set CUDA device BEFORE any model loading.
    #
    # The LTX trainer loads the text encoder in __init__ BEFORE _setup_accelerator(),
    # using device="cuda" which defaults to GPU 0. We must set the device early
    # so that "cuda" maps to the correct GPU for each process.
    #
    # Note: We do NOT call init_process_group() here - let accelerate handle that.
    # We only set the CUDA device based on LOCAL_RANK.

    # Read distributed environment variables (set by accelerate launch / torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = os.environ.get("MASTER_PORT", "29500")

    # Debug: Print all relevant environment variables
    print(
        f"[DEBUG] PID={os.getpid()} RANK={rank} LOCAL_RANK={local_rank} "
        f"WORLD_SIZE={world_size} MASTER_ADDR={master_addr} MASTER_PORT={master_port}"
    )
    print(f"[DEBUG] torch.cuda.device_count()={torch.cuda.device_count()}")

    # Set CUDA device based on LOCAL_RANK - this ensures device="cuda" uses correct GPU
    if torch.cuda.is_available() and local_rank < torch.cuda.device_count():
        torch.cuda.set_device(local_rank)
        print(
            f"[DEBUG] Set CUDA device to {local_rank}, current device: {torch.cuda.current_device()}"
        )
    else:
        print(f"[WARNING] LOCAL_RANK={local_rank} but device_count={torch.cuda.device_count()}")

    logger.info(f"Process RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}")

    args, cli_overrides = parse_args()

    # Load base config from YAML using OmegaConf
    base_config = OmegaConf.load(args.config)

    # Parse CLI overrides using OmegaConf
    # Supports formats like:
    #   distillation.distillation_alpha=0.6
    #   ++distillation.quant_cfg=FP8_DEFAULT_CFG
    #   model.training_mode=lora
    if cli_overrides:
        # Clean up override strings (remove leading ++, +, etc.)
        cleaned_overrides = []
        for override in cli_overrides:
            # Strip leading + or ++ (Hydra-style)
            clean = override.lstrip("+")
            if "=" in clean:
                cleaned_overrides.append(clean)
            elif IS_MAIN_PROCESS:
                logger.warning(f"Ignoring malformed override: {override}")

        if cleaned_overrides:
            cli_config = OmegaConf.from_dotlist(cleaned_overrides)
            # Merge CLI overrides into base config (CLI takes precedence)
            config = OmegaConf.merge(base_config, cli_config)
            if IS_MAIN_PROCESS:
                logger.info(f"Applied {len(cleaned_overrides)} config overrides:")
                for override in cleaned_overrides:
                    logger.info(f"  {override}")
        else:
            config = base_config
    else:
        config = base_config

    # Convert OmegaConf to plain dict for Pydantic
    config_dict = OmegaConf.to_container(config, resolve=True)

    # Create typed config object
    config = DistillationTrainerConfig(**config_dict)

    # Create trainer and run
    trainer = DistillationTrainer(config)

    # Train
    saved_path, stats = trainer.train()

    # Save quantized model if configured
    if config.distillation.quant_cfg is not None:
        trainer.save_quantized_model()

    if IS_MAIN_PROCESS:
        logger.info(f"Training complete. Model saved to: {saved_path}")
        logger.info(f"Training stats: {stats}")


if __name__ == "__main__":
    main()
