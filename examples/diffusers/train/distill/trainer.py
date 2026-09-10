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

"""Unified distillation trainer for QAD/SAD of diffusion models.

Supports quantization-aware and sparsity-aware distillation using ModelOpt,
with Accelerate + FSDP for distributed training. Model-specific logic is
delegated to pluggable interfaces (ModelLoader, TrainingForwardAdapter, InferencePipeline).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch import Tensor
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Callable

    from .config import TrainerConfig

from .dataset import LatentDataset, MockDataset, create_dataloader
from .feature_extractor import FeatureExtractor
from .interfaces import (
    CachedEmbeddings,
    InferencePipeline,
    ModelLoader,
    TrainingForwardAdapter,
    free_gpu_memory,
)
from .utils import get_seq_length, is_global_rank0, resolve_layer_pairs, to_dtype

logger = logging.getLogger(__name__)


class DistillationTrainer:
    """Unified trainer for quantization/sparsity-aware distillation of diffusion models.

    The trainer owns the training loop, distillation loss, ModelOpt integration,
    optimizer, checkpointing, and logging. Model-specific behavior is provided
    by the three pluggable interfaces passed at construction time.
    """

    def __init__(
        self,
        config: TrainerConfig,
        model_loader: ModelLoader,
        training_adapter: TrainingForwardAdapter,
        inference_pipeline: InferencePipeline | None = None,
    ) -> None:
        self._config = config
        self._loader = model_loader
        self._adapter = training_adapter
        self._inference_pipeline = inference_pipeline

        self._global_step = 0
        self._data_epoch = 0
        self._wandb_run = None

        set_seed(config.seed)
        self._accelerator = Accelerator(
            gradient_accumulation_steps=config.optimization.gradient_accumulation_steps,
        )

        model_dtype = to_dtype(config.model.dtype)
        self._weight_dtype = model_dtype

        # --- Load inference components and cache embeddings (before training) ---
        self._cached_val_embeds: list[CachedEmbeddings] = []
        self._cached_calib_embeds: list[CachedEmbeddings] = []
        if self._inference_pipeline is not None:
            self._init_inference_pipeline(model_dtype)

        # --- Load student and teacher ---
        logger.info("Loading student model ...")
        self._student = self._loader.load_transformer(
            config.model.model_path, device="cpu", dtype=model_dtype
        )

        logger.info("Loading teacher model ...")
        teacher_path = config.distillation.teacher_model_path or config.model.model_path
        teacher_dtype = to_dtype(config.distillation.teacher_dtype)
        self._teacher = self._loader.load_transformer(
            teacher_path, device="cpu", dtype=teacher_dtype
        )
        self._teacher.requires_grad_(False)
        self._teacher.eval()

        # --- Optimizer ---
        self._init_optimizer()

        # --- Prepare with accelerator (model + optimizer + scheduler together for FSDP2) ---
        self._prepare_accelerator()

        # --- Apply ModelOpt quantization after FSDP wrapping ---
        if config.distillation.quant_cfg is not None:
            self._apply_quantization()

        # --- Layer-wise distillation hooks ---
        self._layer_pairs: list[tuple[str, str]] = []
        self._student_extractor: FeatureExtractor | None = None
        self._teacher_extractor: FeatureExtractor | None = None
        if config.distillation.layer_distillation_modules:
            self._init_layer_distillation()

        # --- Dataloader ---
        self._init_dataloaders()

        # --- WandB ---
        if config.wandb.enabled and is_global_rank0():
            self._init_wandb()

        # --- Resume ---
        if config.distillation.resume_from_checkpoint is not None:
            self._load_training_state()

    def _init_inference_pipeline(self, dtype: torch.dtype) -> None:
        cfg = self._config
        pipeline = self._inference_pipeline
        assert pipeline is not None

        logger.info("Loading inference pipeline components (text encoder, VAE) ...")
        pipeline.load_components(cfg.model, "cuda", dtype)

        # Cache validation prompt embeddings
        if cfg.validation.prompts:
            logger.info(
                f"Caching embeddings for {len(cfg.validation.prompts)} validation prompts ..."
            )
            self._cached_val_embeds = pipeline.encode_prompts(
                cfg.validation.prompts, cfg.validation.negative_prompt, "cuda"
            )

        # Cache calibration embeddings (only if fresh calibration will be needed)
        if cfg.distillation.quant_cfg and cfg.distillation.calibration_size > 0:
            if self._needs_fresh_calibration():
                calib_prompts = self._load_calibration_prompts()
                if calib_prompts:
                    logger.info(
                        f"Caching embeddings for {len(calib_prompts)} calibration prompts ..."
                    )
                    self._cached_calib_embeds = pipeline.encode_prompts(
                        calib_prompts, cfg.validation.negative_prompt, "cuda"
                    )
            else:
                logger.info("Skipping calibration embedding caching (existing checkpoint found)")

        # Unload heavy text encoder backbone. Backends that need lightweight
        # components (e.g. LTX-2 connectors) keep them for process_text_embeddings().
        logger.info("Unloading text encoder ...")
        pipeline.unload_text_encoder()

        # Move VAE to CPU (will be moved to GPU on-demand during generate())
        pipeline.offload_to_cpu()

    def _load_calibration_prompts(self) -> list[str]:
        cfg = self._config.distillation
        if cfg.calibration_prompts_file:
            with open(cfg.calibration_prompts_file) as f:
                prompts = [line.strip() for line in f if line.strip()]
            return prompts[: cfg.calibration_size]
        # Default: use validation prompts repeated to fill calibration_size
        if self._config.validation.prompts:
            base = self._config.validation.prompts
            repeats = (cfg.calibration_size + len(base) - 1) // len(base)
            return (base * repeats)[: cfg.calibration_size]
        return []

    def _needs_fresh_calibration(self) -> bool:
        """Check whether fresh quantization calibration is needed.

        Returns False if an existing checkpoint can restore the quantized model,
        so we can skip the expensive calibration embedding caching.
        """
        cfg = self._config.distillation

        # Path A: resume checkpoint with modelopt_state.pt
        if cfg.resume_from_checkpoint is not None:
            checkpoint_dir = self._find_resume_checkpoint()
            if checkpoint_dir is not None and (checkpoint_dir / "modelopt_state.pt").exists():
                return False

        # Path B: user-specified quantized checkpoint
        if cfg.restore_quantized_checkpoint is not None:
            return False

        # Path B2: auto-detected step 0 quantized checkpoint
        step0_dir = self._get_checkpoints_dir() / "step_000000_quantized"
        return not (step0_dir / "modelopt_state.pt").exists()

    def _apply_quantization(self) -> None:
        """Apply ModelOpt fake quantization to the student model.

        Called after accelerator.prepare() so the model is already FSDP-wrapped
        and sharded across GPUs. Three paths are supported (checked in order):

        Path A - Resume from training checkpoint:
            Restore only the quantization architecture from modelopt_state.pt.
            The trained weights (including quantizer scales) are loaded later
            by accelerator.load_state() in _load_training_state().

        Path B2 - Auto-detect step 0 quantized checkpoint:
            If a previous run completed calibration and saved step 0, restore
            the architecture and FSDP state from it to avoid re-calibration.

        Path C - Fresh quantization with calibration:
            Run mtq.quantize() with a calibration forward loop. After calibration,
            save the result as step 0 for future runs.
        """
        import modelopt.torch.opt as mto
        import modelopt.torch.quantization as mtq

        cfg = self._config.distillation

        # Path B: Restore from user-specified quantized checkpoint
        if cfg.restore_quantized_checkpoint:
            logger.info(f"Restoring quantized model from {cfg.restore_quantized_checkpoint}")
            mto.restore(self._student, str(cfg.restore_quantized_checkpoint))
            return

        assert cfg.quant_cfg is not None, "quant_cfg must be set for quantization"
        quant_config = getattr(mtq, cfg.quant_cfg, None)
        if quant_config is None:
            raise ValueError(f"Unknown ModelOpt quant config: {cfg.quant_cfg}")

        # Path A: Resume from training checkpoint — restore architecture only.
        # The trained weights are loaded later by accelerator.load_state().
        if cfg.resume_from_checkpoint is not None:
            checkpoint_dir = self._find_resume_checkpoint()
            if checkpoint_dir is not None:
                modelopt_path = checkpoint_dir / "modelopt_state.pt"
                if modelopt_path.exists():
                    logger.info(
                        f"Resuming: restoring quantization architecture from "
                        f"{modelopt_path} (weights loaded later by accelerator)"
                    )
                    state = torch.load(modelopt_path, weights_only=False, map_location="cpu")
                    mto.restore_from_modelopt_state(self._student, state)
                    logger.info("Quantization architecture restored for resume")
                    return
                logger.warning(
                    f"modelopt_state.pt not found in {checkpoint_dir}, "
                    "falling through to fresh quantization"
                )

        # Path B2: Auto-detect step 0 quantized checkpoint.
        step0_dir = self._get_checkpoints_dir() / "step_000000_quantized"
        step0_modelopt = step0_dir / "modelopt_state.pt"
        if step0_modelopt.exists():
            logger.info(f"Found step 0 quantized checkpoint at {step0_dir}, restoring ...")
            try:
                state = torch.load(step0_modelopt, weights_only=False, map_location="cpu")
                mto.restore_from_modelopt_state(self._student, state)
                self._accelerator.load_state(str(step0_dir))
                logger.info("Step 0 quantized checkpoint restored (calibration skipped)")
                return
            except Exception as e:
                logger.warning(
                    f"Failed to restore step 0 checkpoint: {e}. "
                    "Falling through to fresh quantization."
                )

        # Path C: Fresh quantization with calibration.
        logger.info(f"Applying quantization config: {cfg.quant_cfg}")

        def calibration_forward_loop(model: nn.Module) -> None:
            if not self._cached_calib_embeds:
                logger.warning("No calibration embeddings available, skipping calibration loop")
                return
            # mtq.quantize unwraps FSDP via apply_mode → unwrap_model(force_unwrap=True),
            # so `model` here is the inner (unwrapped) module. We must call through
            # self._student (the FSDP wrapper) so the forward hooks trigger all-gather.
            self._run_calibration(self._student)

        mtq.quantize(self._student, quant_config, forward_loop=calibration_forward_loop)

        # Free cached calibration embeddings — no longer needed
        self._cached_calib_embeds = []

        # Save as step 0 quantized checkpoint for future runs.
        self._save_step0_quantized()

    def _save_step0_quantized(self) -> None:
        """Save quantized+calibrated model as step 0 checkpoint.

        Saves both the modelopt architecture state and the full FSDP model state
        (including calibrated quantizer scales). This avoids re-running calibration
        on future runs — Path B2 auto-detects and restores from this checkpoint.
        """
        import modelopt.torch.opt as mto

        step0_dir = self._get_checkpoints_dir() / "step_000000_quantized"

        if is_global_rank0():
            step0_dir.mkdir(parents=True, exist_ok=True)
        self._accelerator.wait_for_everyone()

        # Save FSDP sharded model + optimizer + scheduler state (all ranks participate)
        self._accelerator.save_state(str(step0_dir))
        self._accelerator.wait_for_everyone()

        # Save modelopt architecture metadata (rank 0 only)
        if is_global_rank0():
            try:
                modelopt_state = mto.modelopt_state(self._student)
                torch.save(modelopt_state, step0_dir / "modelopt_state.pt")
                logger.info(f"Step 0 quantized checkpoint saved to {step0_dir}")
            except Exception as e:
                logger.warning(f"Failed to save step 0 modelopt state: {e}")

        self._accelerator.wait_for_everyone()

    def _run_calibration(self, model: nn.Module) -> None:
        """Run calibration forward passes through the FSDP-wrapped model."""
        if self._inference_pipeline is None:
            logger.warning("No inference pipeline, skipping calibration")
            return

        logger.info(f"Running calibration with {len(self._cached_calib_embeds)} prompts ...")
        gen_config = {
            "width": self._config.validation.video_dims[0],
            "height": self._config.validation.video_dims[1],
            "num_frames": self._config.validation.video_dims[2],
            "num_inference_steps": self._config.distillation.calibration_n_steps,
            "guidance_scale": self._config.distillation.calibration_guidance_scale,
            "seed": self._config.seed,
        }

        with torch.no_grad():
            for i, cached_emb in enumerate(self._cached_calib_embeds):
                self._inference_pipeline.generate(
                    model=model,
                    cached_embeds=[cached_emb],
                    config=gen_config,
                    device="cuda",
                )
                if is_global_rank0() and (i + 1) % 10 == 0:
                    logger.info(f"  Calibration: {i + 1}/{len(self._cached_calib_embeds)}")

    def _init_optimizer(self) -> None:
        cfg = self._config.optimization
        trainable_params = [p for p in self._student.parameters() if p.requires_grad]
        logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

        if cfg.optimizer_type == "adamw":
            self._optimizer = torch.optim.AdamW(trainable_params, lr=cfg.learning_rate)
        elif cfg.optimizer_type == "adamw8bit":
            try:
                import bitsandbytes as bnb

                self._optimizer = bnb.optim.AdamW8bit(trainable_params, lr=cfg.learning_rate)
            except ImportError:
                raise ImportError("adamw8bit requires bitsandbytes: pip install bitsandbytes")
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer_type}")

        # LR scheduler
        total_steps = cfg.steps
        warmup = cfg.warmup_steps
        if cfg.scheduler_type == "cosine":
            self._lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer, T_max=total_steps - warmup
            )
        elif cfg.scheduler_type == "linear":
            self._lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self._optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps - warmup
            )
        elif cfg.scheduler_type == "constant":
            self._lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                self._optimizer, factor=1.0, total_iters=total_steps
            )
        else:
            raise ValueError(f"Unknown scheduler: {cfg.scheduler_type}")

        if warmup > 0:
            self._lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self._optimizer,
                schedulers=[
                    torch.optim.lr_scheduler.LinearLR(
                        self._optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup
                    ),
                    self._lr_scheduler,
                ],
                milestones=[warmup],
            )

    def _prepare_accelerator(self) -> None:
        """Prepare models and optimizer with accelerator.

        FSDP2 requires model + optimizer to be prepared together. The teacher
        is prepared separately with a dummy optimizer, then unregistered from
        accelerator state tracking so it's not saved/loaded with checkpoints.
        """
        from torch.optim import SGD

        # Prepare teacher with dummy optimizer (required for FSDP wrapping)
        teacher_params = list(self._teacher.parameters())
        dummy_opt = SGD(teacher_params, lr=0.0)
        self._teacher, wrapped_dummy = self._accelerator.prepare(self._teacher, dummy_opt)
        # Unregister teacher from accelerator so it's excluded from save/load
        self._accelerator._models.remove(self._teacher)
        self._accelerator._optimizers.remove(wrapped_dummy)
        self._teacher.requires_grad_(False)
        self._teacher.eval()

        # Prepare student + optimizer + scheduler together
        self._student, self._optimizer, self._lr_scheduler = self._accelerator.prepare(
            self._student, self._optimizer, self._lr_scheduler
        )

    def _init_dataloaders(self) -> None:
        cfg = self._config

        if cfg.distillation.use_mock_data:
            mock_shape = getattr(self._adapter, "MOCK_LATENT_SHAPE", (48, 4, 32, 32))
            mock_text_dim = getattr(self._adapter, "MOCK_TEXT_EMBED_DIM", 4096)
            mock_audio_shape = (
                getattr(self._adapter, "MOCK_AUDIO_LATENT_SHAPE", None)
                if cfg.distillation.with_audio
                else None
            )
            mock_kwargs = {
                "latent_shape": mock_shape,
                "text_embed_dim": mock_text_dim,
                "audio_latent_shape": mock_audio_shape,
                "dtype": self._weight_dtype,
            }
            train_ds = MockDataset(
                num_samples=cfg.distillation.mock_data_samples,
                **mock_kwargs,
            )
            val_ds = MockDataset(
                num_samples=min(16, cfg.distillation.mock_data_samples),
                **mock_kwargs,
            )
        elif cfg.data.preprocessed_data_root:
            full_ds = LatentDataset(cfg.data.preprocessed_data_root)
            val_size = min(len(full_ds) // 10, 64)
            train_size = len(full_ds) - val_size
            train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
        else:
            raise ValueError(
                "Either data.preprocessed_data_root or distillation.use_mock_data must be set"
            )

        # Don't pass distributed=True here: accelerator.prepare() adds its
        # own DistributedSampler.  Doing both causes double-sharding and an
        # empty dataloader when the dataset is small.
        self._dataloader = create_dataloader(
            train_ds,
            cfg.optimization.batch_size,
            cfg.data.num_dataloader_workers,
            shuffle=True,
            distributed=False,
        )
        self._val_dataloader = create_dataloader(
            val_ds,
            cfg.optimization.batch_size,
            cfg.data.num_dataloader_workers,
            shuffle=False,
            distributed=False,
        )
        self._dataloader = self._accelerator.prepare(self._dataloader)
        self._val_dataloader = self._accelerator.prepare(self._val_dataloader)

    def _init_layer_distillation(self) -> None:
        """Set up forward hooks for layer-wise distillation.

        Hooks are registered on the *unwrapped* inner modules so that
        user-facing module paths (e.g. ``"blocks.9"``) resolve correctly
        even when the model is wrapped by FSDP.  The hooks still fire
        during FSDP forward because FSDP delegates to the inner module.
        """
        cfg = self._config.distillation
        modules = cfg.layer_distillation_modules
        assert modules  # caller checks non-null/non-empty

        self._layer_pairs = resolve_layer_pairs(modules)
        teacher_paths = [t for t, _s in self._layer_pairs]
        student_paths = [s for _t, s in self._layer_pairs]

        # Unwrap FSDP/DDP so we hook onto the original module hierarchy
        unwrapped_student = self._accelerator.unwrap_model(self._student)
        unwrapped_teacher = self._accelerator.unwrap_model(self._teacher)

        # Collect per-module output transforms from the adapter (if available)
        output_transforms: dict[str, Callable[..., Any]] = {}
        if hasattr(self._adapter, "get_output_transforms"):
            output_transforms = self._adapter.get_output_transforms(unwrapped_student)

        self._teacher_extractor = FeatureExtractor(
            unwrapped_teacher,
            teacher_paths,
            output_transforms=output_transforms,
        )
        self._student_extractor = FeatureExtractor(
            unwrapped_student,
            student_paths,
            output_transforms=output_transforms,
        )

        logger.info(
            "Layer-wise distillation enabled: %d layer pairs, weight=%.2f, loss=%s, normalize=%s",
            len(self._layer_pairs),
            cfg.layer_distillation_weight,
            cfg.layer_distillation_loss_type,
            cfg.layer_distillation_normalize,
        )

    def _init_wandb(self) -> None:
        import wandb

        cfg = self._config.wandb
        self._wandb_run = wandb.init(
            project=cfg.project,
            entity=cfg.entity,
            tags=cfg.tags,
            config=self._config.model_dump(),
        )

    def _sample_timesteps(
        self, batch_size: int, device: torch.device, seq_length: int | None = None
    ) -> Tensor:
        """Sample noise levels t ∈ (0, 1) for flow matching.

        Strategies:
          - uniform: t ~ U(0,1)
          - logit_normal: t = sigmoid(N(mu, sigma))  [SD3-style]
          - shifted_logit_normal: logit_normal + post-sigmoid shift
              t_shifted = (shift * t) / (1 + (shift - 1) * t)
              Wan uses shift=8; shift=1 reduces to logit_normal.
          - resolution_dependent: logit_normal where mu is computed from
              seq_length via linear interpolation (LTX-2 style).
        """
        fm = self._config.flow_matching
        mode = fm.timestep_sampling_mode

        if mode == "uniform":
            return torch.rand(batch_size, device=device)

        if mode == "logit_normal":
            normal = torch.randn(batch_size, device=device) * fm.sigma + fm.mu
            return torch.sigmoid(normal)

        if mode == "shifted_logit_normal":
            normal = torch.randn(batch_size, device=device) * fm.sigma + fm.mu
            sigmas = torch.sigmoid(normal)
            s = fm.shift
            return (sigmas * s) / (1 + (s - 1) * sigmas)

        if mode == "resolution_dependent":
            assert seq_length is not None, (
                "resolution_dependent mode requires seq_length "
                "(adapter must implement get_seq_length)"
            )
            # LTX-2 recipe: linear interpolation of mu from seq_length
            # seq_len 1024 -> mu 0.95, seq_len 4096 -> mu 2.05
            mu = 0.95 + (2.05 - 0.95) / (4096 - 1024) * (seq_length - 1024)
            normal = torch.randn(batch_size, device=device) * fm.sigma + mu
            return torch.sigmoid(normal)

        raise ValueError(f"Unknown timestep sampling mode: {mode}")

    def _training_step(self, batch: dict[str, Tensor]) -> Tensor:
        alpha = self._config.distillation.distillation_alpha
        device = self._accelerator.device

        # Safety: ensure float tensors match model weight dtype (real data may
        # be saved in float32 even though mock data is already generated in bf16).
        wdtype = self._weight_dtype
        for k, v in batch.items():
            if v.is_floating_point() and v.dtype != wdtype:
                batch[k] = v.to(dtype=wdtype)

        latents = batch["latents"]
        batch_size = latents.shape[0]

        # Sample noise and timesteps (in weight dtype to avoid float32 promotion)
        noise = torch.randn_like(latents)
        seq_length = get_seq_length(latents, getattr(self._adapter, "patch_size", 1))
        timesteps = self._sample_timesteps(batch_size, device, seq_length).to(dtype=wdtype)

        # Adapter: prepare model-specific inputs
        inputs = self._adapter.prepare_inputs(
            batch, noise, timesteps, pipeline=self._inference_pipeline
        )

        # Student forward (model-specific output, opaque to trainer)
        student_output = self._adapter.forward_model(self._student, inputs)

        # Task loss (adapter handles all modalities: video, audio, etc.)
        if alpha > 0:
            task_loss = self._adapter.compute_task_loss(student_output, inputs)
        else:
            task_loss = torch.tensor(0.0, device=device)

        # Distillation loss
        if alpha < 1.0:
            with torch.no_grad():
                teacher_output = self._adapter.forward_model(self._teacher, inputs)

            # Output-level distillation (adapter handles all modalities)
            output_distill_loss = self._adapter.compute_distillation_loss(
                student_output, teacher_output, inputs
            )

            # Layer-wise distillation loss
            if self._layer_pairs:
                layer_loss = self._compute_layer_distillation_loss()
                gamma = self._config.distillation.layer_distillation_weight
                distill_loss = (1.0 - gamma) * output_distill_loss + gamma * layer_loss
            else:
                layer_loss = torch.tensor(0.0, device=device)
                distill_loss = output_distill_loss
        else:
            output_distill_loss = torch.tensor(0.0, device=device)
            layer_loss = torch.tensor(0.0, device=device)
            distill_loss = torch.tensor(0.0, device=device)

        total_loss = alpha * task_loss + (1.0 - alpha) * distill_loss

        # Log individual losses
        if self._accelerator.is_main_process and self._wandb_run is not None:
            log_dict = {
                "loss/task": task_loss.item(),
                "loss/distillation_output": output_distill_loss.item(),
                "loss/distillation_total": distill_loss.item(),
                "loss/total": total_loss.item(),
            }
            if self._layer_pairs:
                log_dict["loss/distillation_layer"] = layer_loss.item()
            self._wandb_run.log(log_dict, step=self._global_step)

        return total_loss

    def _compute_layer_distillation_loss(self) -> Tensor:
        """Compute distillation loss across hooked intermediate layers."""
        assert self._student_extractor is not None
        assert self._teacher_extractor is not None

        s_feats = self._student_extractor.get_features()
        t_feats = self._teacher_extractor.get_features()

        cfg = self._config.distillation
        normalize = cfg.layer_distillation_normalize
        loss_type = cfg.layer_distillation_loss_type

        losses = []
        for t_path, s_path in self._layer_pairs:
            s = s_feats[s_path]
            t = t_feats[t_path]

            if normalize:
                s = torch.nn.functional.normalize(s.float(), dim=-1)
                t = torch.nn.functional.normalize(t.float(), dim=-1)

            if loss_type == "mse":
                losses.append(torch.nn.functional.mse_loss(s, t))
            elif loss_type == "cosine":
                cos_sim = torch.nn.functional.cosine_similarity(
                    s.flatten(start_dim=1), t.flatten(start_dim=1), dim=-1
                )
                losses.append(1.0 - cos_sim.mean())
            else:
                raise ValueError(f"Unknown layer distillation loss type: {loss_type}")

        self._student_extractor.clear()
        self._teacher_extractor.clear()

        return torch.stack(losses).mean()

    @torch.no_grad()
    def _run_validation(self) -> None:
        """Run validation loss + optional video generation."""
        # Free memory for validation
        self._optimizer.zero_grad(set_to_none=True)
        free_gpu_memory()

        self._run_validation_loss()

        if self._inference_pipeline and self._cached_val_embeds:
            self._run_validation_videos()

        free_gpu_memory()

    def _run_validation_loss(self) -> None:
        """Compute validation loss over the held-out set."""
        val_losses = {"task": 0.0, "distill": 0.0, "total": 0.0}
        n_batches = 0
        alpha = self._config.distillation.distillation_alpha
        device = self._accelerator.device
        wdtype = self._weight_dtype

        for batch in self._val_dataloader:
            # Match training dtype handling
            for k, v in batch.items():
                if v.is_floating_point() and v.dtype != wdtype:
                    batch[k] = v.to(dtype=wdtype)

            latents = batch["latents"]
            batch_size = latents.shape[0]
            noise = torch.randn_like(latents)
            seq_length = get_seq_length(latents, getattr(self._adapter, "patch_size", 1))
            timesteps = self._sample_timesteps(batch_size, device, seq_length).to(dtype=wdtype)

            inputs = self._adapter.prepare_inputs(
                batch, noise, timesteps, pipeline=self._inference_pipeline
            )
            student_output = self._adapter.forward_model(self._student, inputs)

            if alpha > 0:
                tl = self._adapter.compute_task_loss(student_output, inputs)
            else:
                tl = torch.tensor(0.0, device=device)

            if alpha < 1.0:
                teacher_output = self._adapter.forward_model(self._teacher, inputs)
                dl = self._adapter.compute_distillation_loss(student_output, teacher_output, inputs)
            else:
                dl = torch.tensor(0.0, device=device)

            total = alpha * tl + (1.0 - alpha) * dl
            val_losses["task"] += tl.item()
            val_losses["distill"] += dl.item()
            val_losses["total"] += total.item()
            n_batches += 1

        if n_batches > 0:
            for k in val_losses:
                val_losses[k] /= n_batches

        if self._accelerator.is_main_process:
            logger.info(
                f"Validation loss (step {self._global_step}, {n_batches} batches): "
                f"task={val_losses['task']:.6e} distill={val_losses['distill']:.6e} "
                f"total={val_losses['total']:.6e}"
            )
            if self._wandb_run is not None:
                self._wandb_run.log(
                    {f"val/{k}": v for k, v in val_losses.items()},
                    step=self._global_step,
                )

    def _run_validation_videos(self) -> None:
        """Generate validation videos using the inference pipeline."""
        assert self._inference_pipeline is not None

        cfg = self._config.validation
        gen_config = {
            "width": cfg.video_dims[0],
            "height": cfg.video_dims[1],
            "num_frames": cfg.video_dims[2],
            "num_inference_steps": cfg.inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "seed": cfg.seed,
            "frame_rate": cfg.frame_rate,
        }

        # All FSDP ranks must participate (model is sharded), but only rank 0 saves
        videos = self._inference_pipeline.generate(
            model=self._student,
            cached_embeds=self._cached_val_embeds,
            config=gen_config,
            device=str(self._accelerator.device),
        )

        if is_global_rank0() and videos:
            self._save_validation_videos(videos)

    def _save_validation_videos(self, videos: list[Tensor]) -> None:
        out_dir = Path(self._config.output_dir) / "validation" / f"step_{self._global_step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts = self._config.validation.prompts
        for i, video in enumerate(videos):
            path = out_dir / f"video_{i:03d}.mp4"
            try:
                self._write_video(video, path, self._config.validation.frame_rate)
            except Exception as e:
                logger.warning(f"Failed to save validation video {i}: {e}")
                continue

            if self._wandb_run and self._config.wandb.log_validation_videos:
                import wandb

                caption = prompts[i] if i < len(prompts) else ""
                self._wandb_run.log(
                    {f"val/video_{i}": wandb.Video(str(path), caption=caption)},
                    step=self._global_step,
                )

        logger.info(f"Saved {len(videos)} validation videos to {out_dir}")

    @staticmethod
    def _write_video(video_tensor: Tensor, path: Path, fps: float) -> None:
        """Write a [C,F,H,W] float tensor in [0,1] to mp4."""
        import torchvision.io as tvio

        # [C,F,H,W] -> [F,H,W,C], float [0,1] -> uint8
        frames = video_tensor.permute(1, 2, 3, 0).clamp(0, 1).mul(255).byte().cpu()
        tvio.write_video(str(path), frames, fps=fps)

    def _get_checkpoints_dir(self) -> Path:
        return Path(self._config.output_dir) / "checkpoints"

    def _save_training_state(self) -> Path | None:
        """Save full training state for resumption.

        Uses accelerator.save_state() which handles FSDP sharded state dict,
        optimizer, scheduler, RNG, and gradient scaler. Additionally saves
        ModelOpt quantization architecture and custom metadata.

        Returns checkpoint directory on rank 0, None otherwise.
        """
        final_dir = self._get_checkpoints_dir() / f"step_{self._global_step:06d}"
        tmp_dir = self._get_checkpoints_dir() / f"step_{self._global_step:06d}_tmp"

        logger.info(f"Saving training state at step {self._global_step} ...")

        if is_global_rank0():
            tmp_dir.mkdir(parents=True, exist_ok=True)
        self._accelerator.wait_for_everyone()

        self._accelerator.save_state(str(tmp_dir))
        self._accelerator.wait_for_everyone()

        if is_global_rank0():
            # Save ModelOpt state
            if self._config.distillation.quant_cfg is not None:
                try:
                    import modelopt.torch.opt as mto

                    modelopt_state = mto.modelopt_state(self._student)
                    torch.save(modelopt_state, tmp_dir / "modelopt_state.pt")
                except Exception as e:
                    logger.warning(f"Failed to save modelopt state: {e}")

            # Save metadata
            metadata = {
                "global_step": self._global_step,
                "config": self._config.model_dump(),
            }
            with open(tmp_dir / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

            # Atomic rename
            if tmp_dir.exists():
                if final_dir.exists():
                    import shutil

                    shutil.rmtree(final_dir)
                tmp_dir.rename(final_dir)
                logger.info(f"Training state saved to {final_dir}")

        self._accelerator.wait_for_everyone()
        self._cleanup_checkpoints()
        self._accelerator.wait_for_everyone()

        return final_dir if is_global_rank0() else None

    def _cleanup_checkpoints(self) -> None:
        """Remove old training checkpoints, keeping only the last N.

        Excludes step_XXXXXX_tmp (incomplete) and step_000000_quantized
        (calibration-only checkpoint, not training state).
        """
        if not is_global_rank0():
            return
        keep_n = self._config.checkpoints.keep_last_n
        ckpt_dir = self._get_checkpoints_dir()
        if not ckpt_dir.exists():
            return

        checkpoints = sorted(
            [
                d
                for d in ckpt_dir.iterdir()
                if d.is_dir()
                and d.name.startswith("step_")
                and not d.name.endswith("_tmp")
                and not d.name.endswith("_quantized")
            ],
            key=lambda d: d.name,
        )
        for old_ckpt in checkpoints[:-keep_n]:
            import shutil

            logger.info(f"Removing old checkpoint: {old_ckpt.name}")
            shutil.rmtree(old_ckpt)

    def _find_resume_checkpoint(self) -> Path | None:
        resume = self._config.distillation.resume_from_checkpoint
        if resume is None:
            return None

        if resume == "latest":
            ckpt_dir = self._get_checkpoints_dir()
            if not ckpt_dir.exists():
                return None
            checkpoints = sorted(
                [
                    d
                    for d in ckpt_dir.iterdir()
                    if d.is_dir()
                    and d.name.startswith("step_")
                    and not d.name.endswith("_tmp")
                    and not d.name.endswith("_quantized")
                ]
            )
            return checkpoints[-1] if checkpoints else None

        path = Path(resume)
        return path if path.exists() else None

    def _load_training_state(self) -> None:
        """Resume from a previously saved training state.

        The quantization architecture must already be restored BEFORE this
        method is called. This is handled by _apply_quantization() (Path A).
        """
        checkpoint_dir = self._find_resume_checkpoint()
        if checkpoint_dir is None:
            logger.warning("No checkpoint found to resume from, starting from scratch")
            return

        logger.info(f"Resuming training from {checkpoint_dir} ...")
        self._accelerator.load_state(str(checkpoint_dir))

        metadata_path = checkpoint_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            self._global_step = metadata.get("global_step", 0)
            logger.info(f"Resumed at global_step={self._global_step}")

    def _save_final_model(self) -> Path | None:
        """Save inference-ready model weights as safetensors."""
        self._accelerator.wait_for_everyone()
        full_state_dict = self._accelerator.get_state_dict(self._student)

        if not is_global_rank0():
            return None

        from safetensors.torch import save_file

        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_path = out_dir / "model_weights_final.safetensors"

        save_dtype = to_dtype(self._config.checkpoints.precision)
        state_dict = {k: v.to(save_dtype) for k, v in full_state_dict.items()}
        save_file(state_dict, str(save_path))
        logger.info(f"Saved inference-ready weights to {save_path}")
        return save_path

    def save_quantized_model(self, path: str | Path | None = None) -> None:
        """Save ModelOpt quantized model (global rank 0 only)."""
        if not is_global_rank0():
            return

        import modelopt.torch.opt as mto

        if path is None:
            path = self._config.distillation.save_quantized_checkpoint
        if path is None:
            path = Path(self._config.output_dir) / "quantized_model"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving quantized model to {path}")
        mto.save(self._student, str(path))
        logger.info("Quantized model saved")

    def train(self) -> dict:
        cfg = self._config
        total_steps = cfg.optimization.steps
        grad_accum = cfg.optimization.gradient_accumulation_steps

        train_start = time.time()

        # Slurm time-limit
        must_save_by = cfg.checkpoints.must_save_by
        save_deadline = (train_start + must_save_by * 60) if must_save_by else None
        if save_deadline and is_global_rank0():
            logger.info(
                f"Time-limit save enabled: will save and exit after {must_save_by:.0f} minutes"
            )

        # Initial validation
        if cfg.validation.interval and not cfg.validation.skip_initial_validation:
            self._run_validation()

        self._accelerator.wait_for_everyone()

        # Create data iterator
        data_iter = iter(self._dataloader)

        if is_global_rank0():
            logger.info(
                f"Starting training: steps={total_steps}, grad_accum={grad_accum}, "
                f"batch_size={cfg.optimization.batch_size}"
            )

        start_micro = self._global_step * grad_accum
        total_micro = total_steps * grad_accum
        pbar = tqdm(
            range(start_micro, total_micro),
            initial=start_micro,
            total=total_micro,
            desc="Training",
            disable=not is_global_rank0(),
        )

        for micro_step in pbar:
            # Get next batch, cycling the dataloader
            try:
                batch = next(data_iter)
            except StopIteration:
                self._data_epoch += 1
                sampler = getattr(self._dataloader, "sampler", None)
                if sampler is not None and hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(self._data_epoch)
                data_iter = iter(self._dataloader)
                batch = next(data_iter)

            with self._accelerator.accumulate(self._student):
                loss = self._training_step(batch)
                self._accelerator.backward(loss)

                if self._accelerator.sync_gradients:
                    self._accelerator.clip_grad_norm_(
                        self._student.parameters(), cfg.optimization.max_grad_norm
                    )

                self._optimizer.step()
                self._lr_scheduler.step()
                self._optimizer.zero_grad()

            is_opt_step = self._accelerator.sync_gradients
            if is_opt_step:
                self._global_step += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", lr=f"{self._lr_scheduler.get_last_lr()[0]:.2e}"
                )

                if self._wandb_run:
                    self._wandb_run.log(
                        {"lr": self._lr_scheduler.get_last_lr()[0]},
                        step=self._global_step,
                    )

                # Validation
                if cfg.validation.interval and self._global_step % cfg.validation.interval == 0:
                    self._run_validation()

                # Periodic checkpoint
                saved_this_step = False
                if cfg.checkpoints.interval and self._global_step % cfg.checkpoints.interval == 0:
                    logger.info(f"Saving periodic checkpoint at step {self._global_step} ...")
                    self._save_training_state()
                    saved_this_step = True

                # Time-limit exit
                if save_deadline and time.time() >= save_deadline:
                    elapsed = (time.time() - train_start) / 60
                    logger.info(
                        f"Time limit reached ({elapsed:.1f} min >= {must_save_by:.0f} min). "
                        f"Saving at step {self._global_step} and exiting."
                    )
                    if not saved_this_step:
                        self._save_training_state()
                    break

                self._accelerator.wait_for_everyone()

                if self._global_step >= total_steps:
                    break

        # --- End of training ---
        elapsed = time.time() - train_start
        stats = {
            "final_step": self._global_step,
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
        }

        self._save_training_state()
        saved_path = self._save_final_model()

        if cfg.distillation.quant_cfg is not None:
            self.save_quantized_model()

        if is_global_rank0():
            logger.info(
                f"Training complete at step {self._global_step}. Elapsed: {elapsed / 60:.1f} min"
            )
            if saved_path:
                logger.info(f"Model saved to: {saved_path}")

        if self._wandb_run:
            self._wandb_run.finish()

        return stats
