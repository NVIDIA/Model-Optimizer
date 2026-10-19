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

"""Pydantic configuration models for the unified distillation trainer."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    model_name: str = Field(
        description=(
            "Model backend name (e.g. 'wan', 'ltx2'). "
            "Selects the loader/adapter/pipeline from models/ registry."
        )
    )
    model_variant: str | None = Field(
        default=None,
        description="Variant within a model family, e.g. 'ti2v-5B', 't2v-A14B'. Backend-specific.",
    )
    model_path: str = Field(description="Path to model checkpoint directory or file.")
    text_encoder_path: str | None = Field(
        default=None,
        description="Path to text encoder model (e.g. Gemma directory for LTX-2). Required by some model backends.",
    )
    dtype: str = Field(
        default="bfloat16", description="Model dtype: bfloat16, float16, or float32."
    )


class DistillationConfig(BaseModel):
    teacher_model_path: str | None = Field(
        default=None,
        description="Path to teacher checkpoint. None = same as model.model_path.",
    )
    teacher_dtype: str = Field(default="bfloat16")
    distillation_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Loss mixing: L = alpha * L_task + (1-alpha) * L_distill. "
            "0 = pure distillation, 1 = pure task loss."
        ),
    )
    distillation_loss_type: str = Field(default="mse", description="'mse' or 'cosine'.")

    # ModelOpt quantization / sparsity
    quant_cfg: str | None = Field(
        default=None,
        description="ModelOpt quantization config name, e.g. 'FP8_DEFAULT_CFG'.",
    )
    calibration_size: int = Field(default=128, ge=0)
    calibration_n_steps: int = Field(default=30, ge=1)
    calibration_guidance_scale: float = Field(default=4.0)
    calibration_prompts_file: str | None = Field(
        default=None, description="Text file with one prompt per line for calibration."
    )
    restore_quantized_checkpoint: str | None = Field(
        default=None,
        description="Path to restore a previously quantized model via mto.save().",
    )
    save_quantized_checkpoint: str | None = Field(
        default=None, description="Path to save the final quantized model."
    )

    # Layer-wise distillation
    layer_distillation_modules: list[str | list[str]] | None = Field(
        default=None,
        description=(
            "Module paths for layer-wise distillation. "
            "Strings = same path for teacher & student (self-distillation). "
            "[teacher_path, student_path] pairs for cross-architecture. "
            "None = layer distillation disabled."
        ),
    )
    layer_distillation_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Mixing ratio (gamma) between output and layer distillation: "
            "L_distill = (1-gamma)*L_output + gamma*L_layer. "
            "0 = output distillation only (layer hooks still run but don't "
            "affect the loss), 1 = layer distillation only."
        ),
    )
    layer_distillation_loss_type: str = Field(
        default="mse",
        description="Loss function for layer distillation: 'mse' or 'cosine'.",
    )
    layer_distillation_normalize: bool = Field(
        default=True,
        description=(
            "L2-normalize student and teacher features before computing "
            "layer loss. Makes the loss scale-invariant (compares direction "
            "only, ignores magnitude). Recommended when teacher and student "
            "have different activation scales (e.g. quantized student)."
        ),
    )

    @field_validator("layer_distillation_modules", mode="before")
    @classmethod
    def _validate_layer_modules(cls, v):
        if v is None:
            return v
        result: list[str | list[str]] = []
        for entry in v:
            if isinstance(entry, str):
                result.append(entry)
            elif isinstance(entry, (list, tuple)):
                if len(entry) != 2 or not all(isinstance(s, str) for s in entry):
                    raise ValueError(
                        f"Pairwise layer spec must be [teacher_path, student_path], got {entry}"
                    )
                result.append(list(entry))
            else:
                raise ValueError(f"Expected str or [str, str], got {type(entry)}")
        return result

    # Resume
    resume_from_checkpoint: str | None = Field(
        default=None, description="Checkpoint path or 'latest' for auto-resume."
    )

    # Audio-video joint training (LTX-2)
    with_audio: bool = Field(
        default=False,
        description="Enable joint audio-video training. Requires audio latents in dataset.",
    )

    # Mock data for testing
    use_mock_data: bool = False
    mock_data_samples: int = 100


class OptimizationConfig(BaseModel):
    learning_rate: float = 2e-6
    steps: int = 10000
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    optimizer_type: str = Field(default="adamw", description="'adamw' or 'adamw8bit'.")
    scheduler_type: str = Field(
        default="cosine", description="LR scheduler: 'cosine', 'linear', 'constant'."
    )
    warmup_steps: int = 0


class DataConfig(BaseModel):
    preprocessed_data_root: str | None = Field(
        default=None, description="Path to precomputed latents + text embeddings."
    )
    num_dataloader_workers: int = 2


class ValidationConfig(BaseModel):
    prompts: list[str] = Field(default_factory=list)
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    video_dims: list[int] = Field(default=[512, 320, 33], description="[width, height, num_frames]")
    frame_rate: float = 25.0
    inference_steps: int = 30
    guidance_scale: float = 4.0
    seed: int = 42
    interval: int = Field(default=500, description="Run validation every N optimization steps.")
    skip_initial_validation: bool = False


class FlowMatchingConfig(BaseModel):
    timestep_sampling_mode: str = Field(
        default="logit_normal",
        description=(
            "Timestep sampling strategy: "
            "'uniform' - U(0,1); "
            "'logit_normal' - sigmoid(N(mu, sigma)); "
            "'shifted_logit_normal' - post-sigmoid shift: "
            "  t = (s*t)/(1+(s-1)*t), used by Wan (shift=8); "
            "'resolution_dependent' - logit_normal with mu auto-computed "
            "  from patchified seq_length (LTX-2 style)."
        ),
    )
    mu: float = Field(
        default=0.0, description="Mean of the normal for logit_normal / shifted modes."
    )
    sigma: float = Field(
        default=1.0, description="Std of the normal for logit_normal / shifted modes."
    )
    shift: float = Field(
        default=1.0,
        description=(
            "Shift factor for shifted_logit_normal. "
            "Wan uses shift=8. shift=1 is equivalent to logit_normal."
        ),
    )


class CheckpointConfig(BaseModel):
    interval: int = Field(
        default=1000, description="Save training state every N optimization steps."
    )
    keep_last_n: int = Field(default=3, ge=1)
    precision: str = "bfloat16"
    must_save_by: float | None = Field(
        default=None,
        gt=0,
        description="Minutes after which to save and exit (for Slurm time limits).",
    )


class WandbConfig(BaseModel):
    enabled: bool = True
    project: str = "distillation"
    entity: str | None = None
    tags: list[str] = Field(default_factory=lambda: ["distillation", "modelopt"])
    log_validation_videos: bool = True


class TrainerConfig(BaseModel):
    """Top-level config passed to DistillationTrainer."""

    model: ModelConfig
    distillation: DistillationConfig = Field(default_factory=DistillationConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    flow_matching: FlowMatchingConfig = Field(default_factory=FlowMatchingConfig)
    checkpoints: CheckpointConfig = Field(default_factory=CheckpointConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)

    seed: int = 42
    output_dir: str = "./outputs"

    class Config:
        extra = "forbid"
