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

"""Pydantic schema models for recipe YAML validation.

These models define the structure of recipe YAML files. The resolver
(resolver.py) translates validated schema objects into the config dicts
that ModelOpt APIs accept (mtq.quantize, distill.convert, etc.).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, model_validator


class CalibrationConfig(BaseModel):
    """Calibration data configuration."""

    dataset: str | list[str] = "cnn_dailymail"
    num_samples: int | list[int] = 512
    max_sequence_length: int = 512
    batch_size: int = 1


class KVCacheConfig(BaseModel):
    """KV cache quantization configuration."""

    format: str  # fp8, nvfp4


class AlgorithmConfig(BaseModel):
    """Quantization algorithm configuration.

    Extra fields (e.g., fp8_scale_sweep) are passed through to the algorithm dict.
    """

    method: str  # max, smoothquant, awq_lite, awq_clip, awq_full, mse, etc.
    alpha_step: float | None = None
    max_co_batch_size: int | None = None

    model_config = {"extra": "allow"}


class QuantizerSpec(BaseModel):
    """Specifies quantization for weights or activations."""

    format: str | None = None  # human-readable: fp8, nvfp4, int4, int8
    num_bits: int | list[int] | None = None  # expert-mode escape hatch
    axis: int | None = None
    block_sizes: dict[str, Any] | None = None
    enable: bool = True
    stages: list[QuantizerSpec] | None = None  # for multi-stage (W4A8)


class OverrideEntry(BaseModel):
    """Per-layer or per-module-class override."""

    pattern: str | None = None  # glob: "*lm_head*"
    module_class: str | None = None  # class: "nn.LayerNorm"
    enable: bool | None = None
    format: str | None = None
    scale_type: Literal["static", "dynamic"] | None = None  # shorthand for block_sizes.type
    weights: QuantizerSpec | None = None
    activations: QuantizerSpec | None = None
    num_bits: int | list[int] | None = None
    axis: int | None = None

    @model_validator(mode="after")
    def validate_has_selector(self):
        """Ensure at least one of pattern or module_class is set."""
        if not self.pattern and not self.module_class:
            raise ValueError("Override must specify 'pattern' or 'module_class' to target.")
        return self


class QuantizationSection(BaseModel):
    """Quantization technique configuration.

    Covers both PTQ and QAT. For QAT, set mode="qat" and provide training config.
    PTQ: calibrate → quantize (minutes). QAT: quantize → fine-tune (hours, better accuracy).
    """

    mode: Literal["ptq", "qat"] = "ptq"
    preset: str | None = None
    weights: QuantizerSpec | None = None
    activations: QuantizerSpec | None = None
    algorithm: AlgorithmConfig | str | None = None
    kv_cache: KVCacheConfig | None = None
    calibration: CalibrationConfig | None = None
    training: TrainingConfig | None = None  # QAT training config
    overrides: list[OverrideEntry] = []
    disabled_patterns: list[str] = []

    @model_validator(mode="after")
    def validate_preset_or_custom(self):
        """Ensure preset and explicit quantizer specs are not both set."""
        if self.preset and (self.weights or self.activations):
            raise ValueError(
                "Cannot specify both 'preset' and 'weights'/'activations'. "
                "Use preset with overrides, or specify weights/activations from scratch."
            )
        if self.mode == "qat" and self.training is None:
            raise ValueError("QAT mode requires a 'training' configuration.")
        return self


class AutoQuantizeFormatEntry(BaseModel):
    """A candidate format for auto-quantize search."""

    preset: str  # e.g., "nvfp4_awq", "fp8"


class AutoQuantizeSection(BaseModel):
    """Auto-quantize configuration for per-layer format search."""

    effective_bits: float
    formats: list[AutoQuantizeFormatEntry]
    method: str = "gradient"
    num_calib_steps: int = 512
    num_score_steps: int = 128
    disabled_patterns: list[str] = []
    kv_cache: KVCacheConfig | None = None
    calibration: CalibrationConfig | None = None


class TrainingConfig(BaseModel):
    """Training configuration for QAT and distillation."""

    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 1
    max_steps: int | None = None
    warmup_steps: int = 0
    weight_decay: float = 0.0
    gradient_accumulation_steps: int = 1

    model_config = {"extra": "allow"}


class DistillationSection(BaseModel):
    """Knowledge distillation configuration.

    Maps to modelopt.torch.distill.convert() with KDLossConfig.
    """

    teacher: str  # teacher model path (e.g., "meta-llama/Llama-3-70B")
    criterion: str = "kl_div"  # kl_div, mse, cross_entropy
    kd_loss_weight: float = 0.5  # weight for KD loss vs student loss
    layer_pairs: list[dict[str, str]] | None = None  # layer-wise distillation
    training: TrainingConfig | None = None
    calibration: CalibrationConfig | None = None

    model_config = {"extra": "allow"}


class SparsitySection(BaseModel):
    """Sparsity configuration.

    Maps to modelopt.torch.sparsity APIs.
    """

    method: str  # sparse_gpt, magnitude, wanda
    sparsity: float = 0.5  # target sparsity ratio
    pattern: str = "unstructured"  # unstructured, 2:4
    calibration: CalibrationConfig | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_sparsity_range(self):
        """Ensure sparsity is in valid range (0, 1]."""
        if not (0.0 < self.sparsity <= 1.0):
            raise ValueError(f"Sparsity must be in (0.0, 1.0], got {self.sparsity}")
        return self


class ExportConfig(BaseModel):
    """Export configuration."""

    format: Literal["hf", "tensorrt_llm"] = "hf"
    output_dir: str = "./output"
    tensor_parallel: int = 1
    pipeline_parallel: int = 1


class ModelConfig(BaseModel):
    """Model specification."""

    path: str
    trust_remote_code: bool = False
    attn_implementation: str | None = None


class RecipeMetadata(BaseModel):
    """Optional recipe metadata."""

    name: str | None = None
    description: str | None = None
    author: str | None = None
    tags: list[str] = []


class PruningSection(BaseModel):
    """Pruning configuration.

    Maps to modelopt.torch.prune.prune(model, mode, constraints, dummy_input, config).
    Modes: fastnas, gradnas, mcore_minitron.
    """

    mode: str  # fastnas, gradnas, mcore_minitron
    constraints: dict[str, Any] = {}  # flops, params, export_config
    calibration: CalibrationConfig | None = None
    training: TrainingConfig | None = None

    model_config = {"extra": "allow"}


class RecipeConfig(BaseModel):
    """Top-level recipe schema.

    Techniques are composable — a recipe can combine quantization + distillation,
    sparsity + quantization, etc. Each technique owns its own calibration/training config.
    Execution order: pruning → sparsity → quantization → distillation.
    """

    version: str = "1.0"
    metadata: RecipeMetadata | None = None
    model: ModelConfig | None = None
    pruning: PruningSection | None = None
    quantization: QuantizationSection | None = None
    auto_quantize: AutoQuantizeSection | None = None
    distillation: DistillationSection | None = None
    sparsity: SparsitySection | None = None
    export: ExportConfig | None = None

    @model_validator(mode="after")
    def validate_exclusive_sections(self):
        """Ensure quantization and auto_quantize are mutually exclusive."""
        if self.quantization and self.auto_quantize:
            raise ValueError(
                "'quantization' and 'auto_quantize' are mutually exclusive. Use one or the other."
            )
        return self
