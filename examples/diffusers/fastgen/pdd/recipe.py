# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ModelOpt-owned construction of a Qwen-Image PDD student and frozen teacher."""

from __future__ import annotations

import copy
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from modelopt.torch.fastgen import PDDConfig, PDDMetadata, PDDOutputProjection, PDDPipeline
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QwenImagePDDAdapter,
    convert_qwen_image_to_pdd,
)

from .checkpoint import PDDCheckpointManager, build_pdd_checkpoint_identity
from .data import (
    _build_training_dataloader,
    _build_validation_dataloader,
    _build_validation_plan,
    _collective_training_batch,
    _collective_training_iterator,
    _coverage_axis,
    _iter_validation_batches,
    _validate_dataset_contract,
)
from .verify_readonly_automodel import snapshot_installed_distribution


@dataclass(frozen=True)
class PDDParallelConfig:
    """Pure-data-parallel FSDP2 settings for the first Qwen-Image example."""

    dp_size: int | None = None
    activation_checkpointing: bool = False


@dataclass(frozen=True)
class PDDCheckpointConfig:
    """AutoModel Checkpointer settings needed by the PDD lifecycle."""

    checkpoint_dir: str = "checkpoints/pdd_qwen_image"
    enabled: bool = True
    model_save_format: str = "torch_save"
    restore_from: str | None = None
    save_consolidated: bool = False


@dataclass(frozen=True)
class PDDStepSchedulerConfig:
    """AutoModel-compatible batch, cadence, and termination settings."""

    max_steps: int = 10_000
    num_epochs: int = 200
    log_every: int = 10
    ckpt_every_steps: int = 1_000
    local_batch_size: int = 1
    global_batch_size: int | None = None
    save_checkpoint_every_epoch: bool = False


@dataclass(frozen=True)
class PDDTrainingHealthConfig:
    """PDD-only update health settings."""

    max_grad_norm: float = 1.0
    zero_grad_warmup_steps: int = 0


@dataclass(frozen=True)
class PDDValidationConfig:
    """Deterministic held-out validation settings."""

    count: int = 2_000
    seed: int = 2026
    split_seed: int = 2026
    every_steps: int = 1_000


@dataclass(frozen=True)
class PDDGuidanceConfig:
    """Resolved Qwen packed-CFG norm-rescaling settings."""

    rescale: float = 1.0
    eps: float = 1e-5


@dataclass(frozen=True)
class PDDRecipeConfig:
    """Resolved setup inputs; incompatible mutation modes have already been rejected."""

    model_id: str
    model_revision: str | None
    pdd: PDDConfig
    parallel: PDDParallelConfig
    checkpoint: PDDCheckpointConfig
    step_scheduler: PDDStepSchedulerConfig
    training_health: PDDTrainingHealthConfig
    validation: PDDValidationConfig
    guidance: PDDGuidanceConfig
    seed: int
    learning_rate: float
    weight_decay: float
    adam_betas: tuple[float, float]
    adam_eps: float
    device: torch.device
    dtype: torch.dtype
    fuse_qkv_projections: bool


@dataclass(frozen=True)
class PDDSetupArtifacts:
    """Objects produced in the required load-to-checkpoint construction order."""

    pipe: Any
    student: nn.Module
    teacher: nn.Module
    projection: PDDOutputProjection
    optimizer: torch.optim.Optimizer
    distributed_setup: Any
    fsdp_manager: Any
    checkpointer: Any
    metadata: PDDMetadata
    checkpoint_keys: tuple[str, ...]
    lifecycle: tuple[str, ...]
    automodel_snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class PDDTrainingArtifacts:
    """Direct-update objects layered on the already-constructed Task-7 setup."""

    pipeline: PDDPipeline
    trainer: Any
    scheduler: torch.optim.lr_scheduler.LRScheduler
    rng: Any


@dataclass(frozen=True)
class PDDExportSetupArtifacts:
    """Student-only FSDP2 objects needed for collective DCP export."""

    pipe: Any
    student: nn.Module
    projection: PDDOutputProjection
    distributed_setup: Any
    fsdp_manager: Any
    checkpointer: Any
    metadata: PDDMetadata
    checkpoint_keys: tuple[str, ...]
    transformer_config: Mapping[str, Any]
    lifecycle: tuple[str, ...]
    automodel_snapshot: Mapping[str, Any]


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _config_to_mapping(value: Any) -> Mapping[str, Any]:
    """Materialize AutoModel config targets as their original YAML dotted paths."""
    if isinstance(value, Mapping):
        return value
    to_yaml_dict = getattr(value, "to_yaml_dict", None)
    if callable(to_yaml_dict):
        return _as_mapping(
            to_yaml_dict(resolve_env=True, use_orig_values=True),
            name="ConfigNode.to_yaml_dict() result",
        )
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _as_mapping(to_dict(), name="config.to_dict() result")
    raise TypeError(f"config must be a mapping or ConfigNode, got {type(value).__name__}.")


def _reject_enabled(value: Any, *, name: str) -> None:
    if value is None or value is False or value == {}:
        return
    raise ValueError(f"PDD does not support {name}; disable it before model loading.")


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool.")
    return value


def _require_int_at_least(value: Any, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return value


def _require_finite_real(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a real number.")
    resolved = float(value)
    if not math.isfinite(resolved) or (minimum is not None and resolved < minimum):
        qualifier = "finite" if minimum is None else f"finite and >= {minimum}"
        raise ValueError(f"{name} must be {qualifier}.")
    return resolved


def _resolve_dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if not isinstance(value, str):
        raise TypeError("model.torch_dtype must be a torch dtype name.")
    dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return dtypes[value]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model.torch_dtype={value!r}; expected {sorted(dtypes)}."
        ) from error


def resolve_pdd_recipe_config(raw: Any) -> PDDRecipeConfig:
    """Resolve one canonical DMD2-shaped PDD configuration."""
    raw = _config_to_mapping(raw)

    legacy_training = _as_mapping(raw.get("training", {}), name="training")
    legacy_replacements = {
        "seed": "seed",
        "max_steps": "step_scheduler.max_steps",
        "log_every_steps": "step_scheduler.log_every",
        "checkpoint_every_steps": "step_scheduler.ckpt_every_steps",
        "validation_every_steps": "validation.every_steps",
        "local_batch_size": "step_scheduler.local_batch_size",
        "global_batch_size": "step_scheduler.global_batch_size",
        "grad_accumulation_steps": "step_scheduler.global_batch_size",
        "validation_seed": "validation.seed",
        "max_grad_norm": "training_health.max_grad_norm",
        "zero_grad_warmup_steps": "training_health.zero_grad_warmup_steps",
    }
    if legacy_training:
        key = next(iter(legacy_training))
        replacement_key = legacy_replacements.get(key, "the canonical PDD schema")
        raise ValueError(f"training.{key} is unsupported; use {replacement_key}.")

    model = _as_mapping(raw.get("model"), name="model")
    pdd_raw = _as_mapping(raw.get("pdd"), name="pdd")
    fsdp = _as_mapping(raw.get("fsdp", {}), name="fsdp")
    optim = _as_mapping(raw.get("optim", {}), name="optim")
    optimizer_cfg = _as_mapping(optim.get("optimizer", {}), name="optim.optimizer")
    lr_scheduler = _as_mapping(raw.get("lr_scheduler", {}), name="lr_scheduler")
    step_scheduler = _as_mapping(raw.get("step_scheduler", {}), name="step_scheduler")
    training_health = _as_mapping(raw.get("training_health", {}), name="training_health")
    validation = _as_mapping(raw.get("validation", {}), name="validation")
    checkpoint = _as_mapping(raw.get("checkpoint", {}), name="checkpoint")
    guidance = _as_mapping(raw.get("guidance", {}), name="guidance")
    data = _as_mapping(raw.get("data", {}), name="data")
    dataloader = _as_mapping(data.get("dataloader", {}), name="data.dataloader")

    for legacy_key in ("weight_decay", "betas", "eps"):
        if legacy_key in optim:
            raise ValueError(
                f"optim.{legacy_key} is unsupported; use optim.optimizer.{legacy_key}."
            )

    for legacy_key, replacement_key in (
        ("validation_count", "validation.count"),
        ("split_seed", "validation.split_seed"),
    ):
        if legacy_key in data:
            raise ValueError(f"data.{legacy_key} is unsupported; use {replacement_key}.")

    target = dataloader.get("_target_")
    expected_target = "fastgen_data.build_text_to_image_multiresolution_dataloader"
    if target is not None and target != expected_target:
        raise ValueError(f"data.dataloader._target_ must be {expected_target!r}.")
    if _require_bool(dataloader.get("drop_last", True), name="data.dataloader.drop_last") is False:
        raise ValueError("PDD exact sample accounting requires data.dataloader.drop_last=true.")
    if _require_bool(
        dataloader.get("dynamic_batch_size", False),
        name="data.dataloader.dynamic_batch_size",
    ):
        raise ValueError("PDD v1 requires data.dataloader.dynamic_batch_size=false.")
    if _require_bool(
        dataloader.get("train_text_encoder", False),
        name="data.dataloader.train_text_encoder",
    ):
        raise ValueError("PDD requires cached text embeddings; train_text_encoder must be false.")
    _require_bool(dataloader.get("shuffle", True), name="data.dataloader.shuffle")
    if "metadata_index" in dataloader:
        raise ValueError(
            "PDD uses deterministic ordinal splits from metadata.json; "
            "data.dataloader.metadata_index is unsupported."
        )

    _reject_enabled(model.get("transformer_engine_linear"), name="global TE-linear conversion")
    _reject_enabled(model.get("peft"), name="PEFT/LoRA")
    _reject_enabled(model.get("peft_cfg"), name="PEFT/LoRA")
    _reject_enabled(raw.get("peft"), name="PEFT/LoRA")
    _reject_enabled(raw.get("peft_cfg"), name="PEFT/LoRA")
    _reject_enabled(model.get("guidance_embeds"), name="Qwen guidance embeddings")
    _reject_enabled(model.get("guidance_embeddings"), name="Qwen guidance embeddings")
    for option in (
        "device_map",
        "load_in_4bit",
        "load_in_8bit",
        "offload_folder",
        "offload_state_dict",
        "quantization_config",
    ):
        _reject_enabled(model.get(option), name=f"model loader option {option!r}")

    model_id = model.get("pretrained_model_name_or_path")
    if not isinstance(model_id, str) or not model_id:
        raise ValueError("model.pretrained_model_name_or_path must be a non-empty string.")
    model_revision = model.get("revision")
    if model_revision is not None and (
        not isinstance(model_revision, str)
        or len(model_revision) != 40
        or any(character not in "0123456789abcdefABCDEF" for character in model_revision)
    ):
        raise ValueError("model.revision must be null or a full 40-character commit hash.")
    if not Path(model_id).is_dir() and model_revision is None:
        raise ValueError("Remote PDD models require an exact model.revision commit hash.")

    learning_rate = _require_finite_real(
        optim.get("learning_rate", 2.0e-5),
        name="optim.learning_rate",
        minimum=0.0,
    )
    if learning_rate == 0.0:
        raise ValueError("optim.learning_rate must be > 0.")
    optimizer_target = optimizer_cfg.get("_target_", "torch.optim.AdamW")
    if optimizer_target != "torch.optim.AdamW":
        raise ValueError("PDD v1 requires optim.optimizer._target_='torch.optim.AdamW'.")
    allowed_optimizer_keys = {
        "_target_",
        "weight_decay",
        "betas",
        "eps",
        "amsgrad",
        "capturable",
        "differentiable",
        "foreach",
        "fused",
        "maximize",
    }
    unsupported_optimizer_keys = sorted(set(optimizer_cfg) - allowed_optimizer_keys)
    if unsupported_optimizer_keys:
        raise ValueError(f"unsupported PDD optimizer keys: {unsupported_optimizer_keys}.")
    for flag in ("amsgrad", "capturable", "differentiable", "foreach", "fused", "maximize"):
        if _require_bool(optimizer_cfg.get(flag, False), name=f"optim.optimizer.{flag}"):
            raise ValueError(f"PDD v1 requires optim.optimizer.{flag}=false.")
    weight_decay = _require_finite_real(
        optimizer_cfg.get("weight_decay", 0.0),
        name="optim.optimizer.weight_decay",
        minimum=0.0,
    )
    adam_betas_raw = optimizer_cfg.get("betas", [0.9, 0.999])
    if (
        not isinstance(adam_betas_raw, list | tuple)
        or len(adam_betas_raw) != 2
        or any(
            isinstance(beta, bool) or not isinstance(beta, int | float) for beta in adam_betas_raw
        )
    ):
        raise TypeError("optim.optimizer.betas must contain two real numbers.")
    adam_betas = (float(adam_betas_raw[0]), float(adam_betas_raw[1]))
    if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in adam_betas):
        raise ValueError("optim.optimizer.betas values must be finite and in [0, 1).")
    adam_eps = _require_finite_real(
        optimizer_cfg.get("eps", 1e-8),
        name="optim.optimizer.eps",
        minimum=0.0,
    )
    if adam_eps == 0.0:
        raise ValueError("optim.optimizer.eps must be > 0.")

    lr_decay_style = lr_scheduler.get("lr_decay_style", "constant")
    if lr_decay_style != "constant":
        raise ValueError("PDD v1 requires lr_scheduler.lr_decay_style='constant'.")
    lr_warmup_steps = _require_int_at_least(
        lr_scheduler.get("lr_warmup_steps", 0),
        name="lr_scheduler.lr_warmup_steps",
        minimum=0,
    )
    if lr_warmup_steps != 0:
        raise ValueError("PDD v1 requires lr_scheduler.lr_warmup_steps=0.")
    min_lr = _require_finite_real(
        lr_scheduler.get("min_lr", learning_rate),
        name="lr_scheduler.min_lr",
        minimum=0.0,
    )
    if min_lr != learning_rate:
        raise ValueError("lr_scheduler.min_lr must equal optim.learning_rate for constant PDD LR.")
    if "max_lr" in lr_scheduler:
        max_lr = _require_finite_real(
            lr_scheduler["max_lr"],
            name="lr_scheduler.max_lr",
            minimum=0.0,
        )
        if max_lr != learning_rate:
            raise ValueError(
                "lr_scheduler.max_lr must equal optim.learning_rate for constant PDD LR."
            )

    dp_size = fsdp.get("dp_size")
    if dp_size is not None and (type(dp_size) is not int or dp_size < 1):
        raise ValueError("fsdp.dp_size must be null or an integer >= 1.")
    activation_checkpointing = fsdp.get("activation_checkpointing", False)
    if type(activation_checkpointing) is not bool:
        raise TypeError("fsdp.activation_checkpointing must be bool.")
    for dimension in ("tp_size", "cp_size", "pp_size", "ep_size"):
        value = fsdp.get(dimension, 1)
        if type(value) is not int or value != 1:
            raise ValueError(f"PDD v1 supports pure data parallelism; fsdp.{dimension} must be 1.")

    pdd = PDDConfig(**dict(pdd_raw))
    if pdd.num_train_timesteps is not None:
        raise ValueError("Qwen-Image PDD requires pdd.num_train_timesteps=null.")

    checkpoint_enabled = _require_bool(checkpoint.get("enabled", True), name="checkpoint.enabled")
    save_consolidated = _require_bool(
        checkpoint.get("save_consolidated", False),
        name="checkpoint.save_consolidated",
    )
    if save_consolidated:
        raise ValueError("PDD training checkpoints require checkpoint.save_consolidated=false.")
    checkpoint_dir = checkpoint.get("checkpoint_dir", "checkpoints/pdd_qwen_image")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("checkpoint.checkpoint_dir must be a non-empty string.")
    model_save_format = checkpoint.get("model_save_format", "torch_save")
    if model_save_format != "torch_save":
        raise ValueError("PDD training checkpoints require model_save_format='torch_save'.")
    restore_from = checkpoint.get("restore_from")
    if restore_from is not None and (not isinstance(restore_from, str) or not restore_from):
        raise ValueError("checkpoint.restore_from must be null or a non-empty string.")
    if not checkpoint_enabled and restore_from is not None:
        raise ValueError("checkpoint.restore_from requires checkpoint.enabled=true.")
    fuse_qkv_projections = _require_bool(
        model.get("fuse_qkv_projections", False),
        name="model.fuse_qkv_projections",
    )

    seed = _require_int_at_least(raw.get("seed", 42), name="seed", minimum=0)
    max_steps = _require_int_at_least(
        step_scheduler.get("max_steps", 10_000),
        name="step_scheduler.max_steps",
        minimum=1,
    )
    num_epochs = _require_int_at_least(
        step_scheduler.get("num_epochs", 200),
        name="step_scheduler.num_epochs",
        minimum=1,
    )
    log_every = _require_int_at_least(
        step_scheduler.get("log_every", 10),
        name="step_scheduler.log_every",
        minimum=1,
    )
    ckpt_every_steps = _require_int_at_least(
        step_scheduler.get("ckpt_every_steps", 1_000),
        name="step_scheduler.ckpt_every_steps",
        minimum=1,
    )
    save_checkpoint_every_epoch = _require_bool(
        step_scheduler.get("save_checkpoint_every_epoch", False),
        name="step_scheduler.save_checkpoint_every_epoch",
    )
    if save_checkpoint_every_epoch:
        raise ValueError(
            "PDD exact resume requires step_scheduler.save_checkpoint_every_epoch=false."
        )
    local_batch_size = _require_int_at_least(
        step_scheduler.get("local_batch_size", 1),
        name="step_scheduler.local_batch_size",
        minimum=1,
    )
    data_batch_size = _require_int_at_least(
        dataloader.get("batch_size", local_batch_size),
        name="data.dataloader.batch_size",
        minimum=1,
    )
    if data_batch_size != local_batch_size:
        raise ValueError("data.dataloader.batch_size must equal step_scheduler.local_batch_size.")
    global_batch_size = step_scheduler.get("global_batch_size")
    if global_batch_size is not None:
        global_batch_size = _require_int_at_least(
            global_batch_size,
            name="step_scheduler.global_batch_size",
            minimum=1,
        )

    max_grad_norm = _require_finite_real(
        training_health.get("max_grad_norm", 1.0),
        name="training_health.max_grad_norm",
        minimum=0.0,
    )
    if max_grad_norm == 0.0:
        raise ValueError("training_health.max_grad_norm must be > 0.")
    zero_grad_warmup_steps = _require_int_at_least(
        training_health.get("zero_grad_warmup_steps", 0),
        name="training_health.zero_grad_warmup_steps",
        minimum=0,
    )
    validation_count = _require_int_at_least(
        validation.get("count", 2_000),
        name="validation.count",
        minimum=1,
    )
    validation_seed = _require_int_at_least(
        validation.get("seed", 2026),
        name="validation.seed",
        minimum=0,
    )
    split_seed = _require_int_at_least(
        validation.get("split_seed", 2026),
        name="validation.split_seed",
        minimum=0,
    )
    validation_every_steps = _require_int_at_least(
        validation.get("every_steps", 1_000),
        name="validation.every_steps",
        minimum=1,
    )

    guidance_rescale = _require_finite_real(
        guidance.get("rescale", 1.0),
        name="guidance.rescale",
        minimum=0.0,
    )
    if guidance_rescale > 1.0:
        raise ValueError("guidance.rescale must be <= 1.")
    guidance_eps = _require_finite_real(
        guidance.get("eps", 1e-5),
        name="guidance.eps",
        minimum=0.0,
    )
    if guidance_eps == 0.0:
        raise ValueError("guidance.eps must be > 0.")

    return PDDRecipeConfig(
        model_id=model_id,
        model_revision=model_revision,
        pdd=pdd,
        parallel=PDDParallelConfig(
            dp_size=dp_size,
            activation_checkpointing=activation_checkpointing,
        ),
        checkpoint=PDDCheckpointConfig(
            checkpoint_dir=checkpoint_dir,
            enabled=checkpoint_enabled,
            model_save_format=model_save_format,
            restore_from=restore_from,
            save_consolidated=save_consolidated,
        ),
        step_scheduler=PDDStepSchedulerConfig(
            max_steps=max_steps,
            num_epochs=num_epochs,
            log_every=log_every,
            ckpt_every_steps=ckpt_every_steps,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            save_checkpoint_every_epoch=save_checkpoint_every_epoch,
        ),
        training_health=PDDTrainingHealthConfig(
            max_grad_norm=max_grad_norm,
            zero_grad_warmup_steps=zero_grad_warmup_steps,
        ),
        validation=PDDValidationConfig(
            count=validation_count,
            seed=validation_seed,
            split_seed=split_seed,
            every_steps=validation_every_steps,
        ),
        guidance=PDDGuidanceConfig(rescale=guidance_rescale, eps=guidance_eps),
        seed=seed,
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        adam_betas=adam_betas,
        adam_eps=adam_eps,
        device=torch.device(model.get("device", "cuda" if torch.cuda.is_available() else "cpu")),
        dtype=_resolve_dtype(model.get("torch_dtype", "bfloat16")),
        fuse_qkv_projections=fuse_qkv_projections,
    )


def _projection_identity(projection: PDDOutputProjection) -> tuple[int, int, int | None]:
    return (
        id(projection),
        id(projection.weight),
        None if projection.bias is None else id(projection.bias),
    )


def _require_projection_identity(
    student: nn.Module,
    projection: PDDOutputProjection,
    expected: tuple[int, int, int | None],
    *,
    stage: str,
) -> None:
    if student.get_submodule("proj_out") is not projection:
        raise RuntimeError(f"PDD projection was replaced during {stage}.")
    if _projection_identity(projection) != expected:
        raise RuntimeError(f"PDD projection parameter identity changed during {stage}.")


def _require_projection_module(
    student: nn.Module,
    projection: PDDOutputProjection,
    *,
    stage: str,
) -> None:
    if student.get_submodule("proj_out") is not projection:
        raise RuntimeError(f"PDD projection module was replaced during {stage}.")


def _resolve_model_source(config: PDDRecipeConfig) -> str:
    if Path(config.model_id).is_dir():
        return config.model_id
    from huggingface_hub import snapshot_download

    if config.model_revision is None:
        raise ValueError("Remote PDD models require a pinned model revision.")
    model_source = snapshot_download(config.model_id, revision=config.model_revision)
    if Path(model_source).resolve().name != config.model_revision:
        raise RuntimeError(
            "Hugging Face resolved a model snapshot that does not match the pinned revision."
        )
    return model_source


def _load_unwrapped_transformer(
    config: PDDRecipeConfig, pipeline_type: Any
) -> tuple[Any, nn.Module]:
    pipe, loader_managers = pipeline_type.from_pretrained(
        _resolve_model_source(config),
        parallel_scheme=None,
        device=None,
        torch_dtype=config.dtype,
        move_to_device=False,
        load_for_training=True,
        components_to_load=["transformer"],
        peft_cfg=None,
        active_transformer="transformer",
        transformer_engine_linear=False,
        fuse_qkv_projections=False,
        compact_fused_qkv_projections=False,
        low_cpu_mem_usage=True,
        text_encoder=None,
        tokenizer=None,
        vae=None,
    )
    if loader_managers:
        raise RuntimeError("Unwrapped AutoModel load unexpectedly created parallel managers.")
    student = pipe.transformer
    if not isinstance(student, nn.Module):
        raise TypeError("AutoModel pipeline did not return an nn.Module transformer.")
    return pipe, student


def _stage_and_shard_training_models(
    student: nn.Module,
    teacher: nn.Module,
    projection: PDDOutputProjection,
    projection_identity: tuple[int, int, int | None],
    manager: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fuse_qkv_projections: bool,
) -> tuple[nn.Module, nn.Module]:
    """Move and shard one dense model at a time to bound setup-time GPU memory."""
    if fuse_qkv_projections and (
        not hasattr(student, "fuse_qkv_projections") or not hasattr(teacher, "fuse_qkv_projections")
    ):
        raise AttributeError("QKV fusion requires both Qwen transformers to expose the object API.")

    student.to(device=device, dtype=dtype)
    if fuse_qkv_projections:
        student.fuse_qkv_projections()
        if not any(getattr(module, "fused_projections", False) for module in student.modules()):
            logging.warning(
                "Qwen fuse_qkv_projections() was accepted but produced no fused attention "
                "modules in the pinned Diffusers release."
            )
    _require_projection_identity(student, projection, projection_identity, stage="student staging")
    student = manager.parallelize(student)
    _require_projection_module(student, projection, stage="student FSDP2 parallelization")

    # The student is already sharded before the dense teacher reaches the GPU, so multi-rank
    # setup never holds both complete Qwen transformers on one device.
    teacher.to(device=device, dtype=dtype)
    if fuse_qkv_projections:
        teacher.fuse_qkv_projections()
    teacher = manager.parallelize(teacher)
    return student, teacher


def _materialize_zero_step_adamw_state(optimizer: torch.optim.AdamW) -> None:
    """Create complete strict-DCP state without changing parameters or update numbering."""
    if type(optimizer) is not torch.optim.AdamW:
        raise TypeError("PDD state materialization requires the stock torch.optim.AdamW optimizer.")
    if optimizer.state:
        raise RuntimeError("PDD AdamW state must be empty before materialization.")
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("PDD AdamW parameters must not have gradients before materialization.")

    learning_rates = [group["lr"] for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        for parameter in parameters:
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for parameter in parameters:
            state = optimizer.state.get(parameter)
            if state is None or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise RuntimeError("PDD AdamW did not create complete checkpoint state.")
            step = state["step"]
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() != 1:
                raise RuntimeError("PDD AdamW created an unexpected initial step.")
            step.zero_()
    finally:
        for group, learning_rate in zip(optimizer.param_groups, learning_rates, strict=True):
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)


def build_pdd_setup(config: PDDRecipeConfig) -> PDDSetupArtifacts:
    """Compose released AutoModel APIs without editing or patching external packages."""
    if not isinstance(config, PDDRecipeConfig):
        raise TypeError(f"config must be PDDRecipeConfig, got {type(config).__name__}.")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before building the PDD FSDP2 setup.")

    automodel_snapshot = snapshot_installed_distribution()
    lifecycle: list[str] = []

    # Imports are intentionally delayed until the exact installed wheel has passed verification.
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.distributed import (
        DistributedSetup,
        FSDP2Config,
        ParallelismSizes,
    )
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

    pipe, student = _load_unwrapped_transformer(config, NeMoAutoDiffusionPipeline)
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    lifecycle.append("load/select")

    projection = convert_qwen_image_to_pdd(student, config.pdd)
    identity = _projection_identity(projection)
    metadata = PDDMetadata.from_config(config.pdd, projection)
    lifecycle.append("pdd_conversion")

    world_size = dist.get_world_size()
    if config.step_scheduler.global_batch_size is not None:
        effective_global_batch = config.step_scheduler.local_batch_size * world_size
        if effective_global_batch != config.step_scheduler.global_batch_size:
            raise ValueError(
                "PDD global batch mismatch: "
                f"local_batch_size={config.step_scheduler.local_batch_size} * "
                f"world_size={world_size} = {effective_global_batch}, configured "
                "step_scheduler.global_batch_size="
                f"{config.step_scheduler.global_batch_size}. PDD v1 requires one microbatch per "
                "optimizer update."
            )
    dp_size = config.parallel.dp_size or world_size
    if dp_size != world_size:
        raise ValueError(
            f"Pure-DP PDD requires fsdp.dp_size ({dp_size}) to equal world size ({world_size})."
        )
    strategy = FSDP2Config(activation_checkpointing=config.parallel.activation_checkpointing)
    distributed_setup = DistributedSetup.build(
        strategy=strategy,
        parallelism_sizes=ParallelismSizes(dp_size=dp_size),
        activation_checkpointing=config.parallel.activation_checkpointing,
        world_size=world_size,
    )
    mesh_context = distributed_setup.mesh_context
    manager = FSDP2Manager(
        distributed_setup.strategy_config,
        device_mesh=mesh_context.device_mesh,
        moe_mesh=mesh_context.moe_mesh,
    )
    student, teacher = _stage_and_shard_training_models(
        student,
        teacher,
        projection,
        identity,
        manager,
        device=config.device,
        dtype=config.dtype,
        fuse_qkv_projections=config.fuse_qkv_projections,
    )
    pipe.transformer = student
    # Keep the public lifecycle summary stable even though placement, optional QKV fusion, and
    # FSDP2 are deliberately interleaved per model to cap peak device memory.
    lifecycle.extend(("device", "qkv", "parallelize"))

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PDD student has no trainable parameters after FSDP2 setup.")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("PDD teacher became trainable during setup.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.adam_betas,
        eps=config.adam_eps,
        amsgrad=False,
        capturable=False,
        differentiable=False,
        foreach=False,
        fused=False,
        maximize=False,
    )
    _materialize_zero_step_adamw_state(optimizer)
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if not any(parameter is projection.weight for parameter in optimizer_parameters):
        raise RuntimeError("PDD projection parameters are missing from the optimizer.")
    lifecycle.append("optimizer")

    checkpoint_keys = tuple(student.state_dict().keys())
    projection_key = "proj_out.weight"
    if projection_key not in checkpoint_keys:
        raise RuntimeError(
            f"PDD projection key {projection_key!r} is missing from checkpoint state."
        )
    checkpoint_config = CheckpointingConfig(
        enabled=config.checkpoint.enabled,
        checkpoint_dir=config.checkpoint.checkpoint_dir,
        model_save_format=config.checkpoint.model_save_format,
        model_repo_id=config.model_id,
        save_consolidated=config.checkpoint.save_consolidated,
        is_peft=False,
        model_state_dict_keys=list(checkpoint_keys),
    )
    checkpointer = checkpoint_config.build(
        dp_rank=dist.get_rank(),
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    lifecycle.append("checkpoint")

    return PDDSetupArtifacts(
        pipe=pipe,
        student=student,
        teacher=teacher,
        projection=projection,
        optimizer=optimizer,
        distributed_setup=distributed_setup,
        fsdp_manager=manager,
        checkpointer=checkpointer,
        metadata=metadata,
        checkpoint_keys=checkpoint_keys,
        lifecycle=tuple(lifecycle),
        automodel_snapshot=automodel_snapshot,
    )


def build_pdd_export_setup(config: PDDRecipeConfig) -> PDDExportSetupArtifacts:
    """Build only the converted/sharded student needed for collective export."""
    if not isinstance(config, PDDRecipeConfig):
        raise TypeError(f"config must be PDDRecipeConfig, got {type(config).__name__}.")
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Initialize torch.distributed before building PDD export setup.")
    automodel_snapshot = snapshot_installed_distribution()

    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig
    from nemo_automodel.components.distributed import (
        DistributedSetup,
        FSDP2Config,
        ParallelismSizes,
    )
    from nemo_automodel.components.distributed.fsdp2 import FSDP2Manager

    lifecycle = ["load/select"]
    pipe, student = _load_unwrapped_transformer(config, NeMoAutoDiffusionPipeline)
    raw_transformer_config = getattr(student, "config", None)
    to_dict = getattr(raw_transformer_config, "to_dict", None)
    if callable(to_dict):
        transformer_config = to_dict()
    elif isinstance(raw_transformer_config, Mapping):
        transformer_config = dict(raw_transformer_config)
    else:
        raise TypeError("Qwen transformer config must expose to_dict() or Mapping.")
    if not isinstance(transformer_config, Mapping):
        raise TypeError("Qwen transformer to_dict() must return a mapping.")

    projection = convert_qwen_image_to_pdd(student, config.pdd)
    identity = _projection_identity(projection)
    metadata = PDDMetadata.from_config(config.pdd, projection)
    lifecycle.append("pdd_conversion")
    student.to(device=config.device, dtype=config.dtype)
    _require_projection_identity(student, projection, identity, stage="device placement")
    lifecycle.append("device")

    if config.fuse_qkv_projections:
        if not hasattr(student, "fuse_qkv_projections"):
            raise AttributeError("QKV fusion requires Qwen to expose the object API.")
        student.fuse_qkv_projections()
    _require_projection_identity(student, projection, identity, stage="QKV fusion")
    lifecycle.append("qkv")

    world_size = dist.get_world_size()
    dp_size = config.parallel.dp_size or world_size
    if dp_size != world_size:
        raise ValueError(
            f"Pure-DP PDD export requires fsdp.dp_size ({dp_size}) to equal world size "
            f"({world_size})."
        )
    strategy = FSDP2Config(activation_checkpointing=False)
    distributed_setup = DistributedSetup.build(
        strategy=strategy,
        parallelism_sizes=ParallelismSizes(dp_size=dp_size),
        activation_checkpointing=False,
        world_size=world_size,
    )
    mesh_context = distributed_setup.mesh_context
    manager = FSDP2Manager(
        distributed_setup.strategy_config,
        device_mesh=mesh_context.device_mesh,
        moe_mesh=mesh_context.moe_mesh,
    )
    student = manager.parallelize(student)
    pipe.transformer = student
    _require_projection_module(student, projection, stage="FSDP2 export parallelization")
    lifecycle.append("parallelize")

    checkpoint_keys = tuple(student.state_dict())
    if "proj_out.weight" not in checkpoint_keys:
        raise RuntimeError("PDD projection is missing from the export checkpoint key inventory.")
    checkpoint_config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=config.checkpoint.checkpoint_dir,
        model_save_format="torch_save",
        model_repo_id=config.model_id,
        save_consolidated=False,
        is_peft=False,
        model_state_dict_keys=list(checkpoint_keys),
    )
    checkpointer = checkpoint_config.build(
        dp_rank=dist.get_rank(),
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    lifecycle.append("checkpoint")
    return PDDExportSetupArtifacts(
        pipe=pipe,
        student=student,
        projection=projection,
        distributed_setup=distributed_setup,
        fsdp_manager=manager,
        checkpointer=checkpointer,
        metadata=metadata,
        checkpoint_keys=checkpoint_keys,
        transformer_config=transformer_config,
        lifecycle=tuple(lifecycle),
        automodel_snapshot=automodel_snapshot,
    )


def build_pdd_training_artifacts(
    setup: PDDSetupArtifacts,
    config: PDDRecipeConfig,
) -> PDDTrainingArtifacts:
    """Layer the direct-update pipeline, constant-LR scheduler, and ranked RNG on setup."""
    if not isinstance(setup, PDDSetupArtifacts):
        raise TypeError("setup must be PDDSetupArtifacts.")
    if not isinstance(config, PDDRecipeConfig):
        raise TypeError("config must be PDDRecipeConfig.")
    from nemo_automodel.components.training.rng import StatefulRNG

    from .training import PDDTrainer

    adapter = QwenImagePDDAdapter(
        config.pdd,
        guidance_rescale=config.guidance.rescale,
        guidance_eps=config.guidance.eps,
    )
    pipeline = PDDPipeline(setup.student, setup.teacher, config.pdd, adapter)
    scheduler = torch.optim.lr_scheduler.LambdaLR(setup.optimizer, lr_lambda=lambda _: 1.0)
    rng = StatefulRNG(config.seed, ranked=True)
    trainer = PDDTrainer(
        pipeline,
        setup.optimizer,
        projection=setup.projection,
        max_grad_norm=config.training_health.max_grad_norm,
        warmup_steps=config.training_health.zero_grad_warmup_steps,
    )
    return PDDTrainingArtifacts(
        pipeline=pipeline,
        trainer=trainer,
        scheduler=scheduler,
        rng=rng,
    )


def initialize_pdd_distributed(*, backend: str, timeout_minutes: int = 60) -> Any:
    """Verify the wheel, then initialize through AutoModel's released public API."""
    snapshot_installed_distribution()
    from nemo_automodel.components.distributed import initialize_distributed

    return initialize_distributed(backend=backend, timeout_minutes=timeout_minutes)


class PDDDiffusionRecipe:
    """Compose released AutoModel components around the PDD-specific update."""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.raw_config = _config_to_mapping(cfg)
        self.config = resolve_pdd_recipe_config(self.raw_config)

    def setup(self) -> None:
        """Build data, converted models, AutoModel scheduling, and strict resume state."""
        config = self.config
        self.dist_env = initialize_pdd_distributed(
            backend="nccl" if config.device.type == "cuda" else "gloo",
            timeout_minutes=60,
        )
        from nemo_automodel.components.loggers.log_utils import setup_logging
        from nemo_automodel.components.training.step_scheduler import StepScheduler

        setup_logging()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.dataloader, self.sampler = _build_training_dataloader(
            self.raw_config,
            config,
            dp_rank=self.rank,
            dp_world_size=self.world_size,
        )
        self.validation_dataloader, self.validation_sampler = _build_validation_dataloader(
            self.raw_config,
            config,
            dp_rank=self.rank,
            dp_world_size=self.world_size,
        )
        (
            self.snapshot_report,
            train_ordered_id_sha256,
            heldout_ordered_id_sha256,
        ) = _validate_dataset_contract(
            self.sampler.dataset,
            self.validation_sampler.dataset,
            config,
        )
        self.validation_assignments, self.validation_masks = _build_validation_plan(
            self.validation_sampler,
            config,
        )

        self.setup_artifacts = build_pdd_setup(config)
        self.expected_latent_channels, self.expected_condition_features = (
            self._resolve_transformer_dimensions(self.setup_artifacts.student)
        )
        self.training = build_pdd_training_artifacts(self.setup_artifacts, config)
        global_batch_size = (
            config.step_scheduler.global_batch_size
            or config.step_scheduler.local_batch_size * self.world_size
        )
        self.step_scheduler = StepScheduler(
            global_batch_size=global_batch_size,
            local_batch_size=config.step_scheduler.local_batch_size,
            dp_size=self.world_size,
            ckpt_every_steps=config.step_scheduler.ckpt_every_steps,
            save_checkpoint_every_epoch=False,
            dataloader=self.dataloader,
            val_every_steps=None,
            start_step=0,
            start_epoch=0,
            num_epochs=config.step_scheduler.num_epochs,
            max_steps=config.step_scheduler.max_steps,
        )
        if self.step_scheduler.grad_acc_steps != 1:
            raise ValueError("PDD v1 requires exactly one microbatch per optimizer update.")

        identity = build_pdd_checkpoint_identity(
            metadata=self.setup_artifacts.metadata,
            model_id=config.model_id,
            model_revision=config.model_revision,
            guidance_scale=config.pdd.guidance_scale,
            guidance_rescale=config.guidance.rescale,
            guidance_eps=config.guidance.eps,
            automodel_snapshot=self.setup_artifacts.automodel_snapshot,
            ordered_train_id_sha256=train_ordered_id_sha256,
            ordered_heldout_id_sha256=heldout_ordered_id_sha256,
            dataset_snapshot_sha256=self.snapshot_report["dataset_snapshot_sha256"],
            local_batch_size=config.step_scheduler.local_batch_size,
            grad_accumulation_steps=1,
            training_seed=config.seed,
            validation_seed=config.validation.seed,
            validation_every_steps=config.validation.every_steps,
            max_grad_norm=config.training_health.max_grad_norm,
            zero_grad_warmup_steps=config.training_health.zero_grad_warmup_steps,
            activation_checkpointing=config.parallel.activation_checkpointing,
            dtype=str(config.dtype).removeprefix("torch."),
            optimizer=self.setup_artifacts.optimizer,
            scheduler=self.training.scheduler,
        )
        self.checkpoint_manager = PDDCheckpointManager(
            root=config.checkpoint.checkpoint_dir,
            checkpointer=self.setup_artifacts.checkpointer,
            model=self.setup_artifacts.student,
            optimizer=self.setup_artifacts.optimizer,
            scheduler=self.training.scheduler,
            step_scheduler=self.step_scheduler,
            trainer=self.training.trainer,
            sampler=self.sampler,
            rng=self.training.rng,
            identity=identity,
        )
        self.resume = self.checkpoint_manager.load(config.checkpoint.restore_from)
        self.resume_pending = self.resume is not None
        self._log_setup()

    @staticmethod
    def _resolve_transformer_dimensions(student: nn.Module) -> tuple[int, int]:
        transformer_config = getattr(student, "config", None)
        if isinstance(transformer_config, Mapping):
            in_channels = transformer_config.get("in_channels")
            condition_features = transformer_config.get("joint_attention_dim")
        else:
            in_channels = getattr(transformer_config, "in_channels", None)
            condition_features = getattr(transformer_config, "joint_attention_dim", None)
        if type(in_channels) is not int or in_channels <= 0 or in_channels % 4:
            raise RuntimeError("constructed Qwen transformer has invalid packed in_channels.")
        if type(condition_features) is not int or condition_features <= 0:
            raise RuntimeError("constructed Qwen transformer has invalid joint_attention_dim.")
        return in_channels // 4, condition_features

    def _log_setup(self) -> None:
        if self.rank != 0:
            return
        if self.resume is not None:
            logging.info(
                "PDD resume selected: checkpoint=%s parent=%s step=%d sample_slots=%d "
                "expected_first_sample_ids=%s",
                self.resume.checkpoint_path,
                self.resume.parent_checkpoint,
                self.resume.completed_steps,
                self.resume.sample_slots_consumed,
                self.resume.expected_next_sample_ids,
            )
        logging.info(
            "PDD dataset verified: snapshot_sha256=%s metadata_sha256=%s "
            "train=%d validation=%d root=%s",
            self.snapshot_report["dataset_snapshot_sha256"],
            self.snapshot_report["metadata_sha256"],
            self.snapshot_report["train_samples"],
            self.snapshot_report["validation_samples"],
            self.snapshot_report["cache_root"],
        )
        logging.info(
            "PDD setup complete: lifecycle=%s student_keys=%d AutoModel=%s",
            self.setup_artifacts.lifecycle,
            len(self.setup_artifacts.checkpoint_keys),
            self.setup_artifacts.automodel_snapshot["version"],
        )

    def _prepared_training_batches(self):
        iterator = _collective_training_iterator(self.dataloader, self.sampler)
        while True:
            next_batch = _collective_training_batch(
                iterator,
                sampler=self.sampler,
                resume=self.resume,
                resume_pending=self.resume_pending,
                device=self.config.device,
                dtype=self.config.dtype,
                require_negative_condition=self.config.pdd.guidance_scale is not None,
                expected_batch_size=self.config.step_scheduler.local_batch_size,
                expected_latent_channels=self.expected_latent_channels,
                expected_condition_features=self.expected_condition_features,
            )
            if next_batch is None:
                return
            yield next_batch

    def _run_validation(self, completed_step: int) -> None:
        from .training import run_pdd_validation

        self.validation_sampler.set_epoch(0)
        self.validation_sampler.load_state_dict({"epoch": 0, "batches_yielded": 0})
        result = run_pdd_validation(
            self.training.pipeline,
            _iter_validation_batches(
                self.validation_dataloader,
                self.validation_masks,
                self.config,
                self.expected_latent_channels,
                self.expected_condition_features,
            ),
            self.validation_assignments,
            validation_seed=self.config.validation.seed,
        )
        if self.rank == 0:
            logging.info(
                "PDD validation step=%d loss=%.12g pairs=%d starts=%d heads=%d "
                "ordered_id_sha256=%s records=%d",
                completed_step,
                result.mean_loss,
                result.pair_count,
                result.start_count,
                result.head_count,
                result.ordered_id_sha256,
                len(result.records),
            )

    def _log_step(self, diagnostics: Any, data_wait_seconds: float, step_seconds: float) -> None:
        timing = torch.tensor(
            [data_wait_seconds, step_seconds],
            dtype=torch.float64,
            device=self.config.device,
        )
        dist.all_reduce(timing, op=dist.ReduceOp.MAX)
        peak_memory = (
            torch.cuda.max_memory_allocated(self.config.device)
            if self.config.device.type == "cuda"
            else 0
        )
        memory = torch.tensor(peak_memory, dtype=torch.int64, device=self.config.device)
        dist.all_reduce(memory, op=dist.ReduceOp.MAX)
        global_samples = self.config.step_scheduler.local_batch_size * self.world_size
        throughput = global_samples / max(float(timing[1].item()), 1e-12)
        coverage = self.training.trainer.coverage
        bin_loss = [
            None if count == 0 else float(loss_sum / count)
            for loss_sum, count in zip(
                coverage.bin_loss_sums.tolist(),
                coverage.bin_counts.tolist(),
            )
        ]
        if self.rank == 0:
            logging.info(
                "PDD step=%d loss=%.6g grad_norm=%.6g nominal_update_ratio=%.6g "
                "projection_update_ratio=%s lr=%.6g student_rms=%.6g "
                "teacher_rms=%.6g student_teacher_rms_ratio=%.6g "
                "reconstruction_rms=%.6g pairs=%d n_coverage=%s k_coverage=%s "
                "bins=%s bin_loss=%s samples_per_second=%.3f "
                "data_wait_seconds=%.4f peak_memory_bytes=%d",
                diagnostics.completed_step,
                diagnostics.loss,
                diagnostics.grad_norm,
                diagnostics.student_adamw_nominal_update_ratio,
                diagnostics.pdd_projection_update_ratio,
                diagnostics.learning_rate,
                diagnostics.student_velocity_rms,
                diagnostics.teacher_velocity_rms,
                diagnostics.student_teacher_velocity_rms_ratio,
                diagnostics.reconstructed_state_rms,
                int((coverage.pair_counts > 0).sum()),
                _coverage_axis(coverage.n_counts, coverage.n_loss_sums),
                _coverage_axis(coverage.k_counts, coverage.k_loss_sums),
                coverage.bin_counts.tolist(),
                bin_loss,
                throughput,
                float(timing[0].item()),
                int(memory.item()),
            )
        if self.config.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.config.device)

    def run_train_validation_loop(self) -> None:
        """Train through AutoModel StepScheduler without weakening PDD resume semantics."""
        data_wait_started = time.perf_counter()
        try:
            for _epoch in self.step_scheduler.epochs:
                self.step_scheduler.dataloader = self._prepared_training_batches()
                for batch_group in self.step_scheduler:
                    if len(batch_group) != 1:
                        raise RuntimeError("PDD v1 requires one microbatch per optimizer update.")
                    if self.step_scheduler.step != self.training.trainer.completed_steps:
                        raise RuntimeError(
                            "PDD trainer and AutoModel StepScheduler disagree before the update."
                        )
                    (batch, sample_ids) = batch_group[0]
                    data_wait_seconds = time.perf_counter() - data_wait_started
                    if self.resume_pending:
                        if self.resume is None:
                            raise RuntimeError("PDD resume is pending without restored state.")
                        if self.rank == 0:
                            logging.info(
                                "PDD resume first batch verified: checkpoint=%s sample_ids=%s",
                                self.resume.checkpoint_path,
                                sample_ids,
                            )
                        self.resume_pending = False

                    step_started = time.perf_counter()
                    next_step = self.training.trainer.completed_steps + 1
                    diagnostics = self.training.trainer.train_step(
                        batch,
                        measure_updates=(next_step % self.config.step_scheduler.log_every == 0),
                    )
                    self.training.scheduler.step()
                    self.sampler.commit(sample_ids)
                    if self.sampler.remaining_batches == 0:
                        self.sampler.set_epoch(self.sampler.epoch + 1)
                    if self.training.trainer.completed_steps != self.step_scheduler.step + 1:
                        raise RuntimeError(
                            "PDD trainer and AutoModel StepScheduler disagree after the update."
                        )
                    serialized_step = self.step_scheduler.state_dict()["step"]
                    if serialized_step != self.training.trainer.completed_steps:
                        raise RuntimeError("AutoModel StepScheduler serialized the wrong PDD step.")
                    step_seconds = time.perf_counter() - step_started

                    completed_step = diagnostics.completed_step
                    is_final_step = self.step_scheduler.is_last_step
                    if completed_step % self.config.step_scheduler.log_every == 0:
                        self._log_step(diagnostics, data_wait_seconds, step_seconds)
                    if completed_step % self.config.validation.every_steps == 0 or is_final_step:
                        self._run_validation(completed_step)
                    if self.config.checkpoint.enabled and self.step_scheduler.is_ckpt_step:
                        self.checkpoint_manager.save()
                    data_wait_started = time.perf_counter()
        finally:
            self.setup_artifacts.checkpointer.close()
