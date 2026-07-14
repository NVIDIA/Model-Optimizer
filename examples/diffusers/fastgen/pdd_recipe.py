# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelOpt-owned construction of a Qwen-Image PDD student and frozen teacher."""

from __future__ import annotations

import copy
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from verify_readonly_automodel import snapshot_installed_distribution

from modelopt.torch.fastgen import PDDConfig, PDDMetadata, PDDOutputProjection
from modelopt.torch.fastgen.plugins.qwen_image_pdd import convert_qwen_image_to_pdd


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
    save_consolidated: bool = False


@dataclass(frozen=True)
class PDDRecipeConfig:
    """Resolved setup inputs; incompatible mutation modes have already been rejected."""

    model_id: str
    model_revision: str | None
    pdd: PDDConfig
    parallel: PDDParallelConfig
    checkpoint: PDDCheckpointConfig
    learning_rate: float
    weight_decay: float
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


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _reject_enabled(value: Any, *, name: str) -> None:
    if value is None or value is False or value == {}:
        return
    raise ValueError(f"PDD does not support {name}; disable it before model loading.")


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be bool.")
    return value


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


def resolve_pdd_recipe_config(raw: Mapping[str, Any]) -> PDDRecipeConfig:
    """Resolve PDD and reject TE-linear, PEFT, and guidance embeddings before loading."""
    raw = _as_mapping(raw, name="config")
    model = _as_mapping(raw.get("model"), name="model")
    pdd_raw = _as_mapping(raw.get("pdd"), name="pdd")
    fsdp = _as_mapping(raw.get("fsdp", {}), name="fsdp")
    optim = _as_mapping(raw.get("optim", {}), name="optim")
    checkpoint = _as_mapping(raw.get("checkpoint", {}), name="checkpoint")

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
    if not Path(model_id).is_dir():
        if model_revision is None:
            raise ValueError("Remote PDD models require an exact model.revision commit hash.")
    learning_rate = optim.get("learning_rate", 2.0e-5)
    weight_decay = optim.get("weight_decay", 0.0)
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
        raise TypeError("optim.learning_rate must be a real number.")
    if isinstance(weight_decay, bool) or not isinstance(weight_decay, int | float):
        raise TypeError("optim.weight_decay must be a real number.")
    if not math.isfinite(learning_rate) or not math.isfinite(weight_decay):
        raise ValueError("optim.learning_rate and weight_decay must be finite.")
    if learning_rate <= 0 or weight_decay < 0:
        raise ValueError("optim.learning_rate must be > 0 and weight_decay must be >= 0.")

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
        checkpoint.get("save_consolidated", False), name="checkpoint.save_consolidated"
    )
    if save_consolidated:
        raise ValueError("PDD training checkpoints require checkpoint.save_consolidated=false.")
    checkpoint_dir = checkpoint.get("checkpoint_dir", "checkpoints/pdd_qwen_image")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("checkpoint.checkpoint_dir must be a non-empty string.")
    model_save_format = checkpoint.get("model_save_format", "torch_save")
    if model_save_format != "torch_save":
        raise ValueError("PDD training checkpoints require model_save_format='torch_save'.")
    fuse_qkv_projections = _require_bool(
        model.get("fuse_qkv_projections", False), name="model.fuse_qkv_projections"
    )

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
            save_consolidated=save_consolidated,
        ),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
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

    if Path(config.model_id).is_dir():
        model_source = config.model_id
    else:
        from huggingface_hub import snapshot_download

        if config.model_revision is None:
            raise ValueError("Remote PDD models require a pinned model revision.")
        model_source = snapshot_download(config.model_id, revision=config.model_revision)
        if Path(model_source).resolve().name != config.model_revision:
            raise RuntimeError(
                "Hugging Face resolved a model snapshot that does not match the pinned revision."
            )

    pipe, loader_managers = NeMoAutoDiffusionPipeline.from_pretrained(
        model_source,
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
    teacher = copy.deepcopy(student).eval().requires_grad_(False)
    lifecycle.append("load/select")

    projection = convert_qwen_image_to_pdd(student, config.pdd)
    identity = _projection_identity(projection)
    metadata = PDDMetadata.from_config(config.pdd, projection)
    lifecycle.append("pdd_conversion")

    student.to(device=config.device, dtype=config.dtype)
    teacher.to(device=config.device, dtype=config.dtype)
    _require_projection_identity(student, projection, identity, stage="device placement")
    lifecycle.append("device")

    if config.fuse_qkv_projections:
        if not hasattr(student, "fuse_qkv_projections") or not hasattr(
            teacher, "fuse_qkv_projections"
        ):
            raise AttributeError(
                "QKV fusion requires both Qwen transformers to expose the object API."
            )
        student.fuse_qkv_projections()
        teacher.fuse_qkv_projections()
        if not any(getattr(module, "fused_projections", False) for module in student.modules()):
            logging.warning(
                "Qwen fuse_qkv_projections() was accepted but produced no fused attention "
                "modules in the pinned Diffusers release."
            )
    _require_projection_identity(student, projection, identity, stage="QKV fusion")
    lifecycle.append("qkv")

    world_size = dist.get_world_size()
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
    student = manager.parallelize(student)
    teacher = manager.parallelize(teacher)
    pipe.transformer = student
    # FSDP2 shards Parameters in place and may replace the Parameter objects. The registered
    # projection module and FQN must survive; optimizer identity is checked against the new,
    # live post-FSDP Parameters below.
    _require_projection_module(student, projection, stage="FSDP2 parallelization")
    lifecycle.append("parallelize")

    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("PDD student has no trainable parameters after FSDP2 setup.")
    if any(parameter.requires_grad for parameter in teacher.parameters()):
        raise RuntimeError("PDD teacher became trainable during setup.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
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


def initialize_pdd_distributed(*, backend: str, timeout_minutes: int = 60) -> Any:
    """Verify the wheel, then initialize through AutoModel's released public API."""
    snapshot_installed_distribution()
    from nemo_automodel.components.distributed import initialize_distributed

    return initialize_distributed(backend=backend, timeout_minutes=timeout_minutes)
