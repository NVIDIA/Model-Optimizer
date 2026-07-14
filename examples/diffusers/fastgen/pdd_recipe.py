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
from portable_cache import validate_relative_reference
from torch import nn
from verify_readonly_automodel import snapshot_installed_distribution

from modelopt.torch.fastgen import PDDConfig, PDDMetadata, PDDOutputProjection, PDDPipeline
from modelopt.torch.fastgen.plugins.qwen_image_pdd import (
    QwenImagePDDAdapter,
    convert_qwen_image_to_pdd,
)


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
class PDDTrainingConfig:
    """Direct-update and observability settings for the standalone PDD lifecycle."""

    seed: int = 42
    max_steps: int = 10_000
    max_grad_norm: float = 1.0
    zero_grad_warmup_steps: int = 0
    log_every_steps: int = 10
    checkpoint_every_steps: int = 1_000
    validation_every_steps: int = 1_000
    local_batch_size: int = 1
    global_batch_size: int | None = None
    grad_accumulation_steps: int = 1
    validation_seed: int = 2026


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
    training: PDDTrainingConfig
    guidance: PDDGuidanceConfig
    learning_rate: float
    weight_decay: float
    adam_betas: tuple[float, float]
    adam_eps: float
    all_metadata_index: str
    train_metadata_index: str
    validation_metadata_index: str
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


def resolve_pdd_recipe_config(raw: Mapping[str, Any]) -> PDDRecipeConfig:
    """Resolve PDD and reject TE-linear, PEFT, and guidance embeddings before loading."""
    raw = _as_mapping(raw, name="config")
    model = _as_mapping(raw.get("model"), name="model")
    pdd_raw = _as_mapping(raw.get("pdd"), name="pdd")
    fsdp = _as_mapping(raw.get("fsdp", {}), name="fsdp")
    optim = _as_mapping(raw.get("optim", {}), name="optim")
    checkpoint = _as_mapping(raw.get("checkpoint", {}), name="checkpoint")
    training = _as_mapping(raw.get("training", {}), name="training")
    guidance = _as_mapping(raw.get("guidance", {}), name="guidance")
    data = _as_mapping(raw.get("data", {}), name="data")
    dataloader = _as_mapping(data.get("dataloader", {}), name="data.dataloader")

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
    all_metadata_index = validate_relative_reference(
        data.get("all_metadata_index", "metadata.json"),
        label="data.all_metadata_index",
    ).as_posix()
    train_metadata_index = validate_relative_reference(
        dataloader.get("metadata_index", "metadata_train.json"),
        label="data.dataloader.metadata_index",
    ).as_posix()
    validation_metadata_index = validate_relative_reference(
        data.get("validation_metadata_index", "metadata_heldout.json"),
        label="data.validation_metadata_index",
    ).as_posix()

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
    adam_betas_raw = optim.get("betas", [0.9, 0.999])
    if (
        not isinstance(adam_betas_raw, list | tuple)
        or len(adam_betas_raw) != 2
        or any(
            isinstance(beta, bool) or not isinstance(beta, int | float) for beta in adam_betas_raw
        )
    ):
        raise TypeError("optim.betas must contain two real numbers.")
    adam_betas = tuple(float(beta) for beta in adam_betas_raw)
    if any(not math.isfinite(beta) or not 0.0 <= beta < 1.0 for beta in adam_betas):
        raise ValueError("optim.betas values must be finite and in [0, 1).")
    adam_eps = _require_finite_real(
        optim.get("eps", 1e-8),
        name="optim.eps",
        minimum=0.0,
    )
    if adam_eps == 0.0:
        raise ValueError("optim.eps must be > 0.")

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
    restore_from = checkpoint.get("restore_from")
    if restore_from is not None and (not isinstance(restore_from, str) or not restore_from):
        raise ValueError("checkpoint.restore_from must be null or a non-empty string.")
    if not checkpoint_enabled and restore_from is not None:
        raise ValueError("checkpoint.restore_from requires checkpoint.enabled=true.")

    seed = _require_int_at_least(training.get("seed", 42), name="training.seed", minimum=0)
    max_steps = _require_int_at_least(
        training.get("max_steps", 10_000), name="training.max_steps", minimum=1
    )
    zero_grad_warmup_steps = _require_int_at_least(
        training.get("zero_grad_warmup_steps", 0),
        name="training.zero_grad_warmup_steps",
        minimum=0,
    )
    log_every_steps = _require_int_at_least(
        training.get("log_every_steps", 10),
        name="training.log_every_steps",
        minimum=1,
    )
    checkpoint_every_steps = _require_int_at_least(
        training.get("checkpoint_every_steps", 1_000),
        name="training.checkpoint_every_steps",
        minimum=1,
    )
    validation_every_steps = _require_int_at_least(
        training.get("validation_every_steps", 1_000),
        name="training.validation_every_steps",
        minimum=1,
    )
    grad_accumulation_steps = _require_int_at_least(
        training.get("grad_accumulation_steps", 1),
        name="training.grad_accumulation_steps",
        minimum=1,
    )
    if grad_accumulation_steps != 1:
        raise ValueError("PDD v1 exact resume requires training.grad_accumulation_steps=1.")
    local_batch_size = _require_int_at_least(
        dataloader.get("batch_size", training.get("local_batch_size", 1)),
        name="data.dataloader.batch_size",
        minimum=1,
    )
    global_batch_size = training.get("global_batch_size")
    if global_batch_size is not None:
        global_batch_size = _require_int_at_least(
            global_batch_size,
            name="training.global_batch_size",
            minimum=1,
        )
    validation_seed = _require_int_at_least(
        training.get("validation_seed", 2026),
        name="training.validation_seed",
        minimum=0,
    )
    max_grad_norm = _require_finite_real(
        training.get("max_grad_norm", 1.0),
        name="training.max_grad_norm",
        minimum=0.0,
    )
    if max_grad_norm == 0.0:
        raise ValueError("training.max_grad_norm must be > 0.")
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
        training=PDDTrainingConfig(
            seed=seed,
            max_steps=max_steps,
            max_grad_norm=max_grad_norm,
            zero_grad_warmup_steps=zero_grad_warmup_steps,
            log_every_steps=log_every_steps,
            checkpoint_every_steps=checkpoint_every_steps,
            validation_every_steps=validation_every_steps,
            local_batch_size=local_batch_size,
            global_batch_size=global_batch_size,
            grad_accumulation_steps=grad_accumulation_steps,
            validation_seed=validation_seed,
        ),
        guidance=PDDGuidanceConfig(rescale=guidance_rescale, eps=guidance_eps),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        adam_betas=adam_betas,
        adam_eps=adam_eps,
        all_metadata_index=all_metadata_index,
        train_metadata_index=train_metadata_index,
        validation_metadata_index=validation_metadata_index,
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
    if config.training.global_batch_size is not None:
        effective_global_batch = (
            config.training.local_batch_size * world_size * config.training.grad_accumulation_steps
        )
        if effective_global_batch != config.training.global_batch_size:
            raise ValueError(
                "PDD global batch mismatch: "
                f"local_batch_size={config.training.local_batch_size} * world_size={world_size} "
                f"* grad_accumulation_steps={config.training.grad_accumulation_steps} "
                f"= {effective_global_batch}, configured "
                f"training.global_batch_size={config.training.global_batch_size}."
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
        betas=config.adam_betas,
        eps=config.adam_eps,
        amsgrad=False,
        capturable=False,
        differentiable=False,
        foreach=False,
        fused=False,
        maximize=False,
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
    from pdd_training import PDDTrainer

    adapter = QwenImagePDDAdapter(
        config.pdd,
        guidance_rescale=config.guidance.rescale,
        guidance_eps=config.guidance.eps,
    )
    pipeline = PDDPipeline(setup.student, setup.teacher, config.pdd, adapter)
    scheduler = torch.optim.lr_scheduler.LambdaLR(setup.optimizer, lr_lambda=lambda _: 1.0)
    rng = StatefulRNG(config.training.seed, ranked=True)
    trainer = PDDTrainer(
        pipeline,
        setup.optimizer,
        projection=setup.projection,
        max_grad_norm=config.training.max_grad_norm,
        warmup_steps=config.training.zero_grad_warmup_steps,
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
