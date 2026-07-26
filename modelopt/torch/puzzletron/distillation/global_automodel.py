# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import math
import os
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Literal

from ..dataset.config import DataLayout, Modality, PuzzletronDataSpec
from ..identity import cache_key, canonicalize

logger = logging.getLogger(__name__)

__all__ = [
    "GlobalKDConfig",
    "GlobalKDResult",
    "KDLossTermConfig",
    "build_global_kd_config",
    "build_automodel_global_kd_recipe",
    "run_automodel_global_kd",
    "run_global_kd",
]


@dataclass(frozen=True, kw_only=True)
class KDLossTermConfig:
    metric: Literal["kld", "tvd"] = "kld"
    temperature: float = 1.0
    chunk_size: int = 0

    def __post_init__(self):
        if self.metric not in ("kld", "tvd"):
            raise ValueError(f"Unsupported KD metric {self.metric!r}")
        if self.temperature <= 0:
            raise ValueError("KD temperature must be greater than zero")
        if self.chunk_size < 0:
            raise ValueError("KD chunk_size cannot be negative")


@dataclass(frozen=True, kw_only=True)
class GlobalKDConfig:
    teacher_dir: Path
    student_dir: Path
    output_dir: Path
    descriptor: str | None = None
    teacher_descriptor: str | None = None
    student_descriptor: str | None = None
    force_hf: bool = True
    teacher_force_hf: bool | None = None
    student_force_hf: bool | None = None
    teacher_model_kwargs: dict[str, Any] = field(default_factory=dict)
    student_model_kwargs: dict[str, Any] = field(default_factory=dict)
    domain: Literal["auto", "llm", "vlm"] = "auto"
    trust_remote_code: bool = True
    torch_dtype: str = "bf16"
    attn_implementation: str | None = None
    tp: int = 1
    pp: int = 1
    ep: int = 1
    cp: int = 1
    dp: int = 1
    sequence_parallel: bool = False
    activation_checkpointing: bool | str = False
    pp_schedule: Literal["1f1b", "interleaved1f1b"] = "1f1b"
    save_consolidated: bool | str = False
    checkpoint_format: Literal["auto", "safetensors", "torch_save"] = "auto"
    main_ce_weight: float = 1.0
    mtp_ce_weight: float = 0.0
    main_kd_weight: float = 1.0
    mtp_kd_weight: float = 0.0
    main_kd: KDLossTermConfig = field(default_factory=KDLossTermConfig)
    mtp_kd: KDLossTermConfig = field(default_factory=KDLossTermConfig)
    # Legacy fields remain accepted by direct callers and old Hydra configs.
    ce_weight: float | None = None
    kd_weight: float | None = None
    hidden_kd_weight: float = 0.0
    temperature: float | None = None
    global_batch_size: int = 128
    local_batch_size: int = 1
    max_steps: int = 1
    checkpoint_every_steps: int | None = None
    packed_sequence_size: int = 0
    lr: float = 1.0e-5
    weight_decay: float = 0.0
    seed: int = 1111
    dataset_name: str = "rajpurkar/squad"
    dataset_split: str = "train"
    validation_enabled: bool = True
    validation_split: str = "validation"
    resume: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    freeze_policy: Literal["vision_frozen", "projector_and_language", "train_all"] = (
        "vision_frozen"
    )

    def __post_init__(self):
        if self.domain not in ("auto", "llm", "vlm"):
            raise ValueError("distillation.domain must be auto, llm, or vlm")
        if self.pp_schedule not in ("1f1b", "interleaved1f1b"):
            raise ValueError(
                "distillation.pp_schedule must be 1f1b or interleaved1f1b"
            )
        if not self.resolved_student_descriptor or not self.resolved_teacher_descriptor:
            raise ValueError("Global KD requires descriptors for both student and teacher")
        weights = self.objective_weights
        if any(value < 0 for value in weights.values()):
            raise ValueError("Global KD objective weights cannot be negative")
        if not any(weights.values()):
            raise ValueError("At least one global KD objective weight must be non-zero")
        if self.hidden_kd_weight:
            raise ValueError("hidden_kd_weight is not supported by AutoModel global KD")
        for name in ("tp", "pp", "ep", "cp", "dp"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.freeze_policy not in (
            "vision_frozen",
            "projector_and_language",
            "train_all",
        ):
            raise ValueError(f"unknown global KD freeze_policy={self.freeze_policy!r}")
        if self.data:
            data_spec = PuzzletronDataSpec.from_mapping(self.data)
            native_required = (
                data_spec.modality is Modality.MULTIMODAL
                or data_spec.layout is DataLayout.PACKED_VARLEN
            )
            if native_required and (
                self.resolved_student_force_hf or self.resolved_teacher_force_hf
            ):
                raise ValueError(
                    "multimodal or packed global KD requires force_hf=False "
                    "for student and teacher"
                )

    @property
    def resolved_student_descriptor(self) -> str | None:
        return self.student_descriptor or self.descriptor

    @property
    def resolved_teacher_descriptor(self) -> str | None:
        return self.teacher_descriptor or self.descriptor or self.student_descriptor

    @property
    def resolved_student_force_hf(self) -> bool:
        return self.force_hf if self.student_force_hf is None else self.student_force_hf

    @property
    def resolved_teacher_force_hf(self) -> bool:
        return self.force_hf if self.teacher_force_hf is None else self.teacher_force_hf

    @property
    def objective_weights(self) -> dict[str, float]:
        return {
            "main_ce": float(self.main_ce_weight if self.ce_weight is None else self.ce_weight),
            "mtp_ce": float(self.mtp_ce_weight),
            "main_kd": float(self.main_kd_weight if self.kd_weight is None else self.kd_weight),
            "mtp_kd": float(self.mtp_kd_weight),
        }

    @property
    def needs_teacher(self) -> bool:
        weights = self.objective_weights
        return weights["main_kd"] > 0 or weights["mtp_kd"] > 0

    @property
    def identity(self) -> str:
        return cache_key("global_kd", {}, self.to_dict()).value

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(self.__dict__)


@dataclass(frozen=True, kw_only=True)
class GlobalKDResult:
    kd_id: str
    output_dir: Path
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return canonicalize(self.__dict__)


def _int_value(*keys: tuple[dict[str, Any], str], default: int = 1) -> int:
    for source, key in keys:
        value = source.get(key) if isinstance(source, dict) else None
        if value is not None:
            return int(value or default)
    return default


def _loss_term(node: dict[str, Any], *, legacy_temperature: float) -> KDLossTermConfig:
    return KDLossTermConfig(
        metric=str(node.get("metric", "kld")).lower(),
        temperature=float(node.get("temperature", legacy_temperature)),
        chunk_size=int(node.get("chunk_size", 0)),
    )


def build_global_kd_config(config: dict[str, Any]) -> GlobalKDConfig:
    kd_cfg = dict(config.get("distillation") or {})
    student = dict(kd_cfg.get("student") or {})
    teacher = dict(kd_cfg.get("teacher") or {})
    objective = dict(kd_cfg.get("objective") or {})
    automodel = dict(kd_cfg.get("automodel") or {})
    parallel = dict(automodel.get("parallel") or {})
    if not parallel:
        raise ValueError("Global KD requires distillation.automodel.parallel")
    removed_axes = sorted({"tp", "cp", "pp", "ep", "dp"}.intersection(kd_cfg))
    if removed_axes:
        raise ValueError(
            "Global KD top-level parallel axes were removed; use "
            "distillation.automodel.parallel instead "
            f"(found {', '.join(removed_axes)})"
        )
    ep = _int_value((parallel, "ep"))
    dp_shard = _int_value((parallel, "dp_shard"))
    if dp_shard % ep:
        raise ValueError(
            "distillation.automodel.parallel.dp_shard must be divisible by ep; "
            f"got dp_shard={dp_shard}, ep={ep}"
        )
    # AutoModel's ``dp_size`` describes the physical FSDP mesh. EP overlays
    # its shard axis, so the recipe must retain all shard ranks instead of
    # collapsing them to the logical sample-DP degree.
    physical_dp = _int_value((parallel, "dp_replicate")) * dp_shard
    model_cfg = config.get("model") or {}
    runtime_cfg = config.get("_runtime") or {}
    exp_dir = Path((config.get("experiment") or {})["dir"])

    legacy_descriptor = (
        kd_cfg.get("descriptor")
        or model_cfg.get("descriptor_override")
        or config.get("descriptor")
        or model_cfg.get("anymodel_descriptor")
        or runtime_cfg.get("descriptor")
    )
    student_descriptor = (
        student.get("descriptor") or kd_cfg.get("student_descriptor") or legacy_descriptor
    )
    teacher_descriptor = (
        teacher.get("descriptor")
        or kd_cfg.get("teacher_descriptor")
        or legacy_descriptor
        or student_descriptor
    )
    if not student_descriptor or not teacher_descriptor:
        raise ValueError("Global KD requires student and teacher Puzzletron descriptors")

    legacy_force_hf = bool(kd_cfg.get("force_hf", model_cfg.get("force_hf", True)))
    legacy_temperature = float(kd_cfg.get("temperature", 1.0))
    main_kd_node = dict(objective.get("main_kd") or {})
    mtp_kd_node = dict(objective.get("mtp_kd") or {})
    data = dict(config.get("data") or {})
    for cache_key in ("train_token_cache_path", "validation_token_cache_path"):
        if config.get(cache_key):
            data.setdefault(cache_key, config[cache_key])

    def _weight(name: str, legacy: str | None, default: float) -> float:
        value = objective.get(name)
        if isinstance(value, dict):
            value = value.get("weight")
        if value is None and legacy is not None:
            value = kd_cfg.get(legacy)
        return float(default if value is None else value)

    return GlobalKDConfig(
        teacher_dir=Path(teacher.get("dir") or kd_cfg.get("teacher_dir") or exp_dir / "ckpts" / "teacher"),
        student_dir=Path(student.get("dir") or kd_cfg.get("student_dir") or exp_dir / "ckpts" / "solution"),
        output_dir=Path(kd_cfg.get("output_dir") or exp_dir / "ckpts" / "distilled_solution"),
        descriptor=str(legacy_descriptor) if legacy_descriptor else None,
        student_descriptor=str(student_descriptor),
        teacher_descriptor=str(teacher_descriptor),
        force_hf=legacy_force_hf,
        student_force_hf=bool(
            student.get("force_hf", kd_cfg.get("student_force_hf", legacy_force_hf))
        ),
        teacher_force_hf=bool(
            teacher.get("force_hf", kd_cfg.get("teacher_force_hf", legacy_force_hf))
        ),
        student_model_kwargs={
            **dict(kd_cfg.get("student_model_kwargs") or {}),
            **dict(student.get("model_kwargs") or {}),
        },
        teacher_model_kwargs={
            **dict(kd_cfg.get("teacher_model_kwargs") or {}),
            **dict(teacher.get("model_kwargs") or {}),
        },
        domain=str(kd_cfg.get("domain", "auto")).lower(),
        trust_remote_code=bool(kd_cfg.get("trust_remote_code", model_cfg.get("trust_remote_code", True))),
        torch_dtype=str(kd_cfg.get("torch_dtype") or model_cfg.get("torch_dtype") or "bf16"),
        attn_implementation=kd_cfg.get("attn_implementation") or model_cfg.get("attn_implementation"),
        tp=_int_value((parallel, "tp")),
        pp=_int_value((parallel, "pp")),
        ep=ep,
        cp=_int_value((parallel, "cp")),
        dp=physical_dp,
        sequence_parallel=bool(parallel.get("sequence_parallel", False)),
        activation_checkpointing=kd_cfg.get(
            "activation_checkpointing",
            automodel.get("activation_checkpointing", False),
        ),
        pp_schedule=str(parallel.get("pipeline_schedule", "1f1b")).lower(),
        save_consolidated=kd_cfg.get("save_consolidated", False),
        checkpoint_format=str(kd_cfg.get("checkpoint_format", "auto")).lower(),
        main_ce_weight=_weight("main_ce", "ce_weight", 1.0),
        mtp_ce_weight=_weight("mtp_ce", None, 0.0),
        main_kd_weight=_weight("main_kd", "kd_weight", 1.0),
        mtp_kd_weight=_weight("mtp_kd", None, 0.0),
        main_kd=_loss_term(main_kd_node, legacy_temperature=legacy_temperature),
        mtp_kd=_loss_term(mtp_kd_node, legacy_temperature=legacy_temperature),
        hidden_kd_weight=float(kd_cfg.get("hidden_kd_weight", 0.0)),
        global_batch_size=int(kd_cfg.get("global_batch_size", 128)),
        local_batch_size=int(kd_cfg.get("local_batch_size", 1)),
        max_steps=int(kd_cfg.get("max_steps", 1)),
        checkpoint_every_steps=(
            int(kd_cfg["checkpoint_every_steps"])
            if kd_cfg.get("checkpoint_every_steps") is not None
            else None
        ),
        packed_sequence_size=int(kd_cfg.get("packed_sequence_size", 0)),
        lr=float(kd_cfg.get("lr", 1.0e-5)),
        weight_decay=float(kd_cfg.get("weight_decay", 0.0)),
        seed=int(kd_cfg.get("seed", 1111)),
        dataset_name=str(kd_cfg.get("dataset_name", "rajpurkar/squad")),
        dataset_split=str(kd_cfg.get("dataset_split", "train")),
        validation_enabled=bool(kd_cfg.get("validation_enabled", True)),
        validation_split=str(kd_cfg.get("validation_split", "validation")),
        resume=bool(kd_cfg.get("resume", True)),
        metadata=dict(kd_cfg.get("metadata") or {}),
        data=data,
        freeze_policy=str(kd_cfg.get("freeze_policy", "vision_frozen")),
    )


def _resolve_domain(kd_config: GlobalKDConfig) -> Literal["llm", "vlm"]:
    if kd_config.domain != "auto":
        return kd_config.domain
    if kd_config.data:
        data_spec = PuzzletronDataSpec.from_mapping(kd_config.data)
        return "vlm" if data_spec.modality is Modality.MULTIMODAL else "llm"
    config_path = kd_config.student_dir / "config.json"
    if config_path.is_file():
        try:
            checkpoint_config = json.loads(config_path.read_text())
        except (OSError, ValueError):
            checkpoint_config = {}
        architectures = " ".join(checkpoint_config.get("architectures") or ()).lower()
        model_type = str(checkpoint_config.get("model_type", "")).lower()
        if "causallm" in architectures or model_type.endswith("_text"):
            return "llm"
        if (
            any(token in architectures for token in ("conditionalgeneration", "imagetotext", "vision"))
            or "vision_config" in checkpoint_config
        ):
            return "vlm"
    descriptor = kd_config.resolved_student_descriptor.lower()
    if descriptor.endswith("_text"):
        return "llm"
    if any(token in descriptor for token in ("vl", "vision", "image")) or descriptor in {
        "qwen3_5",
        "qwen3_6",
    }:
        return "vlm"
    return "llm"


def _model_recipe(kd_config: GlobalKDConfig, *, teacher: bool, domain: str) -> dict[str, Any]:
    descriptor = (
        kd_config.resolved_teacher_descriptor if teacher else kd_config.resolved_student_descriptor
    )
    force_hf = (
        kd_config.resolved_teacher_force_hf if teacher else kd_config.resolved_student_force_hf
    )
    model_dir = kd_config.teacher_dir if teacher else kd_config.student_dir
    target = (
        "nemo_automodel.NeMoAutoModelForImageTextToText.from_pretrained"
        if domain == "vlm"
        else "nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained"
    )
    model = {
        "_target_": target,
        "pretrained_model_name_or_path": str(model_dir),
        "anymodel_descriptor": descriptor,
        "force_hf": force_hf,
        "torch_dtype": kd_config.torch_dtype,
        "trust_remote_code": kd_config.trust_remote_code,
    }
    model.update(kd_config.teacher_model_kwargs if teacher else kd_config.student_model_kwargs)
    if kd_config.mtp_ce_weight <= 0 and kd_config.mtp_kd_weight <= 0:
        model.setdefault("num_nextn_predict_layers", 0)
    if kd_config.attn_implementation:
        model.setdefault("attn_implementation", kd_config.attn_implementation)
    return model


def _loss_recipe(term: KDLossTermConfig) -> dict[str, Any]:
    target = "KDLoss" if term.metric == "kld" else "TVDLoss"
    return {
        "_target_": f"modelopt.torch.puzzletron.distillation.loss.{target}",
        "ignore_index": -100,
        "temperature": term.temperature,
        "fp32_upcast": True,
        "chunk_size": term.chunk_size,
        "checkpoint_chunks": True,
    }


def _deep_update(destination: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(destination.get(key), dict):
            _deep_update(destination[key], value)
        else:
            destination[key] = value


def _materialize_target_keys(value: Any) -> Any:
    """Keep recipe targets inert in Hydra pipeline configs until global KD runs."""
    if isinstance(value, dict):
        return {
            ("_target_" if key == "target" else key): _materialize_target_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize_target_keys(item) for item in value]
    return value


def _configure_fla_smoke_autotune() -> None:
    """Bound FLA/Triton autotuning when explicitly requested.

    Long-context KD can leave enough memory for one correct GDN kernel launch
    but not for Triton's multi-configuration benchmark harness.  The chosen
    configuration affects performance only; all candidates implement the same
    operation.  Patch both FLA's cached autotuner and plain ``triton.autotune``
    because CP preprocessing kernels use the latter directly.
    """
    if os.environ.get("PUZZLETRON_FLA_SINGLE_CONFIG") != "1":
        return
    import sys

    import triton
    from fla.ops.utils import cache as fla_cache
    from triton.runtime.autotuner import Autotuner as TritonAutotuner

    autotuner = fla_cache.CachedAutotuner
    if getattr(autotuner, "_puzzletron_single_config", False):
        return
    original_init = autotuner.__init__
    original_run = autotuner.run

    def _single_config_init(self, fn, arg_names, configs, *args, **kwargs):
        if not configs:
            raise ValueError("FLA autotuner received no kernel configurations")
        return original_init(self, fn, arg_names, configs[:1], *args, **kwargs)

    def _single_config_run(self, *args, **kwargs):
        # Most FLA kernels are imported while the AutoModel implementation is
        # registered, before global KD gets a chance to install this smoke-run
        # policy.  Capping only __init__ therefore misses those already-created
        # autotuners.  Autotuner.run reads self.configs on every cache miss, so
        # narrow it for the duration of this invocation as well.
        configs = self.configs
        if not configs:
            raise ValueError("FLA autotuner received no kernel configurations")
        key = fla_cache.AutotuneKey.build(self.arg_names, self.keys, args, kwargs)
        self.cache.setdefault(key.autotune_key, configs[0])
        self.configs = configs[:1]
        try:
            return original_run(self, *args, **kwargs)
        finally:
            self.configs = configs

    autotuner.__init__ = _single_config_init
    autotuner.run = _single_config_run
    autotuner._puzzletron_single_config = True

    if not getattr(triton, "_puzzletron_single_config", False):
        original_autotune = triton.autotune

        def _single_triton_autotune(configs, *args, **kwargs):
            if not configs:
                raise ValueError("Triton autotuner received no kernel configurations")
            return original_autotune(configs[:1], *args, **kwargs)

        triton.autotune = _single_triton_autotune
        triton._puzzletron_single_config = True

    # Decorator patching only affects kernels imported later.  Patch the
    # Autotuner class as well so already-decorated FLA CP kernels skip the
    # benchmark path.  With one config Triton's normal run method launches the
    # kernel directly and still records ``best_config``.
    if not getattr(TritonAutotuner, "_puzzletron_single_config", False):
        original_triton_run = TritonAutotuner.run

        def _single_triton_run(self, *args, **kwargs):
            if not self.configs:
                raise ValueError("Triton autotuner received no kernel configurations")
            if len(self.configs) > 1:
                self.configs = self.configs[:1]
            return original_triton_run(self, *args, **kwargs)

        TritonAutotuner.run = _single_triton_run
        TritonAutotuner._puzzletron_single_config = True

    # Some FLA kernels may have been decorated before this policy was enabled.
    # Walk their lightweight decorator wrappers (Heuristics -> Autotuner -> JIT)
    # and narrow existing configuration lists in place.
    narrowed = 0
    for module_name, module in tuple(sys.modules.items()):
        if not module_name.startswith("fla.ops.") or module is None:
            continue
        for value in tuple(vars(module).values()):
            seen = set()
            candidate = value
            while candidate is not None and id(candidate) not in seen:
                seen.add(id(candidate))
                configs = getattr(candidate, "configs", None)
                if isinstance(configs, list) and len(configs) > 1:
                    candidate.configs = configs[:1]
                    narrowed += 1
                candidate = getattr(candidate, "fn", None)
    if narrowed:
        logger.info("Bound %d pre-existing FLA Triton autotuners to one configuration", narrowed)


def build_automodel_global_kd_recipe(kd_config: GlobalKDConfig) -> dict[str, Any]:
    """Translate Puzzletron KD settings into the current NeMo AutoModel recipe schema."""
    domain = _resolve_domain(kd_config)
    if kd_config.local_batch_size % max(1, kd_config.pp) != 0:
        raise ValueError(
            "Global KD local_batch_size must be divisible by pp so every "
            "pipeline stage receives an integral microbatch schedule"
        )
    pp_microbatch_size = max(1, kd_config.local_batch_size // max(1, kd_config.pp))
    checkpoint_format = kd_config.checkpoint_format
    if checkpoint_format == "auto":
        consolidated = str(kd_config.save_consolidated).strip().lower() not in {"false", "0", "none"}
        # AutoModel's safetensors writer always constructs Hugging Face metadata.
        # Pipeline stages own disjoint FQNs, so use plain DCP for resumable PP
        # checkpoints unless an explicit consolidated export was requested.
        checkpoint_format = "torch_save" if kd_config.pp > 1 and not consolidated else "safetensors"

    # AutoModel's scheduler starts at step one and treats max_steps as an
    # exclusive upper bound. Keep Puzzletron's public max_steps equal to the
    # requested number of optimizer updates by adding the internal sentinel.
    scheduler_max_steps = kd_config.max_steps + 1
    checkpoint_every_steps = kd_config.checkpoint_every_steps or scheduler_max_steps
    data_spec = (
        PuzzletronDataSpec.from_mapping(kd_config.data) if kd_config.data else None
    )
    packed_sequence_size = int(kd_config.packed_sequence_size)
    if data_spec is not None and data_spec.layout is DataLayout.PACKED_VARLEN:
        canonical_pack_size = int(data_spec.packing.pack_size)
        if packed_sequence_size not in (0, canonical_pack_size):
            raise ValueError(
                "distillation.packed_sequence_size conflicts with "
                f"data.packing.pack_size: {packed_sequence_size} != {canonical_pack_size}"
            )
        packed_sequence_size = canonical_pack_size

    recipe: dict[str, Any] = {
        "recipe": (
            "KnowledgeDistillationRecipeForVLM"
            if domain == "vlm"
            else "KnowledgeDistillationRecipeForNextTokenPrediction"
        ),
        "step_scheduler": {
            "global_batch_size": kd_config.global_batch_size,
            "local_batch_size": kd_config.local_batch_size,
            "max_steps": scheduler_max_steps,
            "num_epochs": scheduler_max_steps,
            "ckpt_every_steps": checkpoint_every_steps,
            "save_checkpoint_every_epoch": False,
        },
        "dist_env": {"backend": "nccl", "timeout_minutes": 60},
        "rng": {
            "_target_": "nemo_automodel.components.training.rng.StatefulRNG",
            "seed": kd_config.seed,
            "ranked": True,
        },
        "seed": kd_config.seed,
        "puzzletron_resume": kd_config.resume,
        "freeze_policy": kd_config.freeze_policy,
        "model": _model_recipe(kd_config, teacher=False, domain=domain),
        "checkpoint": {
            "enabled": True,
            "checkpoint_dir": str(kd_config.output_dir / "checkpoints"),
            "model_save_format": checkpoint_format,
            "save_consolidated": kd_config.save_consolidated,
        },
        "distributed": {
            "strategy": "fsdp2",
            "dp_size": kd_config.dp,
            "tp_size": kd_config.tp,
            "cp_size": kd_config.cp,
            "ep_size": kd_config.ep,
            "sequence_parallel": kd_config.sequence_parallel,
            "activation_checkpointing": kd_config.activation_checkpointing,
            "pp_size": kd_config.pp,
            "pipeline": {
                "pp_schedule": kd_config.pp_schedule,
                "pp_microbatch_size": pp_microbatch_size,
                "pp_batch_size": kd_config.local_batch_size,
                "scale_grads_in_schedule": False,
                "round_virtual_stages_to_pp_multiple": "up",
                "dtype": kd_config.torch_dtype,
            },
        },
        "packed_sequence": {
            "packed_sequence_size": packed_sequence_size,
            "split_across_pack": False,
        },
        "loss_fn": {
            "_target_": "modelopt.torch.puzzletron.distillation.loss.ChunkedCrossEntropy",
            "chunk_size": max(kd_config.main_kd.chunk_size, kd_config.mtp_kd.chunk_size),
            "checkpoint_chunks": True,
        },
        "objective": kd_config.objective_weights,
        "main_kd_loss_fn": _loss_recipe(kd_config.main_kd),
        "mtp_kd_loss_fn": _loss_recipe(kd_config.mtp_kd),
        # Native AutoModel KD setup still consumes these two fields. The local
        # recipe subclasses replace its normalized mixture with ``objective``.
        "kd_ratio": 0.5,
        "kd_loss_fn": _loss_recipe(kd_config.main_kd),
        "optimizer": {
            "_target_": "torch.optim.AdamW",
            "betas": [0.9, 0.999],
            "eps": 1.0e-8,
            "lr": kd_config.lr,
            "weight_decay": kd_config.weight_decay,
        },
    }
    if kd_config.needs_teacher:
        recipe["teacher_model"] = _model_recipe(kd_config, teacher=True, domain=domain)

    domain_metadata = _materialize_target_keys(dict(kd_config.metadata.get(domain) or {}))
    if domain == "llm":
        if kd_config.data:
            data_spec = PuzzletronDataSpec.from_mapping(kd_config.data)

            def _dataset(split: str, samples_key: str, cache_key: str) -> dict[str, Any]:
                sample_config = dict(kd_config.data.get(samples_key) or {})
                if data_spec.layout is not DataLayout.FIXED:
                    dataset = {
                        "_target_": (
                            "modelopt.torch.puzzletron.distillation.dataset."
                            "make_puzzletron_chat_dataset"
                        ),
                        "dataset_path": str(kd_config.data.get("path") or ""),
                        "split": split,
                        "seq_length": int(data_spec.max_sample_length),
                        "seed": kd_config.seed,
                    }
                    if data_spec.layout is DataLayout.PADDED_VARLEN:
                        dataset["num_samples"] = int(
                            sample_config.get(
                                "num_samples",
                                kd_config.max_steps * kd_config.global_batch_size,
                            )
                        )
                    return dataset
                dataset = {
                    "_target_": (
                        "modelopt.torch.puzzletron.distillation.dataset."
                        "make_puzzletron_llm_dataset"
                    ),
                    "dataset_path": str(kd_config.data.get("path") or ""),
                    "split": split,
                    "num_samples": int(
                        sample_config.get(
                            "num_samples",
                            kd_config.max_steps * kd_config.global_batch_size,
                        )
                    ),
                    "seq_length": int(data_spec.sequence_length),
                    "seed": kd_config.seed,
                }
                if kd_config.data.get(cache_key):
                    dataset["packed_token_cache_path"] = str(
                        kd_config.data[cache_key]
                    )
                return dataset

            dataloader = {
                "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                "collate_fn": (
                    "nemo_automodel.components.datasets.utils.default_collater"
                    if data_spec.layout is not DataLayout.FIXED
                    else (
                        "modelopt.torch.puzzletron.distillation.dataset."
                        "collate_puzzletron_llm_batch"
                    )
                ),
                "shuffle": False,
                "num_workers": 0,
                "pin_memory": True,
            }
            recipe.update(
                dataset=_dataset(
                    kd_config.dataset_split,
                    "calibration",
                    "train_token_cache_path",
                ),
                dataloader=dict(dataloader),
            )
            if data_spec.layout is DataLayout.PACKED_VARLEN:
                calibration = dict(kd_config.data.get("calibration") or {})
                recipe["packed_sequence"] = {
                    "packed_sequence_size": int(data_spec.packing.pack_size),
                    "packing_strategy": "neat",
                    "drop_long_samples": bool(data_spec.packing.drop_long_samples),
                    "max_packs": int(
                        calibration.get(
                            "num_samples",
                            kd_config.max_steps * kd_config.global_batch_size,
                        )
                    ),
                }
            if kd_config.validation_enabled:
                recipe.update(
                    validation_dataset=_dataset(
                        kd_config.validation_split,
                        "replacement_scoring",
                        "validation_token_cache_path",
                    ),
                    validation_dataloader=dict(dataloader),
                )
        else:
            recipe.update(
                dataset={
                    "_target_": (
                        "nemo_automodel.components.datasets.llm.squad."
                        "make_squad_dataset"
                    ),
                    "dataset_name": kd_config.dataset_name,
                    "split": kd_config.dataset_split,
                },
                dataloader={
                    "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                    "collate_fn": (
                        "nemo_automodel.components.datasets.utils.default_collater"
                    ),
                    "shuffle": False,
                },
            )
            if kd_config.validation_enabled:
                recipe.update(
                    validation_dataset={
                        "_target_": (
                            "nemo_automodel.components.datasets.llm.squad."
                            "make_squad_dataset"
                        ),
                        "dataset_name": kd_config.dataset_name,
                        "split": kd_config.validation_split,
                    },
                    validation_dataloader={
                        "_target_": (
                            "torchdata.stateful_dataloader.StatefulDataLoader"
                        ),
                        "collate_fn": (
                            "nemo_automodel.components.datasets.utils.default_collater"
                        ),
                    },
                )
        metadata_keys = ["dataset", "dataloader"]
        if kd_config.validation_enabled:
            metadata_keys.extend(("validation_dataset", "validation_dataloader"))
        for key in metadata_keys:
            if key in domain_metadata:
                recipe[key] = domain_metadata[key]
    else:
        processor = dict(domain_metadata.get("processor") or kd_config.metadata.get("processor") or {})
        processor.setdefault("_target_", "transformers.AutoProcessor.from_pretrained")
        processor.setdefault("pretrained_model_name_or_path", str(kd_config.student_dir))
        recipe["processor"] = processor
        metadata_keys = ["dataset", "dataloader", "freeze_config"]
        if kd_config.validation_enabled:
            metadata_keys.extend(("validation_dataset", "validation_dataloader"))
        for key in metadata_keys:
            if key in domain_metadata:
                recipe[key] = domain_metadata[key]
            elif key in kd_config.metadata:
                recipe[key] = kd_config.metadata[key]
        missing = [key for key in ("dataset", "dataloader") if key not in recipe]
        if kd_config.data:
            data_spec = PuzzletronDataSpec.from_mapping(kd_config.data)
            calibration = dict(kd_config.data.get("calibration") or {})
            recipe["dataset"] = {
                "_target_": (
                    "modelopt.torch.puzzletron.dataset."
                    "load_materialized_conversation_dataset"
                ),
                "path_or_dataset": str(kd_config.data.get("path")),
                "pretokenize": True,
                "truncate": False,
                "inject_fake_images": False,
                "max_length": int(data_spec.max_sample_length),
            }
            if data_spec.layout is DataLayout.PADDED_VARLEN:
                recipe["dataset"]["num_samples"] = int(
                    calibration.get(
                        "num_samples",
                        kd_config.max_steps * kd_config.global_batch_size,
                    )
                )
            recipe["dataloader"] = {
                "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
                "shuffle": False,
                "num_workers": 0,
                "pin_memory": True,
            }
            if data_spec.packing is not None:
                recipe["packed_sequence"] = {
                    "pretokenize": True,
                    "max_length": int(data_spec.max_sample_length),
                    "pack_size": int(data_spec.packing.pack_size),
                    "packing_ratio": float(data_spec.packing.packing_ratio),
                    "drop_long_samples": bool(data_spec.packing.drop_long_samples),
                    "attn_implementation": "flash_attention_2",
                    "collate_max_length": int(data_spec.packing.pack_size),
                    "max_packs": int(
                        calibration.get(
                            "num_samples",
                            kd_config.max_steps * kd_config.global_batch_size,
                        )
                    ),
                }
            missing = []
        if missing:
            raise ValueError(
                "VLM global KD requires distillation.metadata entries for " + ", ".join(missing)
            )
        recipe["freeze_config"] = {
            "freeze_vision_tower": kd_config.freeze_policy == "vision_frozen",
            "freeze_language_model": False,
        }

    # AutoModel does not inject ``local_batch_size`` for IterableDataset
    # dataloaders.  Without an explicit value StatefulDataLoader silently uses
    # one sample, while the PP schedule is built for the configured local batch.
    # Keep metadata-provided choices authoritative and fill the public KD batch
    # size only when the dataloader did not specify one itself.
    for dataloader_key in ("dataloader", "validation_dataloader"):
        dataloader_recipe = recipe.get(dataloader_key)
        if isinstance(dataloader_recipe, dict):
            dataloader_recipe.setdefault("batch_size", kd_config.local_batch_size)

    if kd_config.pp > 1:
        from ..plugins.automodel.config import inject_descriptor_pipeline_config

        inject_descriptor_pipeline_config(
            recipe,
            model_path=kd_config.student_dir,
            descriptor_name=kd_config.resolved_student_descriptor,
            trust_remote_code=kd_config.trust_remote_code,
        )
        if kd_config.needs_teacher:
            teacher_recipe = {
                "model": recipe["teacher_model"],
                "distributed": {
                    **recipe["distributed"],
                    "pipeline": dict(recipe["distributed"]["pipeline"]),
                },
            }
            teacher_recipe["distributed"]["pipeline"].pop("module_fqns_per_model_part", None)
            inject_descriptor_pipeline_config(
                teacher_recipe,
                model_path=kd_config.teacher_dir,
                descriptor_name=kd_config.resolved_teacher_descriptor,
                trust_remote_code=kd_config.trust_remote_code,
            )
            teacher_pipeline = teacher_recipe["distributed"].get("pipeline", {})
            # Only the partition is teacher-specific. Runtime/dtype fields stay
            # on the already-resolved student PipelineConfig so custom config
            # nodes cannot reintroduce unresolved string dtypes.
            recipe["teacher_pipeline"] = {
                "module_fqns_per_model_part": teacher_pipeline.get(
                    "module_fqns_per_model_part"
                )
            }

    overrides = kd_config.metadata.get("recipe_overrides")
    if overrides:
        _deep_update(recipe, dict(overrides))

    from ..plugins.automodel.config import inject_descriptor_model_kwargs

    inject_descriptor_model_kwargs(
        recipe,
        model_path=kd_config.student_dir,
        descriptor_name=kd_config.resolved_student_descriptor,
        trust_remote_code=kd_config.trust_remote_code,
        model_key="model",
    )
    if kd_config.needs_teacher:
        inject_descriptor_model_kwargs(
            recipe,
            model_path=kd_config.teacher_dir,
            descriptor_name=kd_config.resolved_teacher_descriptor,
            trust_remote_code=kd_config.trust_remote_code,
            model_key="teacher_model",
        )

    # AutoModel only infers ``pp_seq_len`` from ``dataset.seq_len``.  Custom
    # datasets commonly expose the equivalent constructor argument as
    # ``seq_length``; with CP+PP, leaving the hint unset falls back to PyTorch's
    # serial runtime shape inference, whose stage-local CP collectives can
    # deadlock.  Pipeline tensors see the CP-local sequence length.
    if kd_config.pp > 1 and domain == "llm":
        pipeline = recipe["distributed"]["pipeline"]
        if pipeline.get("pp_seq_len") is None:
            if packed_sequence_size > 0:
                pp_seq_len = packed_sequence_size
            else:
                dataset = recipe.get("dataset") or {}
                global_seq_len = dataset.get("seq_len", dataset.get("seq_length"))
                pp_seq_len = (
                    None
                    if global_seq_len is None
                    else (int(global_seq_len) + kd_config.cp - 1) // kd_config.cp
                )
            if pp_seq_len is not None:
                pipeline["pp_seq_len"] = pp_seq_len
    return recipe


def _quarantine_incomplete_global_kd_checkpoints(checkpoint_root: Path) -> None:
    """Keep AutoModel's LATEST discovery away from interrupted DCP transactions."""

    if not checkpoint_root.is_dir():
        return
    quarantine_root = checkpoint_root / "_incomplete"
    for checkpoint_path in sorted(checkpoint_root.glob("epoch_*_step_*")):
        if not checkpoint_path.is_dir():
            continue
        marker = checkpoint_path / "saving_completed"
        if marker.is_file() and ".incomplete-" not in checkpoint_path.name:
            continue
        quarantine_root.mkdir(exist_ok=True)
        quarantine = quarantine_root / (
            f"{checkpoint_path.name}.incomplete-{time.time_ns()}"
        )
        checkpoint_path.replace(quarantine)
        logger.warning(
            "Quarantined interrupted Global KD checkpoint %s -> %s",
            checkpoint_path,
            quarantine,
        )


def _reconcile_global_kd_training_log(checkpoint_root: Path) -> None:
    """Align append-only metrics with the latest durable checkpoint.

    AutoModel writes ``training.jsonl`` before it commits the corresponding
    distributed checkpoint.  An interrupted run can therefore leave valid
    JSON records beyond the last resumable optimizer step.  Replaying from
    that checkpoint must discard the uncommitted tail, and repeated attempts
    must not retain duplicate records for the same global step.
    """

    training_log = checkpoint_root / "training.jsonl"
    if not training_log.is_file():
        return

    completed_steps = []
    for checkpoint_path in checkpoint_root.glob("epoch_*_step_*"):
        if not checkpoint_path.is_dir() or not (checkpoint_path / "saving_completed").is_file():
            continue
        try:
            completed_steps.append(int(checkpoint_path.name.rsplit("_step_", 1)[1]))
        except (IndexError, ValueError):
            continue

    latest_step = max(completed_steps, default=-1)
    records_by_step: dict[int, dict[str, Any]] = {}
    for line in training_log.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        step = int(record["step"])
        if step <= latest_step:
            # The final occurrence is from the attempt that produced the
            # durable checkpoint when an earlier interrupted attempt logged
            # the same step first.
            records_by_step[step] = record

    reconciled = "".join(
        json.dumps(records_by_step[step], sort_keys=True) + "\n"
        for step in sorted(records_by_step)
    )
    temporary = training_log.with_name(f".{training_log.name}.{os.getpid()}.tmp")
    temporary.write_text(reconciled)
    os.replace(temporary, training_log)


def _preserve_inactive_mtp_weights(source: Path, consolidated: Path) -> tuple[str, ...]:
    """Restore unchanged MTP tensors omitted from a consolidated KD checkpoint."""

    source_index_path = source / "model.safetensors.index.json"
    consolidated_index_path = consolidated / "model.safetensors.index.json"
    if not source_index_path.is_file() or not consolidated_index_path.is_file():
        return ()

    source_index = json.loads(source_index_path.read_text())
    consolidated_index = json.loads(consolidated_index_path.read_text())
    source_weight_map = dict(source_index.get("weight_map") or {})
    consolidated_weight_map = dict(consolidated_index.get("weight_map") or {})
    missing = sorted(
        key
        for key in source_weight_map
        if key.startswith("mtp.") and key not in consolidated_weight_map
    )
    if not missing:
        return ()

    tensor_headers = {}
    tensor_payloads = []
    total_size = 0
    by_shard: dict[str, list[str]] = {}
    for key in missing:
        by_shard.setdefault(source_weight_map[key], []).append(key)
    for shard_name, keys in by_shard.items():
        with (source / shard_name).open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            data_offset = 8 + header_size
            for key in keys:
                tensor_header = dict(header[key])
                start, stop = tensor_header["data_offsets"]
                stream.seek(data_offset + start)
                payload = stream.read(stop - start)
                tensor_header["data_offsets"] = [total_size, total_size + len(payload)]
                tensor_headers[key] = tensor_header
                tensor_payloads.append(payload)
                total_size += len(payload)

    preserved_name = "model-puzzletron-preserved-mtp.safetensors"
    preserved_path = consolidated / preserved_name
    temporary_weights = preserved_path.with_name(
        f".{preserved_path.name}.{os.getpid()}.tmp"
    )
    output_header = json.dumps(
        {"__metadata__": {"format": "pt"}, **tensor_headers},
        separators=(",", ":"),
    ).encode()
    output_header += b" " * (-len(output_header) % 8)
    with temporary_weights.open("wb") as stream:
        stream.write(struct.pack("<Q", len(output_header)))
        stream.write(output_header)
        for payload in tensor_payloads:
            stream.write(payload)
    os.replace(temporary_weights, preserved_path)

    consolidated_weight_map.update(dict.fromkeys(missing, preserved_name))
    metadata = dict(consolidated_index.get("metadata") or {})
    if isinstance(metadata.get("total_size"), int):
        metadata["total_size"] += total_size
    consolidated_index.update(metadata=metadata, weight_map=consolidated_weight_map)
    temporary_index = consolidated_index_path.with_name(
        f".{consolidated_index_path.name}.{os.getpid()}.tmp"
    )
    temporary_index.write_text(json.dumps(consolidated_index, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_index, consolidated_index_path)
    return tuple(missing)


def _preserve_global_kd_inactive_weights(kd_config: GlobalKDConfig) -> tuple[str, ...]:
    if kd_config.mtp_ce_weight > 0 or kd_config.mtp_kd_weight > 0:
        return ()
    if str(kd_config.save_consolidated).strip().lower() in {"false", "0", "none"}:
        return ()

    preserved = set()
    for config_path in kd_config.output_dir.glob(
        "checkpoints/epoch_*_step_*/model/consolidated/config.json"
    ):
        preserved.update(
            _preserve_inactive_mtp_weights(kd_config.student_dir, config_path.parent)
        )
    return tuple(sorted(preserved))


def run_automodel_global_kd(kd_config: GlobalKDConfig) -> dict[str, Any]:
    _configure_fla_smoke_autotune()
    try:
        from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

        from ..plugins.automodel.patch import apply_patch
        from .global_kd_recipe import (
            KnowledgeDistillationRecipeForNextTokenPrediction,
            KnowledgeDistillationRecipeForVLM,
            install_pp_checkpoint_state_dict_support,
        )
    except ImportError as exc:
        raise NotImplementedError(
            "Global KD requires NeMo AutoModel and its training dependencies. "
            f"Could not import them: {exc}"
        ) from exc

    kd_config.output_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = kd_config.output_dir / "global_kd_recipe.yaml"
    recipe = build_automodel_global_kd_recipe(kd_config)
    recipe_payload = json.dumps(recipe, indent=2, sort_keys=True) + "\n"
    # torchrun enters this function before AutoModel initializes its process
    # group.  Publish the shared recipe atomically from global rank 0 and use
    # the expected payload itself as the readiness condition; this also makes
    # stale output directories safe to resume.
    global_rank = int(os.environ.get("RANK", "0"))
    if global_rank == 0:
        checkpoint_root = kd_config.output_dir / "checkpoints"
        _quarantine_incomplete_global_kd_checkpoints(checkpoint_root)
        _reconcile_global_kd_training_log(checkpoint_root)
        temporary_path = recipe_path.with_name(
            f".{recipe_path.name}.{os.getpid()}.tmp"
        )
        temporary_path.write_text(recipe_payload)
        os.replace(temporary_path, recipe_path)
    deadline = time.monotonic() + 120
    while True:
        try:
            if recipe_path.read_text() == recipe_payload:
                break
        except FileNotFoundError:
            pass
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for global KD recipe publication: {recipe_path}")
        time.sleep(0.1)

    apply_patch()
    cfg = parse_args_and_load_config(str(recipe_path), argv=[])
    recipe_cls = (
        KnowledgeDistillationRecipeForVLM
        if _resolve_domain(kd_config) == "vlm"
        else KnowledgeDistillationRecipeForNextTokenPrediction
    )
    if kd_config.pp > 1:
        install_pp_checkpoint_state_dict_support()
    trainer = recipe_cls(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()
    import torch.distributed as torch_dist

    if global_rank == 0:
        preserved_mtp = _preserve_global_kd_inactive_weights(kd_config)
        if preserved_mtp:
            logger.info(
                "Preserved %d inactive MTP tensors in consolidated global-KD checkpoints",
                len(preserved_mtp),
            )
    if torch_dist.is_available() and torch_dist.is_initialized():
        torch_dist.barrier()
    observability = (
        trainer.observability_metadata()
        if hasattr(trainer, "observability_metadata")
        else {}
    )
    local_terms = {}
    for name, values in getattr(trainer, "_objective_buffers", {}).items():
        local_terms[name] = [float(value.detach().float().cpu()) for value in values]
    gathered_terms = [local_terms]

    if torch_dist.is_available() and torch_dist.is_initialized():
        gathered_terms = [None] * torch_dist.get_world_size()
        try:
            torch_dist.all_gather_object(gathered_terms, local_terms)
        except RuntimeError as error:
            raise RuntimeError("Global-KD objective metric gathering failed") from error
    objective_metrics = {}
    for name in kd_config.objective_weights:
        values = [
            value
            for rank_terms in gathered_terms
            for value in (rank_terms or {}).get(name, [])
        ]
        objective_metrics[name] = sum(values) / len(values) if values else None
    # The weighted recipe flushes its per-step buffers into training.jsonl so
    # they do not accumulate across optimizer steps.  Prefer the final logged
    # values for the stage manifest; they are the metrics associated with the
    # checkpoint we just saved and remain available after the buffers clear.
    training_log = kd_config.output_dir / "checkpoints" / "training.jsonl"
    training_records = []
    latest_record = None
    if training_log.is_file():
        for line in training_log.read_text().splitlines():
            if line.strip():
                latest_record = json.loads(line)
                training_records.append(latest_record)
        if latest_record is not None:
            for name in kd_config.objective_weights:
                value = latest_record.get(name)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    objective_metrics[name] = float(value)
    loss_values = [
        float(record[key])
        for record in training_records
        for key in ("loss", "train_loss")
        if isinstance(record.get(key), (int, float)) and math.isfinite(float(record[key]))
    ]
    trend = None
    if len(loss_values) >= 8:
        trend = {
            "first_four_median": median(loss_values[:4]),
            "last_four_median": median(loss_values[-4:]),
            "decreased": median(loss_values[-4:]) < median(loss_values[:4]),
        }
    if hasattr(trainer, "close_observability"):
        trainer.close_observability()
    if kd_config.max_steps >= 8 and (trend is None or not trend["decreased"]):
        raise RuntimeError(f"global KD acceptance loss did not decrease: {trend}")
    if kd_config.freeze_policy == "train_all":
        missing_gradient_groups = [
            group
            for group in ("vision", "projector", "language", "mtp")
            if not isinstance((latest_record or {}).get(f"gradient_norm_{group}"), (int, float))
            or float((latest_record or {}).get(f"gradient_norm_{group}", 0.0)) <= 0
        ]
        if missing_gradient_groups:
            raise RuntimeError(
                "global KD train_all produced no gradient for required groups: "
                f"{missing_gradient_groups}; latest={latest_record}"
            )
    if _resolve_domain(kd_config) == "vlm" and int(observability.get("vision_forward_count", 0)) <= 0:
        raise RuntimeError(f"global KD observed no ViT forward: {observability}")
    return {
        "recipe_config": str(recipe_path),
        "checkpoint_dir": str(kd_config.output_dir / "checkpoints"),
        "domain": _resolve_domain(kd_config),
        "objective": kd_config.objective_weights,
        "objective_metrics": objective_metrics,
        "loss_history": loss_values,
        "loss_trend": trend,
        "observability": observability,
        "latest_training_metrics": latest_record,
    }


def run_global_kd(kd_config: GlobalKDConfig, recipe_runner=None) -> GlobalKDResult:
    recipe_runner = recipe_runner or run_automodel_global_kd
    metrics = recipe_runner(kd_config)
    return GlobalKDResult(
        kd_id=kd_config.identity,
        output_dir=kd_config.output_dir,
        metrics=dict(metrics or {}),
    )
