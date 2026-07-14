# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch Puzzletron local distillation on NeMo AutoModel."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ...bypass_distillation.bypass_utils import (
    bypass_run_is_complete,
    get_bypass_run_identity,
    mark_bypass_run_completed,
    set_experiment_dir,
    set_experiment_id,
    update_bypass_checkpoint_state,
)
from ...bypass_distillation.checkpointing import (
    find_latest_completed_checkpoint,
    publish_elastic_checkpoint,
    realize_bypass_checkpoints,
    require_distributed_path_consensus,
)
from ...tools.logger import mprint
from .load import validate_force_hf_ep
from .local_kd_config import build_local_kd_recipe_config, prepare_local_kd_training_budget
from .patch import apply_patch

__all__ = ["launch_local_distillation_automodel"]


def _distributed_probe_digest(identity_payload: str) -> str:
    """Return one probe identity even when runtime object reprs differ by rank."""

    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:8]
    if not torch.distributed.is_initialized():
        return digest
    values = [digest if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return str(values[0])


def _distributed_source_identity(identity_payload: str) -> str:
    """Return the rank-zero checkpoint identity on every distributed rank."""

    digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
    if not torch.distributed.is_initialized():
        return digest
    values = [digest if torch.distributed.get_rank() == 0 else None]
    torch.distributed.broadcast_object_list(values, src=0)
    return str(values[0])


_OVERFIT_PROBE_MODES = ("diverse_resampled", "smallest_fixed")


def _overfit_probe_modes(overfit) -> tuple[str, ...]:
    configured = overfit.get("modes", None)
    if configured is None:
        selection = str(overfit.get("selection", "smallest"))
        if selection != "smallest":
            raise ValueError(
                f"legacy bypass.overfit.selection must be 'smallest', got {selection!r}"
            )
        return ("smallest_fixed",)
    modes = tuple(str(mode) for mode in configured)
    if not modes:
        raise ValueError("bypass.overfit.modes cannot be empty")
    unknown = sorted(set(modes) - set(_OVERFIT_PROBE_MODES))
    if unknown:
        raise ValueError(
            f"Unknown bypass.overfit modes {unknown}; expected values from "
            f"{list(_OVERFIT_PROBE_MODES)}"
        )
    if len(set(modes)) != len(modes):
        raise ValueError(f"bypass.overfit.modes contains duplicates: {modes}")
    return modes


def _selected_overfit_probe_modes(overfit, worker_mode: str | None) -> tuple[str, ...]:
    """Select one configured probe mode for an independent worker, if requested."""

    configured = _overfit_probe_modes(overfit)
    if not worker_mode:
        return configured
    worker_mode = str(worker_mode)
    if worker_mode not in configured:
        raise ValueError(
            f"Bypass-sanity worker mode {worker_mode!r} is not configured in "
            f"bypass.overfit.modes={list(configured)}"
        )
    return (worker_mode,)


def _overfit_probe_config(cfg, mode: str = "smallest_fixed"):
    """Return an isolated, fixed-batch probe without mutating the main run.

    The probe deliberately shares the teacher, elastic search space, and
    optimizer policy with the real run.  Its distinct experiment identity and
    disabled publication prevent its checkpoint from becoming the scoring
    parent or resume source for the subsequent nested run.
    """

    # Runtime configs contain already-instantiated descriptor/mixin objects.
    # Round-tripping through a primitive OmegaConf container rejects those
    # values, while a true deep copy preserves them and all interpolation
    # metadata without mutating the production run.
    if mode not in _OVERFIT_PROBE_MODES:
        raise ValueError(
            f"Unknown bypass overfit probe mode {mode!r}; expected one of "
            f"{list(_OVERFIT_PROBE_MODES)}"
        )
    probe = copy.deepcopy(cfg)
    overfit = probe.bypass.get("overfit", {}) or {}
    repetitions = int(overfit.get("repetitions", 32) or 32)
    if repetitions < 8:
        raise ValueError(
            "bypass.overfit.repetitions must be at least 8 so the first/last "
            "loss-median acceptance gate is meaningful"
        )
    probe.bypass.overfit_probe_mode = mode
    probe.bypass.single_batch_overfit_resample_structure = mode == "diverse_resampled"
    probe.bypass.overfit_trend_window = int(
        overfit.get(
            "diverse_trend_window" if mode == "diverse_resampled" else "fixed_trend_window",
            8 if mode == "diverse_resampled" else 4,
        )
    )
    teacher_identity = get_bypass_run_identity(probe).get("teacher") or {}
    probe.bypass.overfit_source_checkpoint_identity = _distributed_source_identity(
        json.dumps(
            teacher_identity,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )
    identity_payload = json.dumps(
        {
            "config": OmegaConf.to_container(probe, resolve=True),
            "mode": mode,
        },
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )
    digest = _distributed_probe_digest(identity_payload)
    base_id = str(probe.bypass.get("experiment_id", None) or "bypass")
    probe.bypass.experiment_id = f"{base_id}_overfit_{mode}_{digest}"
    probe.bypass.experiment_dir = None
    probe.bypass.single_batch_overfit = True
    probe.bypass.single_batch_overfit_steps = repetitions
    probe.bypass.step_num = 1
    probe.bypass.iter_num = 0
    probe.bypass.token_count = 0
    probe.bypass.elastic_fixed_selection = (
        "smallest" if mode == "smallest_fixed" else None
    )
    probe.bypass.training.max_steps = repetitions
    for training_key in (
        "learning_rate",
        "decay_lr",
        "grad_clip",
        "weight_decay",
        "grad_accumulation_steps",
    ):
        if overfit.get(training_key, None) is not None:
            probe.bypass.training[training_key] = overfit[training_key]
    probe.bypass.publish_elastic_checkpoint = False
    probe.bypass.find_last_ckpt_for_resume = False
    probe.bypass.resume_checkpoint_path = None
    probe.bypass.overfit.enabled = False
    return probe


def _barrier() -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def _require_distributed_path_consensus(path: str | Path, purpose: str) -> None:
    require_distributed_path_consensus(path, purpose)


def _broadcast_run_location(cfg) -> None:
    """Compute the bypass run location once and install it on every rank."""

    if not torch.distributed.is_initialized():
        set_experiment_id(cfg)
        set_experiment_dir(cfg)
    else:
        location = [None]
        if torch.distributed.get_rank() == 0:
            set_experiment_id(cfg)
            set_experiment_dir(cfg)
            location[0] = (
                str(cfg.bypass.experiment_id),
                str(cfg.bypass.experiment_dir),
            )
        torch.distributed.broadcast_object_list(location, src=0)
        experiment_id, experiment_dir = location[0]
        cfg.bypass.experiment_id = experiment_id
        cfg.bypass.experiment_dir = experiment_dir
    _require_distributed_path_consensus(
        cfg.bypass.experiment_dir,
        "experiment root",
    )


def _should_publish_elastic_checkpoint(cfg) -> bool:
    """Return whether this run may replace the campaign's elastic parent alias."""

    return bool(cfg.bypass.get("elastic", False)) and bool(
        cfg.bypass.get("publish_elastic_checkpoint", True)
    )


def _resume_path(cfg) -> str | None:
    explicit = cfg.bypass.get("resume_checkpoint_path", None)
    if explicit:
        return str(explicit)
    if not bool(cfg.bypass.get("find_last_ckpt_for_resume", False)):
        return None
    latest = find_latest_completed_checkpoint(cfg.bypass.experiment_dir)
    return str(latest) if latest is not None else None


def _restore_resume_counters(cfg, resume_path: str | None) -> None:
    if resume_path is None:
        return
    args_path = Path(resume_path) / "args.json"
    if not args_path.exists():
        return
    saved = json.loads(args_path.read_text())
    saved_step = int(saved.get("step_num", 0) or 0)
    saved_iter = int(saved.get("iter_num", 0) or 0)
    cfg.bypass.step_num = saved_step + 1
    cfg.bypass.iter_num = saved_iter + 1
    cfg.bypass.token_count = int(saved.get("token_count", 0) or 0)


def _publish_checkpoint(cfg, checkpoint_path: Path, metrics: dict) -> None:
    """Publish an AutoModel checkpoint through the existing bypass manifest contract."""
    _require_distributed_path_consensus(checkpoint_path, "final checkpoint publication")
    final_alias = (
        Path(cfg.bypass.experiment_dir)
        / f"final-step-{int(cfg.bypass.training.max_steps):06d}-ckpt"
    )
    if torch.distributed.get_rank() == 0:
        (checkpoint_path / "args.json").write_text(
            json.dumps(OmegaConf.to_container(cfg.bypass, resolve=True), indent=2, default=str)
            + "\n"
        )
        (checkpoint_path / "bypass_config.json").write_text(
            json.dumps(
                {
                    "backend": "automodel",
                    "keys_to_learn": cfg.bypass.model_factory.keys_to_learn,
                },
                indent=2,
                default=str,
            )
            + "\n"
        )
        (checkpoint_path / "automodel_local_kd_metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n"
        )
        update_bypass_checkpoint_state(cfg, checkpoint_path, "final")
        (checkpoint_path / "saving_completed").touch()
        if final_alias.exists() or final_alias.is_symlink():
            final_alias.unlink()
        final_alias.symlink_to(checkpoint_path.resolve(), target_is_directory=True)
    _barrier()

    if torch.distributed.get_rank() == 0:
        realized, symlink = realize_bypass_checkpoints(cfg)
        mark_bypass_run_completed(cfg, realized, symlink)
    _barrier()

    if _should_publish_elastic_checkpoint(cfg):
        if torch.distributed.get_rank() == 0:
            publish_elastic_checkpoint(cfg)
            coverage = metrics.get("elastic_coverage", {})
            coverage_path = (
                Path(cfg.puzzle_dir) / "artifacts" / "bypass" / "nested_axis_coverage.json"
            )
            coverage_path.parent.mkdir(parents=True, exist_ok=True)
            coverage_path.write_text(
                json.dumps(coverage, indent=2, sort_keys=True, default=str) + "\n"
            )
        _barrier()


def _run_one(cfg) -> None:
    _broadcast_run_location(cfg)
    _barrier()
    complete = bypass_run_is_complete(cfg) if torch.distributed.get_rank() == 0 else None
    values = [complete]
    torch.distributed.broadcast_object_list(values, src=0)
    if values[0]:
        mprint(f"AutoModel bypass run {cfg.bypass.experiment_id} is already complete; skipping")
        return

    resume_path = _resume_path(cfg)
    _restore_resume_counters(cfg, resume_path)
    prepare_local_kd_training_budget(cfg)
    recipe_dict = build_local_kd_recipe_config(cfg)
    distributed = recipe_dict.get("distributed", {})
    validate_force_hf_ep(
        bool(recipe_dict["model"].get("force_hf", True)),
        int(distributed.get("ep_size", 1) or 1),
    )

    apply_patch()
    from nemo_automodel.components.config.loader import ConfigNode

    from .local_kd_recipe import AutoModelLocalDistillationRecipe

    mprint(
        "[bypass/automodel] backend ACTIVE | "
        f"experiment={cfg.bypass.experiment_id} force_hf={recipe_dict['model']['force_hf']} "
        f"tp={distributed.get('tp_size', 1)} pp={distributed.get('pp_size', 1)} "
        f"cp={distributed.get('cp_size', 1)} ep={distributed.get('ep_size', 1)}"
    )
    recipe = AutoModelLocalDistillationRecipe(
        ConfigNode(recipe_dict),
        hydra_cfg=cfg,
        resume_path=resume_path,
    )
    try:
        recipe.setup()
        checkpoint_path, metrics = recipe.run_local_distillation()
        _publish_checkpoint(cfg, checkpoint_path, metrics)
    finally:
        recipe.close()


def _run_with_optional_overfit(cfg) -> None:
    overfit = cfg.bypass.get("overfit", {}) or {}
    if bool(overfit.get("enabled", False)):
        modes = _selected_overfit_probe_modes(
            overfit,
            os.environ.get("PUZZLETRON_BYPASS_SANITY_WORKER_MODE"),
        )
        for index, mode in enumerate(modes, start=1):
            mprint(
                "[bypass/automodel] running fixed-batch overfit acceptance probe "
                f"mode={mode} ({index}/{len(modes)}) for "
                f"{int(overfit.get('repetitions', 32) or 32)} steps"
            )
            _run_one(_overfit_probe_config(cfg, mode))
        if bool(overfit.get("only", False)):
            return
    _run_one(cfg)


def launch_local_distillation_automodel(
    hydra_cfg,
    num_nodes: int = 1,
    node_index: int = 0,
) -> None:
    """Run standard or elastic local KD using AutoModel on the full torchrun world."""
    if node_index != 0 and not hydra_cfg.bypass.get("configs", None):
        mprint(
            "AutoModel local KD consumes the complete distributed world; "
            f"ignoring sweep node_index={node_index}/{num_nodes}"
        )

    configs = list(hydra_cfg.bypass.get("configs", None) or [])
    if not configs:
        _run_with_optional_overfit(hydra_cfg)
        return
    if bool(hydra_cfg.bypass.get("elastic", False)):
        raise ValueError("elastic AutoModel local KD is incompatible with bypass.configs")

    base_overrides = OmegaConf.to_container(
        hydra_cfg.bypass.model.model_config_overrides,
        resolve=True,
    )
    base_keys = hydra_cfg.bypass.model_factory.keys_to_learn
    for index, override in enumerate(configs):
        hydra_cfg.bypass.model.model_config_overrides = OmegaConf.create(base_overrides)
        hydra_cfg.bypass.model_factory.keys_to_learn = base_keys
        if "model_config_overrides" in override:
            hydra_cfg.bypass.model.model_config_overrides = override.model_config_overrides
        if "keys_to_learn" in override:
            hydra_cfg.bypass.model_factory.keys_to_learn = override.keys_to_learn
        hydra_cfg.bypass.experiment_id = None
        hydra_cfg.bypass.experiment_dir = None
        mprint(f"[bypass/automodel] sweep {index + 1}/{len(configs)}")
        _run_with_optional_overfit(hydra_cfg)
