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

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from omegaconf import DictConfig, OmegaConf

from .anymodel.capabilities import CapabilityValidationError, validate_capabilities
from .anymodel.registry import (
    DescriptorResolution,
    resolve_descriptor_by_name,
    resolve_descriptor_from_pretrained,
)
from .manifest import StageManifest, write_stage_manifest
from .pipeline_config import canonical_stage_name, normalize_pipeline_config

__all__ = ["STAGES", "StageResult", "normalize_config", "run_stage"]


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    manifest_path: Path
    message: str
    skip_reason: str | None = None


# Import after StageResult: stage handlers import this public result type.
from .stages.graph import StageSkipReason, stage_ids, stage_is_enabled

STAGES = stage_ids()


def normalize_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        config = OmegaConf.to_container(config, resolve=True)
    return normalize_pipeline_config(config)


def _get_nested(config: dict[str, Any], path: tuple[str, ...], default: Any = None) -> Any:
    value: Any = config
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _experiment_dir(config: dict[str, Any]) -> Path:
    exp_dir = _get_nested(config, ("experiment", "dir"))
    if exp_dir is None:
        raise ValueError("Puzzletron config must define experiment.dir")
    return Path(exp_dir)


def _enabled_axes(config: dict[str, Any]) -> list[str]:
    axes = _get_nested(config, ("search_space", "axes"), {})
    if not isinstance(axes, dict):
        return []
    enabled = []
    for name, axis_cfg in axes.items():
        if not isinstance(axis_cfg, dict):
            continue
        if bool(axis_cfg.get("enabled", False)):
            enabled.append(name)
    return enabled


def _resolve_capabilities(config: dict[str, Any]) -> DescriptorResolution | None:
    model_cfg = config.get("model") or {}
    source = model_cfg.get("source")
    descriptor_override = model_cfg.get("descriptor_override")
    # Post-conversion stages must resolve against the converted local config.
    # Family capabilities can depend on architecture fields (for example MLA
    # latent ranks), which are unavailable when resolving an override by name.
    teacher = config.get("teacher_dir") or _get_nested(config, ("convert", "teacher_dir"))
    if teacher and Path(str(teacher)).exists():
        return resolve_descriptor_from_pretrained(
            str(teacher),
            trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
            descriptor_override=descriptor_override,
        )
    if descriptor_override and (not source or not Path(str(source)).exists()):
        return resolve_descriptor_by_name(descriptor_override)
    if not source:
        raise ValueError("Puzzletron config must define model.source or model.descriptor_override")
    return resolve_descriptor_from_pretrained(
        source,
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=descriptor_override,
    )


def _preflight(
    config: dict[str, Any],
    resolution: DescriptorResolution | None,
    *,
    stage: str,
) -> None:
    if resolution is None:
        return
    if stage == "convert":
        return
    stage_sections = {
        "width_importance": ("pruning",),
        "sort": ("pruning",),
        "depth_importance": ("depth_importance", "depth", "replacement_scoring"),
        "sort_sanity": ("sort_sanity", "replacement_scoring"),
        "width_sanity": ("width_sanity", "replacement_scoring"),
        "bypass_sanity": ("bypass",),
        "bypass": ("bypass",),
        "replacement_scoring": ("replacement_scoring", "scoring"),
        "zero_shot_evaluation": ("replacement_scoring", "scoring"),
        "post_distillation_evaluation": ("replacement_scoring", "scoring"),
    }
    parallel_cfg = {}
    for section_name in stage_sections.get(stage, (stage,)):
        section = config.get(section_name) or {}
        candidate = (section.get("automodel") or {}).get("parallel") or {}
        if candidate:
            parallel_cfg = candidate
            break
    model_cfg = config.get("model") or {}
    library_cfg = config.get("library") or {}
    runtime_stats_cfg = (config.get("calc_subblock_stats") or {}).get("runtime_stats") or {}
    capability_validation = config.get("capability_validation") or {}
    require_vllm = (
        bool((config.get("vllm_stats") or {}).get("enabled", False))
        or bool((library_cfg.get("vllm") or {}).get("enabled", False))
        or (
            bool(runtime_stats_cfg.get("enabled", False))
            and str(runtime_stats_cfg.get("backend", "")).lower() == "vllm"
        )
    )
    validate_capabilities(
        resolution.capabilities,
        enabled_axes=_enabled_axes(config),
        force_hf=bool(model_cfg.get("force_hf", True)),
        ep=int(parallel_cfg.get("ep", 1) or 1),
        require_vllm=require_vllm,
        require_complete_pipeline=bool(
            capability_validation.get("require_complete_pipeline", False)
        ),
    )


def _manifest_path(config: dict[str, Any], stage: str) -> Path:
    return _experiment_dir(config) / "manifests" / f"{stage}.json"


def _skip_stage(
    config: dict[str, Any],
    stage: str,
    manifest: StageManifest,
    *,
    reason: StageSkipReason,
    message: str,
) -> StageResult:
    manifest.complete(
        outputs={},
        status="skipped",
        skip_reason=reason,
    )
    manifest_path = _manifest_path(config, stage)
    write_stage_manifest(manifest_path, manifest)
    return StageResult(
        stage=stage,
        status="skipped",
        manifest_path=manifest_path,
        message=message,
        skip_reason=reason.value,
    )


def run_stage(
    config: DictConfig | dict[str, Any],
    stage: str,
    *,
    handlers: dict[str, Callable[[dict[str, Any], StageManifest], StageResult]] | None = None,
) -> StageResult:
    """Run a Puzzletron stage through the manifest and capability boundary.

    Disabled stages emit a typed skip. Every enabled stage must have a handler
    and propagate implementation failures to the caller.
    """
    stage = canonical_stage_name(stage)
    if stage not in STAGES:
        raise ValueError(f"Unknown Puzzletron stage '{stage}'. Expected one of {STAGES}")

    cfg = normalize_config(config)
    if not stage_is_enabled(stage, cfg):
        return _skip_stage(
            cfg,
            stage,
            StageManifest(stage=stage, inputs={"config": cfg}, config=cfg),
            reason=StageSkipReason.DISABLED,
            message=f"Stage '{stage}' is disabled by configuration.",
        )
    # These reports consume existing JSON artifacts and intentionally do not import a model,
    # initialize distributed state, or require the original checkpoint to remain accessible.
    artifact_only_stages = {"slicing_sanity"}
    resolution = None if stage in artifact_only_stages else _resolve_capabilities(cfg)
    try:
        _preflight(cfg, resolution, stage=stage)
    except CapabilityValidationError:
        raise

    manifest = StageManifest(
        stage=stage,
        inputs={
            "config": cfg,
            "descriptor_resolution": resolution.to_dict() if resolution else None,
        },
        config=cfg,
        capability_snapshot=resolution.capabilities.to_dict() if resolution else None,
    )
    runtime_cfg = copy.deepcopy(cfg)
    if resolution is not None:
        runtime = dict(runtime_cfg.get("_runtime") or {})
        runtime.update(
            descriptor=resolution.name,
            descriptor_reason=resolution.reason,
            descriptor_confidence=resolution.confidence,
        )
        runtime_cfg["_runtime"] = runtime

    handler_map = handlers
    if handler_map is None:
        from .stages import DEFAULT_HANDLERS

        handler_map = DEFAULT_HANDLERS

    handler = handler_map.get(stage)
    if handler is None:
        raise RuntimeError(f"enabled stage {stage!r} has no registered handler")
    return handler(runtime_cfg, manifest)
