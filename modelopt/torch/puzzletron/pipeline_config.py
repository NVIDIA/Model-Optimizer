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

from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from .dataset.config import PuzzletronDataSpec
from .granularity import resolve_granularity
from .tools.hydra_utils import initialize_hydra_config_for_dir, register_hydra_resolvers

__all__ = [
    "STAGE_ALIASES",
    "adapt_runtime_hydra_config",
    "canonical_stage_name",
    "load_runtime_hydra_config",
    "normalize_pipeline_config",
    "pipeline_config_from_path",
]


STAGE_ALIASES = {
    "tokenize-data": "tokenize_data",
    "width-importance": "width_importance",
    "sort-sanity": "sort_sanity",
    "width-sanity": "width_sanity",
    "slicing-sanity": "slicing_sanity",
    "bypass-sanity": "bypass_sanity",
    "build-library": "build_library",
    "vllm-stats": "vllm_stats",
    "replacement-scoring": "replacement_scoring",
    "zero-shot-evaluation": "zero_shot_evaluation",
    "global-distillation-sanity": "global_distillation_sanity",
    "global-distillation": "global_distillation",
    "post-distillation-evaluation": "post_distillation_evaluation",
}

_RUNTIME_SECTION_ALIASES = {
    "build_library": "build_replacement_library",
    "vllm_stats": "calc_subblock_stats",
    "depth_importance": "depth",
    "replacement_scoring": "scoring",
    "sort_sanity": "sort_equivalence",
    "width_sanity": "activation_diagnostic",
    "bypass_sanity": "bypass_overfit",
    "zero_shot_evaluation": "evaluation",
    "global_distillation_sanity": "distillation_overfit",
    "global_distillation": "distillation",
    "post_distillation_evaluation": "post_kd_evaluation",
}

_AXIS_ALIASES = {
    "ffn_intermediate": "ffn_intermediate",
    "intermediate_size": "ffn_intermediate",
    "num_query_heads": "query_heads",
    "query_heads": "query_heads",
    "q_heads_per_group": "q_heads_per_group",
    "query_heads_per_group": "q_heads_per_group",
    "num_kv_heads": "kv_heads",
    "kv_heads": "kv_heads",
    "kv_groups": "kv_groups",
    "qk_head_dim": "qk_head_dim",
    "v_head_dim": "v_head_dim",
    "num_experts": "moe_experts",
    "moe_experts": "moe_experts",
    "expert_intermediate": "moe_expert_intermediate",
    "expert_intermediate_size": "moe_expert_intermediate",
    "shared_expert_intermediate": "moe_shared_expert_intermediate",
    "shared_expert_intermediate_size": "moe_shared_expert_intermediate",
    "latent_dim": "moe_latent_dim",
    "top_k": "moe_top_k",
    "mamba_num_heads": "mamba_heads",
    "mamba_heads": "mamba_heads",
    "mamba_head_dim": "mamba_head_dim",
    "mamba_state_dim": "mamba_state_dim",
    "ssm_state_size": "mamba_state_dim",
    "sliding_window_size": "sliding_window_size",
}


def canonical_stage_name(stage: str) -> str:
    return STAGE_ALIASES.get(stage, stage)


def _config_root_and_name(path: Path) -> tuple[Path, str]:
    """Return the Hydra search root and slash-separated config name for ``path``.

    Clean Puzzletron configs are hierarchical: ``base.yaml`` lives at the tree root
    and experiment entrypoints live below ``families/<family>/<model>/``.  Hydra
    absolute defaults such as ``/base`` only work when the search root is the clean
    tree root, not the experiment file's parent directory.
    """
    for parent in (path.parent, *path.parent.parents):
        if (parent / "base.yaml").is_file() and (parent / "families").is_dir():
            return parent, path.relative_to(parent).with_suffix("").as_posix()
    return path.parent, path.stem


def _to_plain(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return deepcopy(dict(config))


def _deep_update_missing(target: dict[str, Any], key: str, value: Any) -> None:
    if value is not None and key not in target:
        target[key] = value


def _normalize_positive_count(section: dict[str, Any], key: str, path: str) -> None:
    """Normalize a downstream candidate count at the config boundary."""
    value = section.get(key, 1)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path}.{key} must be a positive integer")
    section[key] = value


def _axis_is_enabled(data: dict[str, Any]) -> bool:
    if bool(data.get("enabled", False)):
        return True
    for key in ("values", "sizes", "ratios", "budget_grid"):
        value = data.get(key)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return True
    return data.get("min") is not None or data.get("max") is not None


def _normalize_search_space(raw: dict[str, Any]) -> dict[str, Any]:
    search = deepcopy(dict(raw.get("search_space") or {}))
    axes = dict(search.get("axes") or {})
    for raw_axis, raw_cfg in list(search.items()):
        if raw_axis in {"axes", "layer_filter", "subblock_filter"}:
            continue
        if not isinstance(raw_cfg, dict):
            continue
        axis_id = _AXIS_ALIASES.get(raw_axis, raw_axis)
        axis_cfg = dict(raw_cfg)
        axis_cfg.setdefault("enabled", _axis_is_enabled(axis_cfg))
        if "ratios" in axis_cfg and "values" not in axis_cfg:
            axis_cfg["values"] = axis_cfg["ratios"]
            axis_cfg["value_kind"] = "ratio"
        if "sizes" in axis_cfg and "values" not in axis_cfg:
            axis_cfg["values"] = axis_cfg["sizes"]
            axis_cfg["value_kind"] = "size"
        axes.setdefault(axis_id, axis_cfg)
    search["axes"] = axes
    return search


def normalize_pipeline_config(config: DictConfig | dict[str, Any]) -> dict[str, Any]:
    """Return the strict Puzzletron stage-runner config shape.

    The heavy implementations still consume the Hydra runtime config. This
    normalizer is the boundary where older experiment YAMLs and the new
    `experiment/model/search_space` shape are folded into one manifest-friendly
    dictionary.
    """
    cfg = _to_plain(config)
    if "parallel" in cfg:
        raise ValueError(
            "top-level parallel was removed; configure automodel.parallel on each "
            "model-loading pipeline stage"
        )
    data = dict(cfg.get("data") or {})
    legacy_varlen_paths = [
        f"{section}.varlen"
        for section in (
            "data",
            "pruning",
            "replacement_scoring",
            "realize_model",
            "zero_shot_evaluation",
        )
        if isinstance(cfg.get(section), dict) and "varlen" in cfg[section]
    ]
    if legacy_varlen_paths:
        raise ValueError(
            f"{', '.join(legacy_varlen_paths)} is obsolete; choose data.layout=fixed, "
            "padded_varlen, or packed_varlen and configure data.packing for packed input"
        )
    data_spec = None
    if data:
        data_spec = PuzzletronDataSpec.from_mapping(data)
    experiment = dict(cfg.get("experiment") or {})
    model = dict(cfg.get("model") or {})
    convert = dict(cfg.get("convert") or {})
    library = dict(cfg.get("library") or {})
    width_sanity = dict(cfg.get("width_sanity") or {})
    aiperf = dict(cfg.get("aiperf") or {})
    global_distillation = dict(cfg.get("global_distillation") or {})
    sort_sanity = dict(cfg.get("sort_sanity") or {})
    bypass_diagnostic = dict(cfg.get("bypass_diagnostic") or {})
    bypass = dict(cfg.get("bypass") or {})
    bypass_backend = str(bypass.get("backend", "automodel")).lower()
    if bypass_backend != "automodel":
        raise ValueError(
            f"Unsupported bypass.backend={bypass_backend!r}; "
            "AutoModel is the only supported backend"
        )
    bypass["backend"] = "automodel"

    _deep_update_missing(experiment, "dir", cfg.get("puzzle_dir"))
    _deep_update_missing(model, "source", cfg.get("input_hf_model_path"))
    _deep_update_missing(model, "trust_remote_code", cfg.get("trust_remote_code"))
    _deep_update_missing(model, "descriptor_override", cfg.get("descriptor"))
    _deep_update_missing(
        model,
        "force_hf",
        (cfg.get("pruning") or {}).get("automodel", {}).get("force_hf"),
    )
    _deep_update_missing(convert, "teacher_dir", cfg.get("teacher_dir"))
    _deep_update_missing(sort_sanity, "include_reverse", True)
    _normalize_positive_count(aiperf, "num_best_to_eval", "aiperf")
    _normalize_positive_count(
        global_distillation, "num_best_to_distill", "global_distillation"
    )
    if "dir" in experiment and "teacher_dir" not in convert:
        convert["teacher_dir"] = str(Path(experiment["dir"]) / "ckpts" / "teacher")

    # Keep a compact library view while the complete stage configuration lives
    # under the canonical vllm_stats node.
    vllm = dict(library.get("vllm") or {})
    stats_cfg = dict(cfg.get("vllm_stats") or {})
    if "enabled" not in vllm and "calculate_runtime" in stats_cfg:
        vllm["enabled"] = bool(stats_cfg.get("calculate_runtime"))
    if vllm:
        library["vllm"] = vllm

    cfg["experiment"] = experiment
    cfg["data"] = data
    cfg["model"] = model
    cfg["convert"] = convert
    cfg["library"] = library
    cfg["aiperf"] = aiperf
    cfg["global_distillation"] = global_distillation
    cfg["width_sanity"] = width_sanity
    cfg["sort_sanity"] = sort_sanity
    cfg["bypass_diagnostic"] = bypass_diagnostic
    cfg["bypass"] = bypass
    cfg["search_space"] = _normalize_search_space(cfg)
    if data_spec is not None:
        cfg["data"]["sequence_length"] = data_spec.sequence_length
    if "target_values" not in width_sanity:
        inferred_targets = {
            str(axis): list(axis_cfg.get("values") or ())[0]
            for axis, axis_cfg in (cfg["search_space"].get("axes") or {}).items()
            if isinstance(axis_cfg, dict)
            and axis_cfg.get("enabled")
            and len(list(axis_cfg.get("values") or ())) == 1
        }
        if inferred_targets:
            width_sanity["target_values"] = inferred_targets
    return cfg


def pipeline_config_from_path(
    config_path: str | Path,
    *,
    overrides: list[str] | None = None,
    num_nodes: int = 1,
    node_index: int = 0,
) -> dict[str, Any]:
    """Load a Hydra YAML and attach runtime metadata for stage handlers."""
    register_hydra_resolvers()
    path = Path(config_path).resolve()
    config_dir, config_name = _config_root_and_name(path)
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=str(config_dir),
        config_name=config_name,
        overrides=list(overrides or []),
    )
    cfg = normalize_pipeline_config(hydra_cfg)
    cfg["_runtime"] = {
        "config_path": str(path),
        "overrides": list(overrides or []),
        "num_nodes": int(num_nodes),
        "node_index": int(node_index),
    }
    return cfg


def load_runtime_hydra_config(config: dict[str, Any]) -> DictConfig:
    """Reconstruct the instantiated Hydra config used by GPU-heavy stages."""
    runtime = dict(config.get("_runtime") or {})
    config_path = runtime.get("config_path")
    if runtime.get("resolved_config_path") or not config_path:
        hydra_cfg = OmegaConf.create(config)
        OmegaConf.set_struct(hydra_cfg, False)
        _install_runtime_section_aliases(hydra_cfg)
        return adapt_runtime_hydra_config(hydra.utils.instantiate(hydra_cfg), config)

    register_hydra_resolvers()
    path = Path(config_path).resolve()
    config_dir, config_name = _config_root_and_name(path)
    hydra_cfg = initialize_hydra_config_for_dir(
        config_dir=str(config_dir),
        config_name=config_name,
        overrides=list(runtime.get("overrides") or []),
    )
    OmegaConf.set_struct(hydra_cfg, False)
    _install_runtime_section_aliases(hydra_cfg)
    _overlay_runtime_section_aliases(hydra_cfg, config)
    return adapt_runtime_hydra_config(hydra.utils.instantiate(hydra_cfg), config)


def _install_runtime_section_aliases(hydra_cfg: DictConfig) -> None:
    """Expose canonical public sections under established implementation names."""

    for canonical, runtime_name in _RUNTIME_SECTION_ALIASES.items():
        if canonical in hydra_cfg:
            hydra_cfg[runtime_name] = OmegaConf.create(
                OmegaConf.to_container(hydra_cfg[canonical], resolve=False)
            )


def _overlay_runtime_section_aliases(
    hydra_cfg: DictConfig,
    config: dict[str, Any],
) -> None:
    """Apply derived canonical stage sections over recomposed Hydra defaults."""

    for canonical, runtime_name in _RUNTIME_SECTION_ALIASES.items():
        payload = config.get(canonical)
        if not isinstance(payload, dict):
            continue
        overlay = OmegaConf.create(deepcopy(payload))
        canonical_base = hydra_cfg.get(canonical) or OmegaConf.create({})
        runtime_base = hydra_cfg.get(runtime_name) or OmegaConf.create({})
        hydra_cfg[canonical] = OmegaConf.merge(canonical_base, overlay)
        hydra_cfg[runtime_name] = OmegaConf.merge(runtime_base, overlay)


def _ensure_dictconfig_child(cfg: DictConfig, key: str) -> DictConfig:
    if not hasattr(cfg, key) or getattr(cfg, key) is None:
        cfg[key] = {}
    return getattr(cfg, key)


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _set_missing(node: DictConfig, key: str, value: Any) -> None:
    if value is not None and key not in node:
        node[key] = value


def adapt_runtime_hydra_config(
    hydra_cfg: DictConfig, config: dict[str, Any]
) -> DictConfig:
    """Fill runtime Hydra fields from the unified config boundary.

    This adapter exists only at the stage boundary. It keeps the new canonical
    Puzzletron config clean while allowing the GPU-heavy stage implementations
    to keep their established Hydra section names.
    """
    OmegaConf.set_struct(hydra_cfg, False)
    experiment_dir = _nested(config, "experiment", "dir")
    model_source = _nested(config, "model", "source")
    descriptor = _nested(config, "_runtime", "descriptor") or _nested(
        config, "model", "descriptor_override"
    )
    teacher_dir = _nested(config, "convert", "teacher_dir")
    data_path = (
        _nested(config, "data", "calibration", "path")
        or _nested(config, "data", "path")
        or config.get("dataset_path")
    )
    canonical_data = (
        PuzzletronDataSpec.from_mapping(config["data"]) if config.get("data") else None
    )

    _set_missing(hydra_cfg, "puzzle_dir", experiment_dir)
    _set_missing(hydra_cfg, "input_hf_model_path", model_source)
    _set_missing(hydra_cfg, "descriptor", descriptor)
    _set_missing(hydra_cfg, "teacher_dir", teacher_dir)
    _set_missing(hydra_cfg, "dataset_path", data_path)

    if "pruning" in hydra_cfg:
        pruning = hydra_cfg.pruning
        calibration = dict(config.get("calibration") or {})
        data = dict(_nested(config, "data", "calibration", default={}) or {})
        _set_missing(pruning, "backend", calibration.get("backend", "automodel"))
        _set_missing(pruning, "model_name_or_path", hydra_cfg.get("teacher_dir"))
        _set_missing(pruning, "descriptor", hydra_cfg.get("descriptor"))
        _set_missing(pruning, "dataset_path", hydra_cfg.get("dataset_path"))
        _set_missing(pruning, "eval_samples", data.get("num_samples"))
        _set_missing(pruning, "micro_batch_size", data.get("micro_batch_size", 1))
        _set_missing(pruning, "block_size", data.get("seq_len"))
        if canonical_data is not None:
            _set_missing(pruning, "block_size", canonical_data.sequence_length)
            _set_missing(pruning, "varlen", canonical_data.legacy_varlen)
        _set_missing(
            pruning,
            "activations_log_dir",
            str(
                Path(hydra_cfg.puzzle_dir) / "pruning" / "pruning_scores" / "activation"
            )
            if hydra_cfg.get("puzzle_dir") is not None
            else None,
        )
        if "activation_hooks_kwargs" not in pruning:
            pruning.activation_hooks_kwargs = {
                "method": calibration.get("ffn_method", "iterative")
            }
        automodel = _ensure_dictconfig_child(pruning, "automodel")
        _set_missing(
            automodel, "force_hf", _nested(config, "model", "force_hf", default=True)
        )
        _set_missing(automodel, "use_puzzletron_dataloader", True)

    for stage in ("depth", "bypass"):
        if stage in hydra_cfg:
            section = hydra_cfg[stage]
            _set_missing(section, "granularity", resolve_granularity(stage, section))

    if "calc_subblock_stats" in hydra_cfg:
        stats = hydra_cfg.calc_subblock_stats
        runtime = _ensure_dictconfig_child(stats, "runtime_stats")
        vllm_stats = dict(config.get("vllm_stats") or {})
        if vllm_stats.get("enabled") is True:
            runtime["enabled"] = True
        _set_missing(runtime, "granularity", resolve_granularity("vllm_stats", runtime))

    if "scoring" in hydra_cfg:
        scoring = hydra_cfg.scoring
        _set_missing(scoring, "granularity", resolve_granularity("scoring", scoring))
        data = dict(_nested(config, "data", "replacement_scoring", default={}) or {})
        _set_missing(scoring, "descriptor", hydra_cfg.get("descriptor"))
        _set_missing(scoring, "teacher_dir", hydra_cfg.get("teacher_dir"))
        _set_missing(scoring, "dataset_path", hydra_cfg.get("dataset_path"))
        _set_missing(
            scoring,
            "eval_samples",
            _nested(config, "replacement_scoring", "num_samples")
            or data.get("num_samples"),
        )
        _set_missing(scoring, "micro_batch_size", data.get("micro_batch_size", 1))
        if canonical_data is not None:
            _set_missing(scoring, "block_size", canonical_data.sequence_length)
            _set_missing(scoring, "varlen", canonical_data.legacy_varlen)
        _set_missing(scoring, "backend", "automodel")
        automodel = _ensure_dictconfig_child(scoring, "automodel")
        _set_missing(
            automodel, "force_hf", _nested(config, "model", "force_hf", default=True)
        )
        _set_missing(
            automodel,
            "temperature",
            _nested(config, "replacement_scoring", "temperature"),
        )
        _set_missing(
            automodel,
            "chunk_size",
            _nested(config, "replacement_scoring", "chunk_size"),
        )

    if "realize_model" in hydra_cfg:
        realize_model = hydra_cfg.realize_model
        _set_missing(realize_model, "descriptor", hydra_cfg.get("descriptor"))
        if canonical_data is not None:
            _set_missing(realize_model, "block_size", canonical_data.sequence_length)
            _set_missing(realize_model, "varlen", canonical_data.legacy_varlen)

    return hydra_cfg
