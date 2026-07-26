# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Human-checkable diagnostic stages for Puzzletron.

The activation diagnostic intentionally uses the normal sorted-teacher +
AutoModel replace-one-block scoring path.  It builds three temporary parents:
activation-sorted, reverse-sorted, and random-sorted.  Each candidate is then a
prefix slice from one parent, which exercises the exact runtime slicing contract
used later by the real library/scoring stages.
"""

from __future__ import annotations

import copy
import csv
import dataclasses
import gc
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from omegaconf import OmegaConf

import modelopt.torch.utils.distributed as dist

from ..anymodel.model_descriptor import ModelDescriptorFactory
from ..block_config import (
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    MambaConfig,
    MoEConfig,
    SubblockConfig,
    maybe_cast_block_configs,
)
from ..diagnostics.campaign_findings import MetricSpec
from ..diagnostics.width_sanity import aggregate_parent_sweep_sanity
from ..diagnostics.width_slice_equivalence import (
    evaluate_width_slice_equivalence,
    normalize_width_slice_batch,
    validate_width_slice_artifacts,
)
from ..identity import canonicalize, stable_hash
from ..pruning.sorted_teacher import build_sorted_teacher
from ..tools.checkpoint_utils import load_model_config
from ..tools.logger import mprint
from .common import complete_stage
from .pipeline import (
    _activations_log_dir,
    _distributed,
    _hf_checkpoint_complete,
    _puzzle_dir,
    _teacher_dir,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ..manifest import StageManifest

__all__ = [
    "activation_diagnostic_stage",
    "bypass_diagnostic_stage",
    "sort_equivalence_stage",
    "width_slice_equivalence_stage",
]

_SCORE_KEYS = {
    "score",
    "kv_group_scores",
    "query_head_scores",
    "mamba_head_scores",
    "mamba_head_dim_scores",
    "ssm_channel_contrib",
    "latent_cov_out",
    "key_group_scores",
    "value_lane_scores",
    "key_dim_scores",
    "value_dim_scores",
}
_PRIMARY_METRICS = (
    "raw_replacement_loss",
    "cosine_embedding_loss_hidden_states",
    "normalized_mse_loss_hidden_states",
    "mse_loss_hidden_states",
    "mae_loss_hidden_states",
    "kl_div",
    "lm_loss",
    "token_accuracy_top_1",
    "token_accuracy_top_1_consistency",
    "token_accuracy_top_5",
    "token_accuracy_top_5_consistency",
    "token_accuracy_top_10",
    "token_accuracy_top_10_consistency",
)
_LAYER_RE = re.compile(r"(?:^|\.)(?:layers|blocks)\.(\d+)(?:\.|$)")


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _to_serializable(obj: Any) -> Any:
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {key: _to_serializable(value) for key, value in dataclasses.asdict(obj).items()}
    if isinstance(obj, tuple | list):
        return [_to_serializable(value) for value in obj]
    if isinstance(obj, dict):
        return {key: _to_serializable(value) for key, value in obj.items()}
    return obj


def _entry(layer_idx: int, block_config: BlockConfig) -> dict[str, Any]:
    return {
        "weight_paths": [],
        "parent_layer_indices": [int(layer_idx)],
        "child_block_configs": [_to_serializable(block_config)],
    }


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _transform_tensor_score(
    tensor: torch.Tensor, *, method: str, key: str, seed: int
) -> torch.Tensor:
    if method in {"negative", "reverse"}:
        return -tensor
    if method == "random":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(_stable_seed(seed, key, tuple(tensor.shape)))
        return torch.rand(tensor.shape, dtype=torch.float32, generator=gen)
    return tensor


def _identity_descending_scores_like(tensor: torch.Tensor) -> torch.Tensor:
    """Scores whose descending argsort preserves the original order on the last axis."""
    values = torch.arange(tensor.shape[-1], 0, -1, dtype=torch.float32)
    shape = (1,) * (tensor.ndim - 1) + (tensor.shape[-1],)
    return values.reshape(shape).expand(tensor.shape).clone()


def _attention_axis_payload(
    payload: dict[str, Any], *, method: str, key_path: str, axis: str, seed: int
) -> dict[str, Any]:
    """Keep grouped-attention diagnostics to one axis at a time.

    ``build_sorted_teacher`` can sort KV groups, query heads within each group,
    from one grouped-attention hook payload.  The diagnostic is
    meant to test one pruning axis, so non-tested axes get identity scores rather
    than activation/random/negative permutations.  This keeps methods that pick
    the same KV group physically identical for a KV-only diagnostic.
    """

    out = {key: value for key, value in payload.items()}
    kv_scores = payload.get("kv_group_scores")
    query_scores = payload.get("query_head_scores")
    axis = str(axis)

    if axis in {"kv_groups", "kv_heads", "num_kv_heads"}:
        if torch.is_tensor(kv_scores):
            out["kv_group_scores"] = _transform_tensor_score(
                kv_scores, method=method, key=f"{key_path}.kv_group_scores", seed=seed
            )
        if torch.is_tensor(query_scores):
            out["query_head_scores"] = _identity_descending_scores_like(query_scores)
        return out

    if axis in {"q_heads_per_group", "query_heads", "num_query_heads"}:
        if torch.is_tensor(kv_scores):
            out["kv_group_scores"] = _identity_descending_scores_like(kv_scores)
        if torch.is_tensor(query_scores):
            out["query_head_scores"] = _transform_tensor_score(
                query_scores, method=method, key=f"{key_path}.query_head_scores", seed=seed
            )
        return out

    return out


def _gdn_axis_payload(
    payload: dict[str, Any], *, method: str, key_path: str, axis: str, seed: int
) -> dict[str, Any]:
    out = {key: value for key, value in payload.items()}
    shape = payload.get("shape") or {}
    groups = int(shape.get("num_key_heads", 0))
    value_heads = int(shape.get("num_value_heads", 0))
    ratio = value_heads // groups if groups else 0
    key_dim = int(shape.get("key_head_dim", 0))
    value_dim = int(shape.get("value_head_dim", 0))
    specs = {
        "gdn_key_groups": (
            "key_group_scores",
            "key_group_order_most_important_first",
            (groups,),
        ),
        "gdn_value_heads_per_group": (
            "value_lane_scores",
            "value_lane_order_most_important_first",
            (groups, ratio),
        ),
        "gdn_key_head_dim": (
            "key_dim_scores",
            "key_dim_order_most_important_first",
            (groups, key_dim),
        ),
        "gdn_value_head_dim": (
            "value_dim_scores",
            "value_dim_order_most_important_first",
            (value_dim,),
        ),
    }
    for axis_name, (score_key, order_key, expected_shape) in specs.items():
        scores = payload.get(score_key)
        if not torch.is_tensor(scores):
            continue
        if axis_name == axis:
            scores = _transform_tensor_score(
                scores, method=method, key=f"{key_path}.{score_key}", seed=seed
            )
            out[score_key] = scores
            out[order_key] = torch.argsort(scores, dim=-1, descending=True, stable=True)
        else:
            identity = torch.arange(expected_shape[-1], dtype=torch.long)
            out[order_key] = (
                identity.reshape((1,) * (len(expected_shape) - 1) + (-1,))
                .expand(expected_shape)
                .clone()
            )
    return out


def _all_axes_rank_payload(
    payload: dict[str, Any], *, method: str, key_path: str, seed: int
) -> dict[str, Any]:
    """Transform every sortable field while keeping derived orders consistent."""

    out = {key: value for key, value in payload.items()}
    if "key_group_scores" in payload and "key_dim_scores" in payload:
        order_fields = {
            "key_group_scores": "key_group_order_most_important_first",
            "value_lane_scores": "value_lane_order_most_important_first",
            "key_dim_scores": "key_dim_order_most_important_first",
            "value_dim_scores": "value_dim_order_most_important_first",
        }
        for score_key, order_key in order_fields.items():
            scores = payload.get(score_key)
            if not torch.is_tensor(scores):
                continue
            transformed = _transform_tensor_score(
                scores,
                method=method,
                key=f"{key_path}.{score_key}",
                seed=seed,
            )
            out[score_key] = transformed
            out[order_key] = torch.argsort(
                transformed,
                dim=-1,
                descending=True,
                stable=True,
            )
        return out

    if "kv_group_scores" in payload and "query_head_scores" in payload:
        for score_key in ("kv_group_scores", "query_head_scores"):
            scores = payload.get(score_key)
            if torch.is_tensor(scores):
                out[score_key] = _transform_tensor_score(
                    scores,
                    method=method,
                    key=f"{key_path}.{score_key}",
                    seed=seed,
                )
        return out

    return {
        key: _transform_payload(
            value,
            method=method,
            key_path=f"{key_path}.{key}" if key_path else key,
            seed=seed,
            axis="__all__",
        )
        for key, value in payload.items()
    }


def _transform_payload(
    payload: Any, *, method: str, key_path: str, seed: int, axis: str | None = None
) -> Any:
    if isinstance(payload, dict):
        if (
            method in {"negative", "reverse"}
            and axis in {"__all__", "moe_latent_dim", "latent_dim"}
            and all(
                torch.is_tensor(payload.get(key))
                for key in ("latent_cov_in", "expert_weights_sum", "latent_cov_out")
            )
        ):
            out = {key: value for key, value in payload.items()}
            out["reverse_ranking"] = True
            return out
        if axis == "__all__":
            return _all_axes_rank_payload(
                payload,
                method=method,
                key_path=key_path,
                seed=seed,
            )
        if axis is not None and "key_group_scores" in payload and "key_dim_scores" in payload:
            return _gdn_axis_payload(
                payload, method=method, key_path=key_path, axis=axis, seed=seed
            )
        if axis is not None and "kv_group_scores" in payload and "query_head_scores" in payload:
            return _attention_axis_payload(
                payload, method=method, key_path=key_path, axis=axis, seed=seed
            )
        return {
            key: _transform_payload(
                value,
                method=method,
                key_path=f"{key_path}.{key}" if key_path else key,
                seed=seed,
                axis=axis,
            )
            for key, value in payload.items()
        }
    if torch.is_tensor(payload) and key_path.rsplit(".", 1)[-1] in _SCORE_KEYS:
        return _transform_tensor_score(payload, method=method, key=key_path, seed=seed)
    return payload


def _module_layer_idx(module_name: str) -> int | None:
    match = _LAYER_RE.search(module_name)
    return int(match.group(1)) if match else None


def _write_transformed_activation_logs(
    source_dir: Path,
    output_dir: Path,
    *,
    method: str,
    seed: int,
    selected_passes: tuple[str, ...] = (),
    axis: str | None = None,
    target_layers: set[int] | None = None,
) -> Path:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = set(selected_passes)
    for source_path in source_dir.rglob("*"):
        rel = source_path.relative_to(source_dir)
        if rel.parts and selected and rel.parts[0] not in selected:
            continue
        dst = output_dir / rel
        if source_path.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if source_path.suffix == ".pth":
            raw = torch.load(source_path, map_location="cpu")
            transformed = {
                key: _transform_payload(value, method=method, key_path=str(key), seed=seed)
                for key, value in raw.items()
                if target_layers is None or _module_layer_idx(str(key)) in target_layers
            }
            if axis is not None:
                transformed = {
                    key: _transform_payload(
                        value,
                        method=method,
                        key_path=str(key),
                        seed=seed,
                        axis=axis,
                    )
                    for key, value in raw.items()
                    if target_layers is None or _module_layer_idx(str(key)) in target_layers
                }
            torch.save(transformed, dst)
        else:
            shutil.copy2(source_path, dst)
    if selected:
        (output_dir / "activation_passes_manifest.json").write_text(
            json.dumps({"passes": list(selected_passes)}, indent=2) + "\n"
        )
    if not list(output_dir.rglob("rank_*.pth")):
        raise FileNotFoundError(
            f"No activation rank files copied from {source_dir} to {output_dir}; "
            f"selected_passes={list(selected_passes)}"
        )
    return output_dir


def _select_layers(
    eligible_layers: list[int],
    count: int,
    explicit_layers: list[int] | None = None,
    *,
    selection: str = "spread",
    seed: int = 1234,
    axis: str = "",
) -> list[int]:
    if explicit_layers is not None:
        eligible = set(eligible_layers)
        missing = [idx for idx in explicit_layers if idx not in eligible]
        if missing:
            raise ValueError(
                f"Requested diagnostic layer_indices={missing} are not eligible; "
                f"eligible_layers={eligible_layers}"
            )
        return list(dict.fromkeys(int(idx) for idx in explicit_layers))
    if len(eligible_layers) <= count:
        return list(eligible_layers)
    if count <= 1:
        if selection == "random":
            return [
                min(
                    eligible_layers,
                    key=lambda layer_idx: hashlib.sha256(
                        f"{seed}:{axis}:{layer_idx}".encode("utf-8")
                    ).digest(),
                )
            ]
        if selection != "spread":
            raise ValueError(
                f"Unknown activation diagnostic layer selection {selection!r}; "
                "expected 'spread' or 'random'"
            )
        return [eligible_layers[0]]
    if selection == "random":
        # Rank candidates by a stable hash rather than Python's process-randomized
        # hash().  Including the axis gives every pruning axis an independent,
        # reproducible layer sample while remaining identical on every rank.
        ranked = sorted(
            eligible_layers,
            key=lambda layer_idx: hashlib.sha256(
                f"{seed}:{axis}:{layer_idx}".encode("utf-8")
            ).digest(),
        )
        return sorted(ranked[:count])
    if selection != "spread":
        raise ValueError(
            f"Unknown activation diagnostic layer selection {selection!r}; "
            "expected 'spread' or 'random'"
        )
    selected = {eligible_layers[0], eligible_layers[-1]}
    span = len(eligible_layers) - 1
    for i in range(1, count - 1):
        selected.add(eligible_layers[round(i * span / (count - 1))])
    cursor = 0
    while len(selected) < count and cursor < len(eligible_layers):
        selected.add(eligible_layers[cursor])
        cursor += 1
    return sorted(selected)


def _enabled_diagnostic_axes(config: dict[str, Any], diag_cfg: dict[str, Any]) -> list[str]:
    requested = diag_cfg.get("axes")
    if requested is not None:
        return [str(axis) for axis in requested]
    axes = (config.get("search_space") or {}).get("axes") or {}
    return [
        axis
        for axis, axis_cfg in axes.items()
        if isinstance(axis_cfg, dict) and axis_cfg.get("enabled")
    ]


def _representative_axis_targets(
    config: dict[str, Any],
    axes: Iterable[str],
    explicit: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Select one configured target nearest half-width for every requested axis."""

    explicit = dict(explicit or {})
    search_axes = (config.get("search_space") or {}).get("axes") or {}
    selected: dict[str, int] = {}
    for axis in axes:
        axis_cfg = search_axes.get(axis) if isinstance(search_axes, dict) else None
        raw = explicit.get(axis)
        if raw is None and isinstance(axis_cfg, dict):
            raw = axis_cfg.get("values")
        if raw is None:
            continue
        values = [int(raw)] if isinstance(raw, int | float | str) else [int(v) for v in raw]
        teacher = axis_cfg.get("teacher_value") if isinstance(axis_cfg, dict) else None
        if teacher is None:
            selected[axis] = values[len(values) // 2]
            continue
        legal = [value for value in values if 0 < value < int(teacher)]
        if legal:
            selected[axis] = min(
                legal,
                key=lambda value: (abs(2 * int(value) - int(teacher)), -int(value)),
            )
    return selected


def _near_teacher_axis_targets(
    config: dict[str, Any],
    axes: Iterable[str],
    *,
    count: int = 2,
) -> dict[str, list[int]]:
    """Select up to ``count`` largest legal configured targets per axis."""

    if count < 1:
        raise ValueError(f"axis target count must be positive, got {count}")
    search_axes = (config.get("search_space") or {}).get("axes") or {}
    selected: dict[str, list[int]] = {}
    for axis in axes:
        axis_cfg = search_axes.get(axis) if isinstance(search_axes, dict) else None
        if not isinstance(axis_cfg, dict):
            continue
        raw_values = axis_cfg.get("values") or ()
        values = [raw_values] if isinstance(raw_values, int | float | str) else raw_values
        teacher = axis_cfg.get("teacher_value")
        legal = sorted(
            {
                int(value)
                for value in values
                if int(value) > 0 and (teacher is None or int(value) < int(teacher))
            },
            reverse=True,
        )
        if legal:
            selected[str(axis)] = legal[:count]
    return selected


def _ratio_aligned_hidden_widths(
    teacher_width: int,
    ratios: Iterable[float],
    *,
    alignment: int = 1,
) -> list[int]:
    """Convert width ratios to distinct legal aligned widths in request order."""

    if alignment < 1:
        raise ValueError(f"hidden-width alignment must be positive, got {alignment}")
    ratio_values = [float(ratio) for ratio in ratios]
    widths = []
    for ratio in ratio_values:
        width = int(ratio * int(teacher_width))
        width = max(alignment, (width // alignment) * alignment)
        if 0 < width < int(teacher_width) and width not in widths:
            widths.append(width)
    if not widths:
        raise ValueError(
            f"hidden-width ratios produced no legal target: teacher={teacher_width} "
            f"ratios={ratio_values} alignment={alignment}"
        )
    return widths


def _axis_subblock_and_field(axis: str) -> tuple[str, str] | None:
    mapping = {
        "ffn_intermediate": ("ffn", "intermediate_size"),
        "intermediate_size": ("ffn", "intermediate_size"),
        "kv_groups": ("attention", "num_kv_heads"),
        "kv_heads": ("attention", "num_kv_heads"),
        "num_kv_heads": ("attention", "num_kv_heads"),
        "query_heads": ("attention", "q_heads_per_group"),
        "num_query_heads": ("attention", "num_query_heads"),
        "q_heads_per_group": ("attention", "q_heads_per_group"),
        "moe_experts": ("moe", "num_experts"),
        "num_experts": ("moe", "num_experts"),
        "moe_expert_intermediate": ("moe", "expert_intermediate_size"),
        "expert_intermediate_size": ("moe", "expert_intermediate_size"),
        "moe_shared_expert_intermediate": ("moe", "shared_expert_intermediate_size"),
        "shared_expert_intermediate_size": ("moe", "shared_expert_intermediate_size"),
        "moe_latent_dim": ("moe", "latent_dim"),
        "latent_dim": ("moe", "latent_dim"),
        "moe_top_k": ("moe", "top_k"),
        "top_k": ("moe", "top_k"),
        "mamba_heads": ("mamba", "num_heads"),
        "num_heads": ("mamba", "num_heads"),
        "mamba_head_dim": ("mamba", "head_dim"),
        "mamba_state_dim": ("mamba", "state_dim"),
        "state_dim": ("mamba", "state_dim"),
        "gdn_key_groups": ("mamba", "gdn_key_groups"),
        "gdn_value_heads_per_group": ("mamba", "gdn_value_heads_per_group"),
        "gdn_key_head_dim": ("mamba", "state_dim"),
        "gdn_value_head_dim": ("mamba", "head_dim"),
    }
    return mapping.get(axis)


def _teacher_axis_value(
    subblock: SubblockConfig, field: str, fallback: int | None = None
) -> int | None:
    if field == "q_heads_per_group":
        q = getattr(subblock, "num_query_heads", None)
        kv = getattr(subblock, "num_kv_heads", None)
        if q is None or kv is None:
            return None
        return int(q) // int(kv)
    if field == "gdn_key_groups":
        value = getattr(subblock, "num_groups", None)
        return int(value) if value is not None else fallback
    if field == "gdn_value_heads_per_group":
        heads = getattr(subblock, "num_heads", None)
        groups = getattr(subblock, "num_groups", None)
        if heads is None or groups is None:
            return fallback
        return int(heads) // int(groups)
    value = getattr(subblock, field, None)
    return int(value) if value is not None else fallback


def _ratio_targets(teacher_value: int, ratios: list[float]) -> list[tuple[float, int]]:
    values: list[tuple[float, int]] = []
    for ratio in ratios:
        target = max(1, int(round(float(teacher_value) * float(ratio))))
        if target >= teacher_value:
            continue
        values.append((float(target) / float(teacher_value), target))
    deduped: dict[int, float] = {}
    for ratio, target in values:
        deduped.setdefault(target, ratio)
    return [(ratio, target) for target, ratio in sorted(deduped.items())]


def _as_int_list(value: Any) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    if isinstance(value, int):
        return [int(value)]
    return [int(item) for item in value]


def _replace_axis(
    block_config: BlockConfig,
    *,
    subblock_kind: str,
    field: str,
    target: int,
    teacher_value_fallback: int | None = None,
) -> BlockConfig | None:
    subblock = block_config.get_subblock(subblock_kind)
    if subblock is None or getattr(subblock, "no_op", False):
        return None
    teacher_value = _teacher_axis_value(subblock, field, teacher_value_fallback)
    if teacher_value is None or int(target) >= int(teacher_value):
        return None
    updates = {field: int(target)}
    if subblock_kind == "mamba" and field == "gdn_key_groups":
        groups = int(getattr(subblock, "num_groups"))
        ratio = int(getattr(subblock, "num_heads")) // groups
        updates = {"num_groups": int(target), "num_heads": int(target) * ratio}
    if subblock_kind == "mamba" and field == "gdn_value_heads_per_group":
        groups = int(getattr(subblock, "num_groups"))
        updates = {"num_groups": groups, "num_heads": groups * int(target)}
    if subblock_kind == "attention" and field == "num_kv_heads":
        q = getattr(subblock, "num_query_heads", None)
        kv = getattr(subblock, "num_kv_heads", None)
        if q is not None and kv is not None:
            updates["num_query_heads"] = int(target) * (int(q) // int(kv))
    if subblock_kind == "attention" and field == "num_query_heads":
        kv = getattr(subblock, "num_kv_heads", None)
        if kv is not None and int(target) % int(kv) != 0:
            return None
    if subblock_kind == "attention" and field == "q_heads_per_group":
        kv = getattr(subblock, "num_kv_heads", None)
        if kv is None:
            return None
        updates = {
            "num_kv_heads": int(kv),
            "num_query_heads": int(kv) * int(target),
        }
    child = dataclasses.replace(subblock, **updates)
    replace_kinds = {
        "attention": ("attention", "mamba"),
        "mamba": ("attention", "mamba"),
        "ffn": ("ffn", "moe"),
        "moe": ("ffn", "moe"),
    }.get(subblock_kind, (subblock_kind,))
    return block_config.with_subblock(child, replace_kinds=replace_kinds)


def _teacher_replacements(block_configs: list[BlockConfig]) -> dict[int, dict[str, Any]]:
    return {idx: _entry(idx, block_config) for idx, block_config in enumerate(block_configs)}


def _diagnostic_solutions(
    block_configs: list[BlockConfig],
    *,
    axes: list[str],
    ratios: list[float],
    target_values: dict[str, Any] | None,
    layer_count: int,
    layer_indices: list[int] | None = None,
    layer_selection: str = "spread",
    layer_seed: int = 1234,
    fallback_values: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fallback_values = fallback_values or {}
    target_values = target_values or {}
    teacher_entries = _teacher_replacements(block_configs)
    entries = list(teacher_entries.values())
    solutions: list[dict[str, Any]] = []
    solution_id = 0
    for axis in axes:
        target_info = _axis_subblock_and_field(axis)
        if target_info is None:
            continue
        subblock_kind, field = target_info
        eligible = [
            idx
            for idx, block in enumerate(block_configs)
            if block.get_subblock(subblock_kind) is not None
            and not getattr(block.get_subblock(subblock_kind), "no_op", False)
            and (
                _teacher_axis_value(block.get_subblock(subblock_kind), field) is not None
                or f"{subblock_kind}.{field}" in fallback_values
            )
        ]
        for layer_idx in _select_layers(
            eligible,
            layer_count,
            layer_indices,
            selection=layer_selection,
            seed=layer_seed,
            axis=axis,
        ):
            base = block_configs[layer_idx]
            subblock = base.require_subblock(subblock_kind)
            teacher_value = int(
                _teacher_axis_value(
                    subblock,
                    field,
                    fallback_values.get(f"{subblock_kind}.{field}"),
                )
            )
            configured_targets = target_values.get(axis)
            if configured_targets is None:
                targets = _ratio_targets(teacher_value, ratios)
            else:
                if not isinstance(configured_targets, (list, tuple)):
                    configured_targets = [configured_targets]
                targets = []
                for raw_target in configured_targets:
                    target = int(raw_target)
                    if target <= 0 or target >= teacher_value:
                        raise ValueError(
                            f"Invalid activation diagnostic target for {axis}: "
                            f"teacher={teacher_value}, target={target}"
                        )
                    targets.append((float(target) / float(teacher_value), target))
                targets = list(dict.fromkeys(targets))
            for ratio, target in targets:
                child = _replace_axis(
                    base,
                    subblock_kind=subblock_kind,
                    field=field,
                    target=target,
                    teacher_value_fallback=fallback_values.get(f"{subblock_kind}.{field}"),
                )
                if child is None or child == base:
                    continue
                candidate = _entry(layer_idx, child)
                candidate["diagnostic"] = {
                    "axis": axis,
                    "subblock_kind": subblock_kind,
                    "field": field,
                    "layer_idx": layer_idx,
                    "teacher_value": teacher_value,
                    "target_value": target,
                    "ratio": ratio,
                    "solution_id": solution_id,
                }
                chosen = [
                    candidate if idx == layer_idx else teacher_entries[idx]
                    for idx in range(len(block_configs))
                ]
                solution = {
                    "single_sequence_replacement": candidate,
                    "chosen_replacements": chosen,
                    "block_configs": [
                        replacement["child_block_configs"][0] for replacement in chosen
                    ],
                    "diagnostic": candidate["diagnostic"],
                }
                entries.append(candidate)
                solutions.append(solution)
                solution_id += 1
    return entries, solutions


def _configured_bypass_block_targets(
    block_configs: list[BlockConfig],
    *,
    axes: list[str],
    ratios: list[float],
    layer_count: int,
    layer_indices: list[int] | None = None,
    config: dict[str, Any],
    fallback_values: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build block-level pruning combinations for bypass-vs-pruned diagnosis.

    Activation diagnostics are intentionally one-axis-at-a-time.  Nested block
    bypass, however, trains a block supernet with all configured subblock sizes
    sampled together.  Its validation should therefore compare a pruned block
    candidate with the same candidate sliced from the bypass checkpoint.
    """

    fallback_values = fallback_values or {}
    teacher_entries = _teacher_replacements(block_configs)
    entries = list(teacher_entries.values())
    solutions: list[dict[str, Any]] = []

    pruning_cfg = config.get("pruning") or {}
    ffn_sizes = [int(x) for x in (pruning_cfg.get("intermediate_size_list") or [])]
    attn_targets = [
        (int(pair[0]), int(pair[1]))
        for pair in (
            pruning_cfg.get("attn_heads_list") or pruning_cfg.get("attention_groups_list") or []
        )
    ]
    requested = {
        _axis_subblock_and_field(axis)[0] for axis in axes if _axis_subblock_and_field(axis)
    }
    use_ffn = bool(ffn_sizes) and (not requested or "ffn" in requested or "attention" in requested)
    use_attn = bool(attn_targets) and (not requested or "attention" in requested)
    prefer_attention_layers = use_attn and bool(attn_targets)

    eligible: list[int] = []
    for idx, block in enumerate(block_configs):
        has_ffn_target = False
        has_attn_target = False
        ffn = block.get_subblock("ffn")
        attn = block.get_subblock("attention")
        if use_ffn and ffn is not None and not getattr(ffn, "no_op", False):
            teacher_ffn = _teacher_axis_value(ffn, "intermediate_size")
            has_ffn_target = any(size < teacher_ffn for size in ffn_sizes)
        if use_attn and attn is not None and not getattr(attn, "no_op", False):
            teacher_q = _teacher_axis_value(attn, "num_query_heads")
            teacher_kv = _teacher_axis_value(attn, "num_kv_heads")
            has_attn_target = any(
                q <= teacher_q and kv <= teacher_kv and (q < teacher_q or kv < teacher_kv)
                for q, kv in attn_targets
            )
        has_target = (
            has_attn_target if prefer_attention_layers else (has_ffn_target or has_attn_target)
        )
        if has_target:
            eligible.append(idx)

    solution_id = 0
    for layer_idx in _select_layers(eligible, layer_count, layer_indices):
        base = block_configs[layer_idx]
        ffn = base.get_subblock("ffn")
        attn = base.get_subblock("attention")
        teacher_ffn = _teacher_axis_value(ffn, "intermediate_size") if ffn is not None else None
        teacher_q = _teacher_axis_value(attn, "num_query_heads") if attn is not None else None
        teacher_kv = _teacher_axis_value(attn, "num_kv_heads") if attn is not None else None

        layer_ffn_sizes = [None]
        if use_ffn and teacher_ffn is not None:
            layer_ffn_sizes = sorted({size for size in ffn_sizes if 0 < size < int(teacher_ffn)})
            if not layer_ffn_sizes:
                layer_ffn_sizes = [None]
        layer_attn_targets: list[tuple[int, int] | None] = [None]
        if use_attn and teacher_q is not None and teacher_kv is not None:
            layer_attn_targets = sorted(
                {
                    (q, kv)
                    for q, kv in attn_targets
                    if 0 < q <= int(teacher_q)
                    and 0 < kv <= int(teacher_kv)
                    and q % kv == 0
                    and (q < int(teacher_q) or kv < int(teacher_kv))
                }
            )
            if not layer_attn_targets:
                layer_attn_targets = [None]

        for ffn_size in layer_ffn_sizes:
            for attn_target in layer_attn_targets:
                if ffn_size is None and attn_target is None:
                    continue
                child = base
                pieces: list[str] = []
                if ffn_size is not None:
                    child_ffn = dataclasses.replace(
                        child.require_subblock("ffn"),
                        intermediate_size=int(ffn_size),
                    )
                    child = child.with_subblock(child_ffn, replace_kinds=("ffn", "moe"))
                    pieces.append(f"ffn={ffn_size}")
                if attn_target is not None:
                    q, kv = attn_target
                    child_attn = dataclasses.replace(
                        child.require_subblock("attention"),
                        num_query_heads=int(q),
                        num_kv_heads=int(kv),
                    )
                    child = child.with_subblock(child_attn, replace_kinds=("attention", "mamba"))
                    pieces.append(f"attn_q={q}")
                    pieces.append(f"attn_kv={kv}")
                if child == base:
                    continue

                combo = ",".join(pieces)
                candidate = _entry(layer_idx, child)
                candidate["diagnostic"] = {
                    "axis": "block_combo",
                    "combo": combo,
                    "subblock_kind": "block",
                    "field": "block_config",
                    "layer_idx": layer_idx,
                    "teacher_value": 1,
                    "target_value": solution_id,
                    "ratio": 1.0,
                    "solution_id": solution_id,
                }
                chosen = [
                    candidate if idx == layer_idx else teacher_entries[idx]
                    for idx in range(len(block_configs))
                ]
                solution = {
                    "single_sequence_replacement": candidate,
                    "chosen_replacements": chosen,
                    "block_configs": [
                        replacement["child_block_configs"][0] for replacement in chosen
                    ],
                    "diagnostic": candidate["diagnostic"],
                }
                entries.append(candidate)
                solutions.append(solution)
                solution_id += 1

    return entries, solutions


def _entries_for_solutions(
    block_configs: list[BlockConfig],
    solutions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    entries = list(_teacher_replacements(block_configs).values())
    entries.extend(solution["single_sequence_replacement"] for solution in solutions)
    return entries


def _json_cell(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"))


def _read_sorted_permutations(sorted_teacher_dir: Path) -> dict[str, Any]:
    path = sorted_teacher_dir / "sorted_permutations.json"
    if not path.is_file():
        return {}
    permutations = json.loads(path.read_text())
    sidecars: dict[Path, dict[str, torch.Tensor]] = {}
    for key, value in list(permutations.items()):
        if not isinstance(value, dict) or not value.get("sidecar"):
            continue
        sidecar_path = sorted_teacher_dir / str(value["sidecar"])
        if sidecar_path not in sidecars:
            sidecars[sidecar_path] = torch.load(sidecar_path, map_location="cpu", weights_only=True)
        tensor = sidecars[sidecar_path].get(key)
        if not torch.is_tensor(tensor):
            raise KeyError(f"Missing permutation {key!r} in {sidecar_path}")
        expected_shape = tuple(int(dim) for dim in value.get("shape", tensor.shape))
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(f"Permutation {key!r} shape {tuple(tensor.shape)} != {expected_shape}")
        permutations[key] = tensor.tolist()
    return permutations


def _annotate_solution_selections(
    *,
    solutions: list[dict[str, Any]],
    teacher_block_configs: list[BlockConfig],
    sorted_teacher_dir: Path,
) -> None:
    """Attach human-checkable evidence for prefix-slice diagnostics.

    For KV-group diagnosis the pruned candidate always keeps the first
    ``target_num_kv`` groups from the sorted parent. This annotation records
    which original teacher groups those sorted-prefix entries came from, so a
    two-group teacher visibly has only two possible choices.
    """

    teacher_serialized = [_to_serializable(block_config) for block_config in teacher_block_configs]
    permutations = _read_sorted_permutations(sorted_teacher_dir)
    for solution in solutions:
        diag = solution.get("diagnostic") or {}
        layer_idx = int(diag.get("layer_idx"))
        changed_layers = [
            idx
            for idx, child in enumerate(solution.get("block_configs") or [])
            if canonicalize(child) != canonicalize(teacher_serialized[idx])
        ]
        diag["changed_layers"] = changed_layers
        diag["num_changed_layers"] = len(changed_layers)
        if changed_layers != [layer_idx]:
            raise ValueError(
                "Activation diagnostic must change exactly one layer at a time; "
                f"axis={diag.get('axis')} layer={layer_idx} changed_layers={changed_layers}"
            )

        if diag.get("field") == "num_kv_heads":
            order = permutations.get(f"attn.kv.{layer_idx}")
            if order is not None:
                order = [int(item) for item in order]
                target = int(diag["target_value"])
                diag["kv_group_order"] = order
                diag["kept_kv_groups"] = order[:target]
                diag["removed_kv_groups"] = order[target:]
                diag["selection_basis"] = "sorted_prefix_kv_groups"

        if diag.get("field") == "q_heads_per_group":
            order = permutations.get(f"attn.q.{layer_idx}")
            teacher_attn = teacher_block_configs[layer_idx].get_subblock("attention")
            if order is not None and teacher_attn is not None:
                order = [int(item) for item in order]
                num_kv = int(teacher_attn.num_kv_heads)
                heads_per_group = int(teacher_attn.num_query_heads) // num_kv
                target = int(diag["target_value"])
                grouped_order = [
                    order[idx * heads_per_group : (idx + 1) * heads_per_group]
                    for idx in range(num_kv)
                ]
                diag["query_head_order_per_group"] = grouped_order
                diag["kept_query_heads_per_group"] = [group[:target] for group in grouped_order]
                diag["removed_query_heads_per_group"] = [group[target:] for group in grouped_order]
                diag["selection_basis"] = "sorted_prefix_query_heads_per_group"

        if diag.get("field") == "num_experts" and diag.get("subblock_kind") == "moe":
            order = permutations.get(f"moe.experts.{layer_idx}")
            if order is not None:
                order = [int(item) for item in order]
                target = int(diag["target_value"])
                diag["expert_order"] = order
                diag["kept_experts"] = order[:target]
                diag["removed_experts"] = order[target:]
                diag["selection_basis"] = "ranked_original_expert_ids"

        gdn_perm_key = {
            "gdn_key_groups": "gdn.key_groups",
            "state_dim": "gdn.key_dim",
            "head_dim": "gdn.value_dim",
        }.get(diag.get("field"))
        if diag.get("subblock_kind") == "mamba" and gdn_perm_key is not None:
            order = permutations.get(f"{gdn_perm_key}.{layer_idx}")
            if order is not None:
                target = int(diag["target_value"])
                if order and isinstance(order[0], list):
                    order = [[int(item) for item in group] for group in order]
                    diag["unit_order"] = order
                    diag["kept_units"] = [group[:target] for group in order]
                    diag["removed_units"] = [group[target:] for group in order]
                else:
                    order = [int(item) for item in order]
                    diag["unit_order"] = order
                    diag["kept_units"] = order[:target]
                    diag["removed_units"] = order[target:]
                diag["selection_basis"] = f"sorted_prefix_{diag.get('axis')}"

        solution["diagnostic"] = diag
        candidate = solution.get("single_sequence_replacement")
        if isinstance(candidate, dict):
            candidate["diagnostic"] = diag


def _write_library_and_solutions(
    method_dir: Path,
    sorted_teacher_dir: Path,
    entries: list[dict[str, Any]],
    solutions: list[dict[str, Any]],
) -> tuple[Path, Path]:
    method_dir.mkdir(parents=True, exist_ok=True)
    library_path = method_dir / "replacement_library.json"
    solutions_path = method_dir / "single_sequence_replacement_solutions.json"
    library = {
        "version": 2,
        "sorted_teacher_dir": str(sorted_teacher_dir.resolve()),
        "entries": entries,
    }
    library_path.write_text(json.dumps(canonicalize(library), indent=2, sort_keys=True) + "\n")
    solutions_path.write_text(json.dumps(canonicalize(solutions), indent=2, sort_keys=True) + "\n")
    return library_path, solutions_path


def _identity_width_solution(block_configs: list[BlockConfig], hidden_width: int) -> dict[str, Any]:
    replacements = [_entry(layer_idx, block) for layer_idx, block in enumerate(block_configs)]
    identity = json.loads(json.dumps(canonicalize(replacements[0])))
    identity["diagnostic"] = {
        "axis": "hidden_width",
        "target_value": int(hidden_width),
        "num_changed_layers": 0,
    }
    return {
        "single_sequence_replacement": identity,
        "chosen_replacements": replacements,
        "block_configs": [_to_serializable(block) for block in block_configs],
        "hidden_width": int(hidden_width),
        "scenario": f"width-{int(hidden_width):04d}",
    }


def _hidden_width_ranking_verdict(
    values: dict[str, float],
    *,
    tolerance: float,
    require_beats_random: bool,
) -> dict[str, bool]:
    """Evaluate ranking gates while keeping the original-prefix control explicit."""

    original = values.get("original", values.get("random"))
    if original is None:
        raise ValueError(f"hidden-width diagnosis is missing original-order score: {values}")
    beats_random = values["activation"] <= original + tolerance
    beats_reverse = values["activation"] <= values["reverse"] + tolerance
    return {
        "passed": beats_reverse and (beats_random or not require_beats_random),
        "beats_random": beats_random,
        "beats_reverse": beats_reverse,
        "require_beats_random": bool(require_beats_random),
    }


def _hidden_width_realization_tolerance(
    diag_cfg: dict[str, Any], metric: str
) -> float:
    """Resolve the physical gate independently from ranking comparisons."""

    comparison_tolerance = float(diag_cfg.get("comparison_tolerance", 0.0))
    default = float(
        diag_cfg.get("physical_equivalence_tolerance", comparison_tolerance)
    )
    overrides = dict(diag_cfg.get("physical_equivalence_tolerances") or {})
    return float(overrides.get(str(metric), default))


def _select_diagnostic_hidden_width(
    teacher_width: int,
    widths: Iterable[int],
) -> int:
    """Select the legal reduced width nearest to seven eighths of the teacher."""

    candidates = sorted({int(width) for width in widths if 0 < int(width) < int(teacher_width)})
    if not candidates:
        raise ValueError(
            "activation hidden-width diagnosis requires at least one configured "
            f"reduced width; teacher={teacher_width} configured={list(widths)}"
        )
    return min(
        candidates,
        key=lambda width: (abs(8 * width - 7 * int(teacher_width)), -width),
    )


def _run_hidden_width_diagnostic(
    hydra_cfg: Any,
    *,
    descriptor,
    teacher_dir: Path,
    sorted_dir: Path,
    reverse_dir: Path,
    block_configs: list[BlockConfig],
    puzzle_dir: Path,
    artifacts_dir: Path,
    diag_cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(diag_cfg.get("hidden_width_diagnostic", True)):
        return None
    embedding_cfg = dict(_get(hydra_cfg, "embedding_pruning", {}) or {})
    if not bool(embedding_cfg.get("enabled", False)):
        return None
    teacher_config = load_model_config(
        teacher_dir,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    teacher_width = int(descriptor.get_language_model_config(teacher_config).hidden_size)
    alignment = int(embedding_cfg.get("alignment", 1))
    configured_targets = diag_cfg.get("hidden_width_targets")
    if configured_targets is None:
        configured_widths = [int(width) for width in embedding_cfg.get("widths", ())]
        targets = [_select_diagnostic_hidden_width(teacher_width, configured_widths)]
    else:
        targets = [int(width) for width in configured_targets]
        invalid = [width for width in targets if width <= 0 or width >= teacher_width]
        if invalid or len(targets) != len(set(targets)):
            raise ValueError(
                "hidden_width_targets must be distinct reduced widths: "
                f"teacher={teacher_width} targets={targets}"
            )

    summaries = []
    for width in targets:
        summary = _run_hidden_width_diagnostic_at_width(
            hydra_cfg,
            descriptor=descriptor,
            teacher_dir=teacher_dir,
            block_configs=block_configs,
            puzzle_dir=puzzle_dir,
            artifacts_dir=artifacts_dir,
            diag_cfg=diag_cfg,
            teacher_width=teacher_width,
            width=width,
            alignment=alignment,
        )
        if dist.is_master() and summary is not None:
            summaries.append(summary)
    if not dist.is_master():
        return None

    combined = {
        "hidden_width": targets[0] if len(targets) == 1 else None,
        "hidden_widths": targets,
        "teacher_hidden_width": teacher_width,
        "primary_metric": str(diag_cfg.get("embedding_primary_metric", "raw_replacement_loss")),
        "passed": all(summary.get("passed") is True for summary in summaries),
        "realization_passed": all(
            summary.get("realization_passed") is True for summary in summaries
        ),
        "rows": [row for summary in summaries for row in summary.get("rows", ())],
        "cases": summaries,
        "retained_activation_checkpoints": [
            summary.get("retained_activation_checkpoint") for summary in summaries
        ],
    }
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / "hidden_width_diagnostic_summary.json").write_text(
        json.dumps(canonicalize(combined), indent=2, sort_keys=True) + "\n"
    )
    return combined


def _run_hidden_width_diagnostic_at_width(
    hydra_cfg: Any,
    *,
    descriptor,
    teacher_dir: Path,
    block_configs: list[BlockConfig],
    puzzle_dir: Path,
    artifacts_dir: Path,
    diag_cfg: dict[str, Any],
    teacher_width: int,
    width: int,
    alignment: int,
) -> dict[str, Any] | None:

    from ..plugins.automodel.solution_launch import launch_score_solutions_automodel
    from ..pruning.materialize import materialize_hidden_width_checkpoint

    root = puzzle_dir / "diagnostics" / "hidden_width" / f"width-{width:04d}"
    activation_parent = root / "parents" / "activation_sorted"
    reverse_parent = root / "parents" / "reverse_sorted"
    reverse_hidden_logs = root / "reverse_activation_logs"
    activations_log_dir = _activations_log_dir({"puzzle_dir": str(puzzle_dir)}, hydra_cfg)
    sort_cfg = _get(hydra_cfg, "sort", {})
    if _diagnostic_checkpoint_needs_rebuild(activation_parent):
        if dist.is_master() and activation_parent.exists():
            shutil.rmtree(activation_parent)
        dist.barrier()
        build_sorted_teacher(
            teacher_dir,
            activations_log_dir,
            activation_parent,
            descriptor,
            deferred_axes=tuple(_get(sort_cfg, "deferred_axes", ()) or ()),
            mamba_state_score_key=str(
                _get(sort_cfg, "mamba_state_score_key", "ssm_channel_contrib")
            ),
            embedding_widths=(width,),
        )
    dist.barrier()
    if _diagnostic_checkpoint_needs_rebuild(reverse_parent):
        if dist.is_master():
            if reverse_parent.exists():
                shutil.rmtree(reverse_parent)
            _write_transformed_activation_logs(
                activations_log_dir,
                reverse_hidden_logs,
                method="reverse",
                seed=int(diag_cfg.get("seed", 1234)),
                axis="__all__",
            )
        dist.barrier()
        build_sorted_teacher(
            teacher_dir,
            reverse_hidden_logs,
            reverse_parent,
            descriptor,
            deferred_axes=tuple(_get(sort_cfg, "deferred_axes", ()) or ()),
            mamba_state_score_key=str(
                _get(sort_cfg, "mamba_state_score_key", "ssm_channel_contrib")
            ),
            embedding_widths=(width,),
        )
    dist.barrier()

    solution = _identity_width_solution(block_configs, width)
    rows = []
    parallel = _diagnostic_parallel(hydra_cfg, diag_cfg)
    runtime_sources = {
        "original": teacher_dir,
        "activation": activation_parent,
        "reverse": reverse_parent,
    }
    for role, checkpoint in runtime_sources.items():
        role_dir = root / role
        solutions_path = role_dir / "single_sequence_replacement_solutions.json"
        output_dir = role_dir / "single_sequence_replacement_solutions--validation"
        if dist.is_master():
            role_dir.mkdir(parents=True, exist_ok=True)
            solutions_path.write_text(json.dumps([solution], indent=2, sort_keys=True) + "\n")
        dist.barrier()
        cfg = OmegaConf.create(_diagnostic_scoring_container(hydra_cfg))
        OmegaConf.set_struct(cfg, False)
        cfg.puzzle_dir = str(puzzle_dir)
        cfg.scoring.teacher_dir = str(teacher_dir)
        cfg.scoring.target_teacher_dir = str(teacher_dir)
        cfg.scoring.source_checkpoint_dir = str(checkpoint)
        cfg.scoring.solutions_path = str(solutions_path)
        cfg.scoring.output_dir = str(output_dir)
        cfg.scoring.solutions_to_validate = None
        cfg.scoring.score_source_baseline = False
        cfg.scoring.zero_pad_hidden_to_teacher_width = True
        cfg.scoring.skip_existing_solutions = not bool(diag_cfg.get("force_rescore", False))
        if parallel:
            cfg.scoring.automodel.parallel = parallel
        for key in ("eval_samples", "micro_batch_size", "block_size"):
            if key in diag_cfg:
                cfg.scoring[key] = diag_cfg[key]
        launch_score_solutions_automodel(cfg)
        dist.barrier()
        if dist.is_master():
            result_path = output_dir / "solution_0.json"
            raw = json.loads(result_path.read_text())
            rows.append(
                {
                    "role": role,
                    "hidden_width": width,
                    "checkpoint_dir": str(checkpoint),
                    "result_path": str(result_path),
                    "execution": "runtime_slice",
                    "metrics": _hidden_width_result_metrics(raw),
                }
            )

    realized_parent = root / "realized" / "checkpoint"
    if dist.is_master():
        materialize_hidden_width_checkpoint(
            activation_parent,
            descriptor,
            width,
            realized_parent,
            alignment=alignment,
            overwrite=bool(diag_cfg.get("force_rescore", False)),
        )
    dist.barrier()
    realized_output = root / "realized" / "single_sequence_replacement_solutions--validation"
    realized_cfg = OmegaConf.create(_diagnostic_scoring_container(hydra_cfg))
    OmegaConf.set_struct(realized_cfg, False)
    realized_cfg.puzzle_dir = str(puzzle_dir)
    realized_cfg.scoring.teacher_dir = str(teacher_dir)
    realized_cfg.scoring.target_teacher_dir = str(teacher_dir)
    realized_cfg.scoring.source_checkpoint_dir = str(realized_parent)
    realized_cfg.scoring.output_dir = str(realized_output)
    realized_cfg.scoring.baseline_only = True
    realized_cfg.scoring.zero_pad_hidden_to_teacher_width = True
    if parallel:
        realized_cfg.scoring.automodel.parallel = parallel
    realized_cfg.scoring.baseline_payload = {
        "axis": "hidden_width",
        "target_value": width,
        "execution": "physical_realization",
    }
    for key in ("eval_samples", "micro_batch_size", "block_size"):
        if key in diag_cfg:
            realized_cfg.scoring[key] = diag_cfg[key]
    launch_score_solutions_automodel(realized_cfg)
    dist.barrier()
    if dist.is_master():
        result_path = realized_output / "sliced_teacher.json"
        raw = json.loads(result_path.read_text())
        rows.append(
            {
                "role": "realized",
                "hidden_width": width,
                "checkpoint_dir": str(realized_parent),
                "result_path": str(result_path),
                "execution": "physical_realization",
                "metrics": _hidden_width_result_metrics(raw),
            }
        )

    summary = None
    if dist.is_master():
        by_role = {row["role"]: row for row in rows}
        metric = str(diag_cfg.get("embedding_primary_metric", "raw_replacement_loss"))
        values = {role: by_role[role]["metrics"].get(metric) for role in by_role}
        if any(value is None or not math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"hidden-width diagnosis has invalid {metric}: {values}")
        tolerance = float(diag_cfg.get("comparison_tolerance", 0.0))
        verdict = _hidden_width_ranking_verdict(
            values,
            tolerance=tolerance,
            require_beats_random=bool(diag_cfg.get("require_beats_random", True)),
        )
        activation_value = values["activation"]
        realized_value = values["realized"]
        realization_delta = abs(activation_value - realized_value)
        realization_tolerance = _hidden_width_realization_tolerance(diag_cfg, metric)
        realization_passed = realization_delta <= realization_tolerance
        summary = {
            "hidden_width": width,
            "teacher_hidden_width": teacher_width,
            "primary_metric": metric,
            **verdict,
            "passed": bool(verdict["passed"] and realization_passed),
            "realization_delta": realization_delta,
            "realization_passed": realization_passed,
            "realization_tolerance": realization_tolerance,
            "rows": rows,
            "retained_activation_checkpoint": str(activation_parent),
        }
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        summary_path = artifacts_dir / "hidden_width_diagnostic_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        if summary["passed"] and bool(diag_cfg.get("cleanup_physical_checkpoints", True)):
            shutil.rmtree(realized_parent, ignore_errors=True)
    dist.barrier()
    return summary


def _write_hidden_only_diagnostic_artifacts(
    *,
    artifacts_dir: Path,
    temporary_root: Path,
    hidden_width_summary: dict[str, Any],
    cleanup_reverse: bool,
) -> dict[str, Any]:
    """Finalize a global-width-only diagnosis without an empty block sweep."""

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "status": "complete",
        "axes": ["hidden_width"],
        "rows": [],
        "parent_sweep": {"status": "not_applicable"},
        "hidden_width": hidden_width_summary,
        "baseline": {
            "method_key": "random",
            "selection_basis": "original_order_prefix",
            "is_seeded_random_permutation": False,
        },
    }
    (artifacts_dir / "activation_diagnostic_summary.json").write_text(
        json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
    )

    primary_metric = str(hidden_width_summary.get("primary_metric", "raw_replacement_loss"))
    table_lines = [
        "# Hidden-width activation diagnosis",
        "",
        f"width: {hidden_width_summary.get('hidden_width')}",
        "",
        f"| ordering | {primary_metric} |",
        "| --- | ---: |",
    ]
    csv_rows = []
    for row in hidden_width_summary.get("rows", []):
        value = (row.get("metrics") or {}).get(primary_metric)
        table_lines.append(f"| {row.get('role')} | {value} |")
        csv_rows.append((row.get("role"), primary_metric, value))
    table_lines.extend(
        [
            "",
            f"pass: {'yes' if hidden_width_summary.get('passed') else 'no'}",
        ]
    )
    (artifacts_dir / "activation_diagnostic_table.md").write_text("\n".join(table_lines) + "\n")
    with (artifacts_dir / "activation_diagnostic_scores.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("role", "metric", "value"))
        writer.writerows(csv_rows)

    if cleanup_reverse:
        shutil.rmtree(temporary_root, ignore_errors=True)
    cleanup = {
        "reverse_checkpoint_removed": not temporary_root.exists(),
        "reverse_activation_logs_removed": not temporary_root.exists(),
        "retained_activation_sorted_checkpoint": hidden_width_summary.get(
            "retained_activation_checkpoint"
        ),
        "cleanup_requested": bool(cleanup_reverse),
    }
    (artifacts_dir / "diagnostic_cleanup.json").write_text(
        json.dumps(canonicalize(cleanup), indent=2, sort_keys=True) + "\n"
    )
    return summary


def _hidden_only_diagnostic_ready(
    *,
    axes: list[str],
    hidden_width_summary: dict[str, Any] | None,
    is_master: bool,
) -> bool:
    """Validate the rank-asymmetric result of the dedicated width sweep."""

    if axes != ["hidden_width"]:
        raise RuntimeError(f"activation diagnosis produced no block-level solutions: axes={axes}")
    if is_master and hidden_width_summary is None:
        raise RuntimeError("rank 0 is missing the hidden-width diagnosis verdict")
    return True


def _diagnostic_checkpoint_needs_rebuild(path: Path) -> bool:
    """Reject partial sorted parents left by an interrupted distributed write."""

    return not _hf_checkpoint_complete(path, required_files=("sorted_permutations.json",))


def _passes_for_axis(axis: str) -> tuple[str, ...]:
    mapping = {
        "ffn_intermediate": ("ffn_iterative",),
        "intermediate_size": ("ffn_iterative",),
        "kv_groups": ("attention_grouped",),
        "kv_heads": ("attention_grouped",),
        "num_kv_heads": ("attention_grouped",),
        "query_heads": ("attention_grouped",),
        "num_query_heads": ("attention_grouped",),
        "q_heads_per_group": ("attention_grouped",),
        "gdn_key_groups": ("gdn_activation",),
        "gdn_value_heads_per_group": ("gdn_activation",),
        "gdn_key_head_dim": ("gdn_activation",),
        "gdn_value_head_dim": ("gdn_activation",),
        "moe_experts": ("moe_expert_removal",),
        "num_experts": ("moe_expert_removal",),
        "moe_expert_intermediate": ("moe_expert_intermediate",),
        "expert_intermediate_size": ("moe_expert_intermediate",),
        "moe_shared_expert_intermediate": ("moe_shared_expert_intermediate",),
        "shared_expert_intermediate_size": ("moe_shared_expert_intermediate",),
        "moe_latent_dim": ("moe_latent",),
        "latent_dim": ("moe_latent",),
        "moe_top_k": (),
        "top_k": (),
        "mamba_heads": ("mamba_head_and_dim",),
        "num_heads": ("mamba_head_and_dim",),
        "mamba_head_dim": ("mamba_head_and_dim",),
    }
    return mapping.get(axis, (axis,))


def _metric_avg(raw: dict[str, Any], metric: str) -> float | None:
    value = raw.get(metric)
    if isinstance(value, dict) and "avg" in value:
        try:
            return float(value["avg"])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _hidden_width_result_metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    """Keep the full solution metric surface for model-global width diagnosis."""

    metric_names = (
        *_PRIMARY_METRICS,
        # Retain the legacy aliases for old report consumers.  The explicit
        # token_accuracy_*_consistency fields above are the canonical names.
        "top_1_logit_agreement",
        "top_5_logit_agreement",
        "top_10_logit_agreement",
    )
    return {metric: _metric_avg(raw, metric) for metric in metric_names}


def _merge_reused_sort_equivalence(
    existing: dict[str, Any], reuse: dict[str, Any]
) -> dict[str, Any]:
    """Add parent-sweep provenance without discarding an earlier rich diagnosis."""

    merged = dict(existing)
    merged.update(reuse)
    return merged


def _extract_rows(method: str, output_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for result_path in sorted(output_dir.glob("solution_*.json")):
        raw = json.loads(result_path.read_text())
        solution = raw.get("puzzle_solution") or {}
        diag = solution.get("diagnostic") or {}
        row = {
            "method": diag.get("report_method", method),
            "parent_role": diag.get("parent_role"),
            "selection_basis": diag.get("selection_basis"),
            "solution_file": str(result_path),
            "solution_id": raw.get("i_solution"),
            "axis": diag.get("axis"),
            "combo": diag.get("combo"),
            "layer_idx": diag.get("layer_idx"),
            "teacher_value": diag.get("teacher_value"),
            "target_value": diag.get("target_value"),
            "ratio": diag.get("ratio"),
            "kept_kv_groups": _json_cell(diag.get("kept_kv_groups")),
            "removed_kv_groups": _json_cell(diag.get("removed_kv_groups")),
            "kv_group_order": _json_cell(diag.get("kv_group_order")),
            "kept_query_heads_per_group": _json_cell(diag.get("kept_query_heads_per_group")),
            "removed_query_heads_per_group": _json_cell(diag.get("removed_query_heads_per_group")),
            "query_head_order_per_group": _json_cell(diag.get("query_head_order_per_group")),
            "kept_units": _json_cell(diag.get("kept_units")),
            "removed_units": _json_cell(diag.get("removed_units")),
            "unit_order": _json_cell(diag.get("unit_order")),
            "changed_layers": _json_cell(diag.get("changed_layers")),
            "num_changed_layers": diag.get("num_changed_layers"),
            "ranking_applicable": diag.get("ranking_applicable", True),
            "ranking_reason": diag.get("ranking_reason"),
        }
        for metric in _PRIMARY_METRICS:
            row[metric] = _metric_avg(raw, metric)
        rows.append(row)
    return rows


def _collect_existing_rows(
    diag_root: Path,
    *,
    axes: set[str] | None = None,
    methods: set[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_dir in sorted(diag_root.rglob("single_sequence_replacement_solutions--validation")):
        rel = output_dir.relative_to(diag_root).parts
        if len(rel) < 3 or rel[0] == "transformed_logs":
            continue
        method_dir = output_dir.parent
        axis = rel[0]
        method = rel[1]
        if axes is not None and axis not in axes:
            continue
        if methods is not None and method not in methods:
            continue
        rows.extend(_extract_rows(method, output_dir))
    return rows


def _primary_metric(rows: list[dict[str, Any]], requested: str | None = None) -> str:
    candidates = (requested,) if requested else _PRIMARY_METRICS
    for metric in candidates:
        if metric and any(
            row.get(metric) is not None and math.isfinite(row[metric]) for row in rows
        ):
            return metric
    return _PRIMARY_METRICS[0]


def _write_summary(
    rows: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    requested_metric: str | None,
    comparison_tolerance: float = 0.0,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "activation_diagnostic_scores.csv"
    metric = _primary_metric(rows, requested_metric)
    fieldnames = [
        "axis",
        "layer_idx",
        "ratio",
        "teacher_value",
        "target_value",
        "method",
        "parent_role",
        "selection_basis",
        "ranking_applicable",
        "ranking_reason",
        "kept_kv_groups",
        "removed_kv_groups",
        "kv_group_order",
        "kept_query_heads_per_group",
        "removed_query_heads_per_group",
        "query_head_order_per_group",
        "kept_units",
        "removed_units",
        "unit_order",
        "changed_layers",
        "num_changed_layers",
        *_PRIMARY_METRICS,
        "solution_id",
        "solution_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda x: (
                str(x.get("axis")),
                int(x.get("layer_idx") or -1),
                float(x.get("ratio") or -1),
                str(x.get("method")),
            ),
        ):
            writer.writerow({key: row.get(key) for key in fieldnames})

    grouped: dict[tuple[Any, Any, Any], dict[str, float]] = {}
    selections: dict[tuple[Any, Any, Any], dict[str, str | None]] = {}
    for row in rows:
        if row.get("ranking_applicable") is False:
            continue
        value = row.get(metric)
        if value is None or not math.isfinite(value):
            continue
        key = (row.get("axis"), row.get("layer_idx"), row.get("target_value"))
        grouped.setdefault(key, {})[str(row.get("method"))] = float(value)
        selections.setdefault(key, {})[str(row.get("method"))] = row.get("kept_kv_groups")

    table_rows = []
    for (axis, layer_idx, target), values in sorted(
        grouped.items(), key=lambda item: (str(item[0][0]), int(item[0][1]), int(item[0][2]))
    ):
        activation = values.get("activation")
        random = values.get("random")
        negative = values.get("reverse", values.get("negative"))
        realized = values.get("realized")
        ranking_ok = (
            activation is not None
            and random is not None
            and negative is not None
            and activation <= random + comparison_tolerance
            and activation <= negative + comparison_tolerance
        )
        realization_ok = (
            realized is None
            or activation is not None
            and abs(realized - activation) <= comparison_tolerance
        )
        table_rows.append(
            {
                "axis": axis,
                "layer": layer_idx,
                "target": target,
                "activation": activation,
                "random": random,
                "negative": negative,
                "realized": realized,
                "random_minus_activation": None
                if activation is None or random is None
                else random - activation,
                "negative_minus_activation": None
                if activation is None or negative is None
                else negative - activation,
                "realized_minus_activation": (
                    None if activation is None or realized is None else realized - activation
                ),
                "expected_loss_order": ranking_ok,
                "realization_matches_runtime": realization_ok,
                "passed": ranking_ok and realization_ok,
                "ties_random": (
                    None
                    if activation is None or random is None
                    else abs(activation - random) <= comparison_tolerance
                ),
                "ties_reverse": (
                    None
                    if activation is None or negative is None
                    else abs(activation - negative) <= comparison_tolerance
                ),
                "activation_keep": selections.get((axis, layer_idx, target), {}).get("activation"),
                "random_keep": selections.get((axis, layer_idx, target), {}).get("random"),
                "negative_keep": selections.get((axis, layer_idx, target), {}).get("negative"),
            }
        )

    md_path = artifacts_dir / "activation_diagnostic_table.md"
    has_selection = any(
        row.get("activation_keep") or row.get("random_keep") or row.get("negative_keep")
        for row in table_rows
    )
    headers = [
        "axis",
        "layer",
        "target",
    ]
    if has_selection:
        headers.extend(["act_keep", "rand_keep", "neg_keep"])
    headers.extend(
        [
            "activation",
            "random",
            "negative",
            "realized",
            "rand-act",
            "neg-act",
            "real-act",
            "physical=runtime",
            "ok",
        ]
    )
    lines = [
        f"# Activation Diagnostic ({metric})",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table_rows:

        def fmt(value):
            if value is None:
                return ""
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        cells = [
            fmt(row["axis"]),
            fmt(row["layer"]),
            fmt(row["target"]),
        ]
        if has_selection:
            cells.extend(
                [
                    fmt(row.get("activation_keep")),
                    fmt(row.get("random_keep")),
                    fmt(row.get("negative_keep")),
                ]
            )
        cells.extend(
            [
                fmt(row["activation"]),
                fmt(row["random"]),
                fmt(row["negative"]),
                fmt(row["realized"]),
                fmt(row["random_minus_activation"]),
                fmt(row["negative_minus_activation"]),
                fmt(row["realized_minus_activation"]),
                fmt(row["realization_matches_runtime"]),
                fmt(row["passed"]),
            ]
        )
        lines.append("| " + " | ".join(cells) + " |")
    non_sortable_rows = [row for row in rows if row.get("ranking_applicable") is False]
    if non_sortable_rows:
        lines.extend(
            [
                "",
                "## Exact-target distortion for non-sortable axes",
                "",
                "| axis | layer | target | reason | " + " | ".join(_PRIMARY_METRICS) + " |",
                "| --- | --- | --- | --- | " + " | ".join(["---"] * len(_PRIMARY_METRICS)) + " |",
            ]
        )
        for row in sorted(
            non_sortable_rows,
            key=lambda item: (str(item.get("axis")), int(item.get("layer_idx") or -1)),
        ):
            values = [
                str(row.get("axis")),
                str(row.get("layer_idx")),
                str(row.get("target_value")),
                str(row.get("ranking_reason") or "not sortable"),
                *[
                    "" if row.get(metric_name) is None else f"{float(row[metric_name]):.6g}"
                    for metric_name in _PRIMARY_METRICS
                ],
            ]
            lines.append("| " + " | ".join(values) + " |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    metric_comparisons = {}
    for metric_name in _PRIMARY_METRICS:
        by_case: dict[tuple[Any, Any, Any], dict[str, float]] = {}
        for row in rows:
            if row.get("ranking_applicable") is False:
                continue
            value = row.get(metric_name)
            if value is None or not math.isfinite(value):
                continue
            key = (row.get("axis"), row.get("layer_idx"), row.get("target_value"))
            by_case.setdefault(key, {})[str(row.get("method"))] = float(value)
        higher_is_better = metric_name.startswith("token_accuracy_")
        comparisons = []
        for key, values in sorted(by_case.items(), key=lambda item: tuple(map(str, item[0]))):
            activation = values.get("activation")
            random_value = values.get("random")
            reverse_value = values.get("reverse", values.get("negative"))
            realized_value = values.get("realized")
            if activation is None or random_value is None or reverse_value is None:
                continue
            if higher_is_better:
                better_random = activation + comparison_tolerance >= random_value
                better_reverse = activation + comparison_tolerance >= reverse_value
            else:
                better_random = activation <= random_value + comparison_tolerance
                better_reverse = activation <= reverse_value + comparison_tolerance
            comparisons.append(
                {
                    "axis": key[0],
                    "layer_idx": key[1],
                    "target_value": key[2],
                    "activation": activation,
                    "random": random_value,
                    "reverse": reverse_value,
                    "realized": realized_value,
                    "activation_minus_random": activation - random_value,
                    "activation_minus_reverse": activation - reverse_value,
                    "beats_random": better_random,
                    "beats_reverse": better_reverse,
                    "ties_random": abs(activation - random_value) <= comparison_tolerance,
                    "ties_reverse": abs(activation - reverse_value) <= comparison_tolerance,
                    "realization_matches_runtime": (
                        realized_value is None
                        or abs(activation - realized_value) <= comparison_tolerance
                    ),
                }
            )
        metric_comparisons[metric_name] = {
            "higher_is_better": higher_is_better,
            "num_comparisons": len(comparisons),
            "beats_both": sum(
                int(item["beats_random"] and item["beats_reverse"]) for item in comparisons
            ),
            "comparisons": comparisons,
        }
    summary = {
        "primary_metric": metric,
        "num_scores": len(rows),
        "num_comparisons": len(table_rows),
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "table": table_rows,
        "rows": rows,
        "metric_comparisons": metric_comparisons,
        "comparison_tolerance": comparison_tolerance,
        "non_sortable": non_sortable_rows,
        "ranking_warnings": [
            {
                "axis": row["axis"],
                "layer_idx": row["layer"],
                "target_value": row["target"],
                "activation": row["activation"],
                "random": row["random"],
                "reverse": row["negative"],
            }
            for row in table_rows
            if not row["passed"]
        ],
    }
    (artifacts_dir / "activation_diagnostic_summary.json").write_text(
        json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
    )
    mprint("\n".join(lines[: min(len(lines), 80)]))
    return summary


def _validate_parent_sweep_checkpoint_loads(sweep_manifest: dict) -> None:
    """Require each parent checkpoint to be loaded at most once in this invocation."""

    for role, value in (sweep_manifest.get("checkpoint_loads") or {}).items():
        loads = int(value)
        if loads not in (0, 1):
            raise RuntimeError(
                f"fresh parent sweep loaded {role} more than once; loads={loads}"
            )


def _publish_parent_sweep_sanity(
    *,
    puzzle_dir: Path,
    parent_summary: dict[str, Any],
    hidden_width_summary: dict[str, Any] | None,
    diag_cfg: dict[str, Any],
) -> tuple[Path, Path]:
    """Publish scalable width and physical-equivalence summaries from one sweep."""

    tolerance = float(diag_cfg.get("physical_equivalence_tolerance", 1.0e-3))
    tolerance_overrides = {
        str(metric): float(value)
        for metric, value in dict(
            diag_cfg.get("physical_equivalence_tolerances") or {}
        ).items()
    }
    unknown_metrics = sorted(set(tolerance_overrides).difference(_PRIMARY_METRICS))
    if unknown_metrics:
        raise ValueError(
            "physical_equivalence_tolerances contains unknown metrics: "
            f"{unknown_metrics}"
        )
    invalid_tolerances = {
        metric: value
        for metric, value in tolerance_overrides.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid_tolerances:
        raise ValueError(
            "physical equivalence tolerances must be finite and non-negative: "
            f"{invalid_tolerances}"
        )
    metric_specs = {
        metric: MetricSpec(
            name=metric,
            direction="higher" if metric.startswith("token_accuracy_") else "lower",
            abs_tolerance=tolerance_overrides.get(metric, tolerance),
        )
        for metric in _PRIMARY_METRICS
    }
    width_summary, slicing_summary, axes = aggregate_parent_sweep_sanity(
        parent_summary,
        hidden_width_summary,
        metric_specs=metric_specs,
    )
    provenance = {
        "backend": "distributed_parent_sweep",
        "axes": axes,
        "physical_equivalence_tolerance": tolerance,
        "physical_equivalence_tolerances": tolerance_overrides,
    }
    paths = []
    for stage, payload in (
        ("width_sanity", width_summary),
        ("slicing_sanity", slicing_summary),
    ):
        payload["passed"] = not payload.get("findings")
        payload["verdict"] = "passed" if payload["passed"] else "warning"
        payload["provenance"] = provenance
        output = puzzle_dir / "artifacts" / stage / "summary.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        paths.append(output)
    return paths[0], paths[1]


def _nested_update(mapping: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    target = mapping
    for key in parts[:-1]:
        if not hasattr(target, key) or getattr(target, key) is None:
            target[key] = {}
        target = getattr(target, key)
    target[parts[-1]] = value


def _diagnostic_scoring_container(hydra_cfg: Any) -> dict[str, Any]:
    data = OmegaConf.to_container(hydra_cfg, resolve=True)
    pruning = data.get("pruning") if isinstance(data, dict) else None
    if isinstance(pruning, dict):
        # Runtime Hydra instantiates activation pass descriptors/mixins for the scorer.
        # The diagnostic replace-one-block scorer does not consume these fields, and
        # OmegaConf cannot round-trip arbitrary Python objects back through create().
        for key in ("activation_passes", "pruning_mixin", "hook_class"):
            pruning.pop(key, None)
    return data


def _diagnostic_parallel(hydra_cfg: Any, diag_cfg: Any) -> dict[str, Any] | None:
    """Resolve the diagnostic stage mesh, falling back to replacement scoring."""

    candidates = (
        _get(_get(diag_cfg, "automodel", {}), "parallel", None),
        _get(_get(_get(hydra_cfg, "scoring", {}), "automodel", {}), "parallel", None),
    )
    for parallel in candidates:
        if parallel:
            if OmegaConf.is_config(parallel):
                return dict(OmegaConf.to_container(parallel, resolve=True))
            return dict(parallel)
    return None


def _scoring_cfg_for_method(
    hydra_cfg: Any,
    *,
    method_dir: Path,
    scoring_output_dir: Path,
    parallel: dict[str, Any] | None,
    source_checkpoint_dir: Path | str | None = None,
    target_teacher_dir: Path | str | None = None,
) -> Any:
    cfg = OmegaConf.create(_diagnostic_scoring_container(hydra_cfg))
    OmegaConf.set_struct(cfg, False)
    cfg.puzzle_dir = str(method_dir)
    cfg.teacher_dir = str(hydra_cfg.teacher_dir)
    cfg.replacement_library_path = str(method_dir / "replacement_library.json")
    cfg.scoring.replacement_library_path = str(method_dir / "replacement_library.json")
    cfg.scoring.solutions_path = method_dir / "single_sequence_replacement_solutions.json"
    cfg.scoring.output_dir = str(scoring_output_dir)
    cfg.scoring.skip_existing_solutions = False
    cfg.scoring.solutions_to_validate = None
    cfg.scoring.teacher_dir = hydra_cfg.teacher_dir
    if source_checkpoint_dir is not None:
        cfg.scoring.source_checkpoint_dir = str(source_checkpoint_dir)
    if target_teacher_dir is not None:
        cfg.scoring.target_teacher_dir = str(target_teacher_dir)
    if parallel:
        cfg.scoring.automodel.parallel = parallel
    return cfg


def _build_diagnostic_sorted_parent(
    *,
    teacher_dir: Path,
    activations_log_dir: Path,
    transformed_log_dir: Path,
    sorted_dir: Path,
    descriptor: Any,
    method: str,
    seed: int,
    selected_passes: tuple[str, ...],
    axis: str,
    layer_idx: int,
    embedding_widths: tuple[int, ...],
) -> None:
    """Build one diagnostic parent without letting non-master ranks race ahead.

    ``build_sorted_teacher`` uses distributed collectives when a process group
    exists, so every rank must enter it.  The transformed-log write remains a
    rank-zero filesystem operation and is published before the shared sort.
    """

    if dist.is_master():
        _write_transformed_activation_logs(
            activations_log_dir,
            transformed_log_dir,
            method=method,
            seed=seed,
            selected_passes=selected_passes,
            axis=axis,
            target_layers={int(layer_idx)},
        )
    dist.barrier()
    build_sorted_teacher(
        teacher_dir,
        transformed_log_dir,
        sorted_dir,
        descriptor,
        embedding_widths=embedding_widths,
    )
    dist.barrier()


def _scoring_cfg_for_parent_sweep(
    hydra_cfg: Any,
    *,
    puzzle_dir: Path,
    teacher_dir: Path,
    parent_specs: list[dict[str, Any]],
    manifest_path: Path,
    tolerances: dict[str, float],
    parallel: dict[str, Any] | None,
    force_rescore: bool,
) -> Any:
    cfg = OmegaConf.create(_diagnostic_scoring_container(hydra_cfg))
    OmegaConf.set_struct(cfg, False)
    cfg.puzzle_dir = str(puzzle_dir)
    cfg.teacher_dir = str(teacher_dir)
    cfg.scoring.teacher_dir = str(teacher_dir)
    cfg.scoring.parent_sweeps = parent_specs
    cfg.scoring.parent_sweep_manifest = str(manifest_path)
    cfg.scoring.parent_equivalence_tolerances = tolerances
    cfg.scoring.force_rescore = bool(force_rescore)
    cfg.scoring.solutions_path = parent_specs[0]["solutions_path"]
    cfg.scoring.output_dir = parent_specs[0]["output_dir"]
    cfg.scoring.skip_existing_solutions = not bool(force_rescore)
    cfg.scoring.solutions_to_validate = None
    if parallel:
        cfg.scoring.automodel.parallel = parallel
    return cfg


def _activation_diagnostic_parent_sweep(
    config: dict[str, Any],
    manifest: StageManifest,
    hydra_cfg: Any,
    diag_cfg: dict[str, Any],
):
    methods = [
        str(method) for method in diag_cfg.get("methods", ["activation", "random", "reverse"])
    ]
    required_methods = {"activation", "random", "reverse"}
    if set(methods) != required_methods:
        raise ValueError(
            "parent-sweep diagnosis requires methods activation, random, and reverse; "
            f"got {methods}"
        )
    ratios = [float(ratio) for ratio in diag_cfg.get("ratios", [0.25, 0.5, 0.75])]
    target_values = dict(diag_cfg.get("target_values") or {})
    non_sortable_axes = {str(axis) for axis in diag_cfg.get("non_sortable_axes", ())}
    layer_count = int(diag_cfg.get("layer_count", 5))
    layer_indices = _as_int_list(diag_cfg.get("layer_indices"))
    layer_selection = str(diag_cfg.get("layer_selection", "spread"))
    layer_seed = int(diag_cfg.get("layer_seed", diag_cfg.get("seed", 1234)))
    seed = int(diag_cfg.get("seed", 1234))
    requested_metric = diag_cfg.get("primary_metric")
    axes = _enabled_diagnostic_axes(config, diag_cfg)
    if bool(diag_cfg.get("one_case_per_axis", False)):
        layer_count = 1
        layer_selection = "random"
        ratios = [0.5]
        target_values = _representative_axis_targets(config, axes, target_values)
    elif int(diag_cfg.get("target_count_per_axis", 0) or 0) > 0:
        target_values.update(
            _near_teacher_axis_targets(
                config,
                axes,
                count=int(diag_cfg["target_count_per_axis"]),
            )
        )
    deferred_axes = tuple(str(axis) for axis in diag_cfg.get("deferred_axes", ()))
    sort_cfg = _get(hydra_cfg, "sort", {})
    mamba_state_score_key = str(
        diag_cfg.get("mamba_state_score_key")
        or _get(sort_cfg, "mamba_state_score_key", "ssm_channel_contrib")
    )
    embedding_widths = tuple(_get(_get(hydra_cfg, "embedding_pruning", {}), "widths", ()) or ())

    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    experiment_id = str(diag_cfg.get("experiment_id") or "").strip()
    diag_root_name = "activation_scores" + (f"_{experiment_id}" if experiment_id else "")
    artifacts_name = "activation_diagnostic" + (f"_{experiment_id}" if experiment_id else "")
    diag_root = puzzle_dir / "diagnostics" / diag_root_name
    artifacts_dir = puzzle_dir / "artifacts" / artifacts_name
    parent_root = diag_root / "parent_sweeps"
    temporary_root = diag_root / "temporary"
    configured_reverse_dir = diag_cfg.get("reverse_checkpoint_dir")
    configured_reverse_logs = diag_cfg.get("reverse_activation_logs_dir")
    reverse_dir = (
        Path(configured_reverse_dir)
        if configured_reverse_dir
        else temporary_root / "reverse_sorted_teacher"
    )
    reverse_logs = (
        Path(configured_reverse_logs)
        if configured_reverse_logs
        else temporary_root / "reverse_activation_logs"
    )
    reverse_is_temporary = not bool(configured_reverse_dir)
    sorted_dir = puzzle_dir / "ckpts" / "sorted_teacher"
    load_manifest_path = diag_root / "parent_sweep_manifest.json"
    force_rescore = bool(diag_cfg.get("force_rescore", False))
    overwrite = bool(diag_cfg.get("overwrite", True))
    # Keep durable parent-sweep scores/realizations across orchestrator retries
    # unless the caller forces a rebuild.  Fresh runs (no manifest) still honor
    # overwrite=True and clear stale diagnostics.
    preserve_progress = load_manifest_path.is_file() and not force_rescore

    with _distributed(hydra_cfg):
        if dist.is_master() and overwrite and not preserve_progress:
            if diag_root.exists():
                shutil.rmtree(diag_root)
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
        dist.barrier()

        descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
        teacher_dir = _teacher_dir(config, hydra_cfg)
        activations_log_dir = _activations_log_dir(config, hydra_cfg)
        teacher_config = load_model_config(
            teacher_dir,
            trust_remote_code=descriptor.requires_trust_remote_code(),
        )
        lm = descriptor.get_language_model_config(teacher_config)
        block_configs = list(maybe_cast_block_configs(teacher_config.block_configs))
        head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // lm.num_attention_heads)

        needs_sorted_parent = _diagnostic_checkpoint_needs_rebuild(sorted_dir)
        if needs_sorted_parent:
            if dist.is_master():
                if sorted_dir.exists():
                    shutil.rmtree(sorted_dir)
                mprint(
                    "[activation_diagnostic] building one global activation-sorted parent -> "
                    f"{sorted_dir}"
                )
            dist.barrier()
            build_sorted_teacher(
                teacher_dir,
                activations_log_dir,
                sorted_dir,
                descriptor,
                deferred_axes=deferred_axes,
                mamba_state_score_key=mamba_state_score_key,
                embedding_widths=embedding_widths,
            )
        dist.barrier()
        if _diagnostic_checkpoint_needs_rebuild(sorted_dir):
            raise FileNotFoundError(f"global activation-sorted parent is incomplete: {sorted_dir}")

        needs_reverse_parent = _diagnostic_checkpoint_needs_rebuild(reverse_dir)
        if needs_reverse_parent:
            if dist.is_master():
                if reverse_dir.exists():
                    shutil.rmtree(reverse_dir)
                mprint(
                    "[activation_diagnostic] building one temporary global reverse parent -> "
                    f"{reverse_dir}"
                )
                _write_transformed_activation_logs(
                    activations_log_dir,
                    reverse_logs,
                    method="reverse",
                    seed=seed,
                    axis="__all__",
                )
            dist.barrier()
            build_sorted_teacher(
                teacher_dir,
                reverse_logs,
                reverse_dir,
                descriptor,
                deferred_axes=deferred_axes,
                mamba_state_score_key=mamba_state_score_key,
                embedding_widths=embedding_widths,
            )
        dist.barrier()
        if _diagnostic_checkpoint_needs_rebuild(reverse_dir):
            raise FileNotFoundError(f"global reverse-sorted parent is incomplete: {reverse_dir}")

        hidden_width_summary = _run_hidden_width_diagnostic(
            hydra_cfg,
            descriptor=descriptor,
            teacher_dir=teacher_dir,
            sorted_dir=sorted_dir,
            reverse_dir=reverse_dir,
            block_configs=block_configs,
            puzzle_dir=puzzle_dir,
            artifacts_dir=artifacts_dir,
            diag_cfg=diag_cfg,
        )

        planned: list[dict[str, Any]] = []
        for axis in axes:
            _, axis_solutions = _diagnostic_solutions(
                block_configs,
                axes=[axis],
                ratios=ratios,
                target_values=target_values,
                layer_count=layer_count,
                layer_indices=layer_indices,
                layer_selection=layer_selection,
                layer_seed=layer_seed,
            )
            for solution in axis_solutions:
                diagnostic = solution.get("diagnostic") or {}
                diagnostic["ranking_applicable"] = axis not in non_sortable_axes
                if axis in non_sortable_axes:
                    diagnostic["ranking_reason"] = (
                        "behavioral variant has no static channel ordering; "
                        "exact-target distortion is reported without ranking"
                    )
                solution["diagnostic"] = diagnostic
                candidate = solution.get("single_sequence_replacement")
                if isinstance(candidate, dict):
                    candidate["diagnostic"] = diagnostic
                planned.append(solution)

        if not planned:
            _hidden_only_diagnostic_ready(
                axes=axes,
                hidden_width_summary=hidden_width_summary,
                is_master=dist.is_master(),
            )
            if dist.is_master():
                parent_sweep = {
                    "version": 1,
                    "status": "not_applicable",
                    "reason": "hidden_width is model-global and uses the dedicated three-parent sweep",
                    "checkpoint_loads": {},
                }
                load_manifest_path.parent.mkdir(parents=True, exist_ok=True)
                load_manifest_path.write_text(
                    json.dumps(parent_sweep, indent=2, sort_keys=True) + "\n"
                )
                _write_hidden_only_diagnostic_artifacts(
                    artifacts_dir=artifacts_dir,
                    temporary_root=temporary_root,
                    hidden_width_summary=hidden_width_summary,
                    cleanup_reverse=bool(diag_cfg.get("cleanup_reverse_on_success", True)),
                )
            dist.barrier()
            return complete_stage(
                config,
                manifest,
                outputs={
                    "diagnostic_root": str(diag_root),
                    "artifacts_dir": str(artifacts_dir),
                    "methods": methods,
                    "axes": axes,
                    "ratios": ratios,
                    "target_values": target_values,
                    "non_sortable_axes": sorted(non_sortable_axes),
                    "summary_path": str(artifacts_dir / "activation_diagnostic_summary.json"),
                    "table_path": str(artifacts_dir / "activation_diagnostic_table.md"),
                    "csv_path": str(artifacts_dir / "activation_diagnostic_scores.csv"),
                    "parent_sweep_manifest": str(load_manifest_path),
                    "cleanup_path": str(artifacts_dir / "diagnostic_cleanup.json"),
                    "hidden_width_summary_path": str(
                        artifacts_dir / "hidden_width_diagnostic_summary.json"
                    ),
                },
            )

        realized_definitions: list[tuple[str, str, Path, bool, str, list[dict[str, Any]]]] = []
        if bool(diag_cfg.get("physical_realization", False)):
            from ..pruning.materialize import materialize_checkpoint_from_sorted

            for case_idx, solution in enumerate(planned):
                realized_dir = parent_root / "realized" / f"case_{case_idx:04d}" / "checkpoint"
                if dist.is_master():
                    child_config = copy.deepcopy(teacher_config)
                    descriptor.set_block_configs(
                        child_config,
                        maybe_cast_block_configs(solution["block_configs"]),
                    )
                    materialize_checkpoint_from_sorted(
                        sorted_dir,
                        solution["chosen_replacements"],
                        descriptor,
                        child_config,
                        realized_dir,
                        overwrite=bool(diag_cfg.get("force_rescore", False)),
                        solution_identity=stable_hash(
                            solution, prefix="width_diagnostic_realization"
                        ),
                    )
                dist.barrier()
                realized_definitions.append(
                    (
                        f"realized_{case_idx:04d}",
                        "realized",
                        realized_dir,
                        True,
                        "realized_baseline",
                        [solution],
                    )
                )

        parent_definitions = [
            ("original", "random", teacher_dir, True, "runtime_slice", planned),
            ("activation", "activation", sorted_dir, False, "runtime_slice", planned),
            ("reverse", "reverse", reverse_dir, False, "runtime_slice", planned),
            *realized_definitions,
        ]
        parent_specs: list[dict[str, Any]] = []
        parent_metadata: dict[str, Any] = {}
        for (
            role,
            report_method,
            checkpoint_dir,
            include_non_sortable,
            evaluation_mode,
            source_solutions,
        ) in parent_definitions:
            solutions = []
            for source_solution in source_solutions:
                source_diag = source_solution.get("diagnostic") or {}
                axis = str(source_diag.get("axis"))
                if axis in non_sortable_axes and not include_non_sortable:
                    continue
                solution = json.loads(json.dumps(canonicalize(source_solution)))
                diagnostic = solution.get("diagnostic") or {}
                diagnostic["parent_role"] = role
                diagnostic["report_method"] = (
                    "not_applicable" if axis in non_sortable_axes else report_method
                )
                solution["diagnostic"] = diagnostic
                candidate = solution.get("single_sequence_replacement")
                if isinstance(candidate, dict):
                    candidate["diagnostic"] = diagnostic
                solutions.append(solution)

            _annotate_solution_selections(
                solutions=solutions,
                teacher_block_configs=block_configs,
                sorted_teacher_dir=checkpoint_dir,
            )
            for solution in solutions:
                diagnostic = solution.get("diagnostic") or {}
                diagnostic.setdefault(
                    "selection_basis",
                    "original_order_prefix" if role == "original" else f"global_{role}_prefix",
                )
                solution["diagnostic"] = diagnostic
                candidate = solution.get("single_sequence_replacement")
                if isinstance(candidate, dict):
                    candidate["diagnostic"] = diagnostic

            method_dir = parent_root / role
            output_dir = method_dir / "single_sequence_replacement_solutions--validation"
            if dist.is_master():
                entries = _entries_for_solutions(block_configs, solutions)
                _, solutions_path = _write_library_and_solutions(
                    method_dir,
                    checkpoint_dir,
                    entries,
                    solutions,
                )
                parent_specs.append(
                    {
                        "role": role,
                        "checkpoint_dir": str(checkpoint_dir),
                        "solutions_path": str(solutions_path),
                        "output_dir": str(output_dir),
                        "hidden_basis_permuted": role != "original",
                        "evaluation_mode": evaluation_mode,
                        "skip_parent_equivalence": bool(
                            diag_cfg.get("reuse_sort_equivalence", False)
                            and role in {"activation", "reverse"}
                        ),
                    }
                )
                parent_metadata[role] = {
                    "checkpoint_dir": str(checkpoint_dir),
                    "solutions": len(solutions),
                    "report_method": report_method,
                }
            dist.barrier()

        if not dist.is_master():
            parent_specs = [
                {
                    "role": role,
                    "checkpoint_dir": str(checkpoint_dir),
                    "solutions_path": str(
                        parent_root / role / "single_sequence_replacement_solutions.json"
                    ),
                    "output_dir": str(
                        parent_root / role / "single_sequence_replacement_solutions--validation"
                    ),
                    "hidden_basis_permuted": role != "original",
                }
                for role, _, checkpoint_dir, _, evaluation_mode, _ in parent_definitions
            ]
            for parent, definition in zip(parent_specs, parent_definitions):
                parent["evaluation_mode"] = definition[4]
                parent["skip_parent_equivalence"] = bool(
                    diag_cfg.get("reuse_sort_equivalence", False)
                    and parent["role"] in {"activation", "reverse"}
                )
        dist.barrier()

        if dist.is_master():
            metadata = {
                "architecture": "single_load_parent_sweep_v1",
                "baseline": "original_order_prefix",
                "axes": axes,
                "non_sortable_axes": sorted(non_sortable_axes),
                "layer_count": layer_count,
                "layer_selection": layer_selection,
                "layer_seed": layer_seed,
                "target_values": target_values,
                "parents": parent_metadata,
                "reverse_checkpoint": str(reverse_dir),
                "reverse_logs": str(reverse_logs),
                "reverse_is_temporary": reverse_is_temporary,
            }
            diag_root.mkdir(parents=True, exist_ok=True)
            (diag_root / "diagnostic_metadata.json").write_text(
                json.dumps(canonicalize(metadata), indent=2, sort_keys=True) + "\n"
            )
        dist.barrier()

        parallel = _diagnostic_parallel(hydra_cfg, diag_cfg)
        tolerances = dict(diag_cfg.get("parent_equivalence_tolerances") or {})
        scoring_cfg = _scoring_cfg_for_parent_sweep(
            hydra_cfg,
            puzzle_dir=puzzle_dir,
            teacher_dir=teacher_dir,
            parent_specs=parent_specs,
            manifest_path=load_manifest_path,
            tolerances=tolerances,
            parallel=parallel,
            force_rescore=bool(diag_cfg.get("force_rescore", False)),
        )
        for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
            if key in diag_cfg:
                scoring_cfg.scoring[key] = diag_cfg[key]

        from ..plugins.automodel.solution_launch import launch_score_solution_parents_automodel

        launch_score_solution_parents_automodel(scoring_cfg)
        dist.barrier()

        summary = None
        if dist.is_master():
            rows = _collect_existing_rows(diag_root)
            sortable_solutions = sum(
                1
                for solution in planned
                if str((solution.get("diagnostic") or {}).get("axis")) not in non_sortable_axes
            )
            non_sortable_solutions = len(planned) - sortable_solutions
            sortable_methods = 4 if realized_definitions else 3
            expected_rows = sortable_solutions * sortable_methods + non_sortable_solutions * (
                2 if realized_definitions else 1
            )
            if len(rows) != expected_rows:
                raise RuntimeError(
                    "parent-sweep diagnosis row count mismatch: "
                    f"observed={len(rows)} expected={expected_rows}"
                )
            sweep_manifest = json.loads(load_manifest_path.read_text())
            if sweep_manifest.get("status") != "complete":
                raise RuntimeError(f"parent sweep did not complete: {sweep_manifest}")
            _validate_parent_sweep_checkpoint_loads(sweep_manifest)
            summary = _write_summary(
                rows,
                artifacts_dir,
                requested_metric=requested_metric,
                comparison_tolerance=float(diag_cfg.get("comparison_tolerance", 0.0)),
            )
            summary["parent_sweep"] = sweep_manifest
            summary["hidden_width"] = hidden_width_summary
            summary["baseline"] = {
                "method_key": "random",
                "selection_basis": "original_order_prefix",
                "is_seeded_random_permutation": False,
            }
            summary_path = artifacts_dir / "activation_diagnostic_summary.json"
            summary_path.write_text(
                json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
            )
            _publish_parent_sweep_sanity(
                puzzle_dir=puzzle_dir,
                parent_summary=summary,
                hidden_width_summary=hidden_width_summary,
                diag_cfg=diag_cfg,
            )

            activation_equivalence = (
                (sweep_manifest.get("parents") or {}).get("activation") or {}
            ).get("equivalence") or {}
            equivalence_findings = list(activation_equivalence.get("findings") or ())
            sort_passed = activation_equivalence.get("passed") is True
            sort_equivalence_dir = puzzle_dir / "artifacts" / "sort_sanity"
            sort_equivalence_dir.mkdir(parents=True, exist_ok=True)
            sort_summary_path = sort_equivalence_dir / "summary.json"
            existing_sort_summary = (
                json.loads(sort_summary_path.read_text()) if sort_summary_path.is_file() else {}
            )
            reuse_sort_summary = {
                "passed": sort_passed,
                "reused_parent_sweep": True,
                "teacher_dir": str(teacher_dir),
                "sorted_teacher_dir": str(sorted_dir),
                "equivalence": activation_equivalence,
                "findings": equivalence_findings,
                "parent_sweep_manifest": str(load_manifest_path),
            }
            sort_summary_path.write_text(
                json.dumps(
                    _merge_reused_sort_equivalence(
                        existing_sort_summary,
                        reuse_sort_summary,
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )

            cleanup_reverse = bool(diag_cfg.get("cleanup_reverse_on_success", True))
            if cleanup_reverse and reverse_is_temporary:
                shutil.rmtree(temporary_root)
            cleanup = {
                "reverse_checkpoint_removed": not reverse_dir.exists(),
                "reverse_activation_logs_removed": not reverse_logs.exists(),
                "retained_activation_sorted_checkpoint": str(sorted_dir),
                "cleanup_requested": cleanup_reverse,
            }
            (artifacts_dir / "diagnostic_cleanup.json").write_text(
                json.dumps(cleanup, indent=2, sort_keys=True) + "\n"
            )
            if realized_definitions and bool(
                diag_cfg.get("cleanup_physical_checkpoints", True)
            ):
                shutil.rmtree(parent_root / "realized", ignore_errors=True)
        dist.barrier()

    width_summary_path = puzzle_dir / "artifacts" / "width_sanity" / "summary.json"
    width_verdict = json.loads(width_summary_path.read_text(encoding="utf-8"))
    width_findings = list(width_verdict.get("findings") or ())
    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "diagnostic_root": str(diag_root),
            "artifacts_dir": str(artifacts_dir),
            "methods": methods,
            "axes": axes,
            "ratios": ratios,
            "target_values": target_values,
            "non_sortable_axes": sorted(non_sortable_axes),
            "summary_path": str(artifacts_dir / "activation_diagnostic_summary.json"),
            "table_path": str(artifacts_dir / "activation_diagnostic_table.md"),
            "csv_path": str(artifacts_dir / "activation_diagnostic_scores.csv"),
            "parent_sweep_manifest": str(load_manifest_path),
            "cleanup_path": str(artifacts_dir / "diagnostic_cleanup.json"),
            "hidden_width_summary_path": str(
                artifacts_dir / "hidden_width_diagnostic_summary.json"
            ),
            "width_summary_path": str(width_summary_path),
            "slicing_summary_path": str(
                puzzle_dir / "artifacts" / "slicing_sanity" / "summary.json"
            ),
        },
        verdict=SanityVerdict(
            passed=bool(width_verdict.get("passed", not width_findings)),
            findings=width_findings,
        ),
    )


def activation_diagnostic_stage(config: dict[str, Any], manifest: StageManifest):
    hydra_cfg = __import__(
        "modelopt.torch.puzzletron.pipeline_config",
        fromlist=["load_runtime_hydra_config"],
    ).load_runtime_hydra_config(config)
    diag_cfg = dict(config.get("width_sanity") or {})
    if not diag_cfg.get("enabled", True):
        return complete_stage(
            config,
            manifest,
            outputs={"skipped": True},
            status="skipped",
            message="Activation diagnostic is disabled.",
        )
    if bool(diag_cfg.get("single_load_parent_sweep", False)):
        return _activation_diagnostic_parent_sweep(
            config,
            manifest,
            hydra_cfg,
            diag_cfg,
        )

    methods = [
        str(method) for method in diag_cfg.get("methods", ["activation", "random", "negative"])
    ]
    ratios = [float(ratio) for ratio in diag_cfg.get("ratios", [0.25, 0.5, 0.75])]
    target_values = dict(diag_cfg.get("target_values") or {})
    non_sortable_axes = {str(axis) for axis in diag_cfg.get("non_sortable_axes", ())}
    layer_count = int(diag_cfg.get("layer_count", 5))
    layer_indices = _as_int_list(diag_cfg.get("layer_indices"))
    seed = int(diag_cfg.get("seed", 1234))
    requested_metric = diag_cfg.get("primary_metric")
    axes = _enabled_diagnostic_axes(config, diag_cfg)

    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    diagnostic_experiment_id = str(diag_cfg.get("experiment_id") or "").strip()
    diag_root_name = "activation_scores"
    artifacts_name = "activation_diagnostic"
    if diagnostic_experiment_id:
        diag_root_name = f"{diag_root_name}_{diagnostic_experiment_id}"
        artifacts_name = f"{artifacts_name}_{diagnostic_experiment_id}"
    diag_root = puzzle_dir / "diagnostics" / diag_root_name
    artifacts_dir = puzzle_dir / "artifacts" / artifacts_name
    with _distributed(hydra_cfg):
        if dist.is_master() and bool(diag_cfg.get("overwrite", True)) and diag_root.exists():
            shutil.rmtree(diag_root)
        dist.barrier()

        descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
        teacher_dir = _teacher_dir(config, hydra_cfg)
        activations_log_dir = _activations_log_dir(config, hydra_cfg)
        method_outputs: dict[str, str] = {}
        teacher_config = load_model_config(
            teacher_dir, trust_remote_code=descriptor.requires_trust_remote_code()
        )
        lm = descriptor.get_language_model_config(teacher_config)
        block_configs = list(maybe_cast_block_configs(teacher_config.block_configs))
        head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // lm.num_attention_heads)

        for axis in axes:
            selected_passes = _passes_for_axis(axis)
            _, planned_solutions = _diagnostic_solutions(
                block_configs,
                axes=[axis],
                ratios=ratios,
                target_values=target_values,
                layer_count=layer_count,
                layer_indices=layer_indices,
            )
            solutions_by_layer: dict[int, list[dict[str, Any]]] = {}
            for solution in planned_solutions:
                diagnostic = solution.get("diagnostic") or {}
                diagnostic["ranking_applicable"] = axis not in non_sortable_axes
                if axis in non_sortable_axes:
                    diagnostic["ranking_reason"] = (
                        "behavioral variant has no static channel ordering; "
                        "exact-target distortion is reported without ranking"
                    )
                solution["diagnostic"] = diagnostic
                candidate = solution.get("single_sequence_replacement")
                if isinstance(candidate, dict):
                    candidate["diagnostic"] = diagnostic
                layer_idx = int((solution.get("diagnostic") or {})["layer_idx"])
                solutions_by_layer.setdefault(layer_idx, []).append(solution)

            axis_methods = ["not_applicable"] if axis in non_sortable_axes else methods
            for method in axis_methods:
                for layer_idx, layer_solutions in sorted(solutions_by_layer.items()):
                    layer_name = f"layer_{layer_idx}"
                    method_dir = diag_root / axis / method / layer_name
                    transformed_log_dir = (
                        diag_root / "transformed_logs" / axis / method / layer_name
                    )
                    sorted_dir = method_dir / "ckpts" / "sorted_teacher"
                    _build_diagnostic_sorted_parent(
                        teacher_dir=teacher_dir,
                        activations_log_dir=activations_log_dir,
                        transformed_log_dir=transformed_log_dir,
                        sorted_dir=sorted_dir,
                        descriptor=descriptor,
                        method=method,
                        seed=seed,
                        selected_passes=selected_passes,
                        axis=axis,
                        layer_idx=layer_idx,
                        embedding_widths=tuple(
                            _get(
                                _get(hydra_cfg, "embedding_pruning", {}),
                                "widths",
                                (),
                            )
                            or ()
                        ),
                    )
                    if dist.is_master():
                        solutions = [
                            json.loads(json.dumps(canonicalize(solution)))
                            for solution in layer_solutions
                        ]
                        _annotate_solution_selections(
                            solutions=solutions,
                            teacher_block_configs=block_configs,
                            sorted_teacher_dir=sorted_dir,
                        )
                        entries = _entries_for_solutions(block_configs, solutions)
                        if axis in {"kv_groups", "kv_heads", "num_kv_heads"}:
                            for solution in solutions:
                                diag = solution.get("diagnostic") or {}
                                mprint(
                                    "[activation_diagnostic] kv selection "
                                    f"method={method} layer={diag.get('layer_idx')} "
                                    f"target={diag.get('target_value')} "
                                    f"order={diag.get('kv_group_order')} "
                                    f"keep={diag.get('kept_kv_groups')} "
                                    f"remove={diag.get('removed_kv_groups')} "
                                    f"changed_layers={diag.get('changed_layers')}"
                                )
                        _write_library_and_solutions(method_dir, sorted_dir, entries, solutions)
                        metadata = {
                            "method": method,
                            "axis": axis,
                            "layer_idx": layer_idx,
                            "selected_passes": selected_passes,
                            "ratios": ratios,
                            "target_values": target_values,
                            "ranking_applicable": axis not in non_sortable_axes,
                            "layer_count": layer_count,
                            "layer_indices": layer_indices,
                            "num_solutions": len(solutions),
                            "sorted_teacher_dir": str(sorted_dir),
                            "transformed_log_dir": str(transformed_log_dir),
                            "limitations": [],
                            "identity": stable_hash(
                                {
                                    "method": method,
                                    "axis": axis,
                                    "layer_idx": layer_idx,
                                    "selected_passes": selected_passes,
                                    "ratios": ratios,
                                    "target_values": target_values,
                                    "layer_count": layer_count,
                                    "layer_indices": layer_indices,
                                    "teacher_dir": str(teacher_dir),
                                    "activations_log_dir": str(activations_log_dir),
                                },
                                prefix="activation_diag",
                            ),
                        }
                        (method_dir / "diagnostic_metadata.json").write_text(
                            json.dumps(canonicalize(metadata), indent=2, sort_keys=True) + "\n"
                        )
                    dist.barrier()

                    scoring_output_dir = (
                        method_dir / "single_sequence_replacement_solutions--validation"
                    )
                    solutions_file = method_dir / "single_sequence_replacement_solutions.json"
                    expected_solutions = (
                        len(json.loads(solutions_file.read_text()))
                        if solutions_file.is_file()
                        else None
                    )
                    existing_solutions = len(list(scoring_output_dir.glob("solution_*.json")))
                    skip_scoring = (
                        expected_solutions is not None
                        and existing_solutions >= expected_solutions
                        and not bool(diag_cfg.get("force_rescore", False))
                    )
                    if skip_scoring:
                        if dist.is_master():
                            mprint(
                                "[activation_diagnostic] skipping completed "
                                f"{axis}/{method}/{layer_name}: {existing_solutions}/{expected_solutions} solutions"
                            )
                        dist.barrier()
                        method_outputs[f"{axis}:{method}:{layer_name}"] = str(scoring_output_dir)
                        continue

                    parallel = _diagnostic_parallel(hydra_cfg, diag_cfg)
                    scoring_cfg = _scoring_cfg_for_method(
                        hydra_cfg,
                        method_dir=method_dir,
                        scoring_output_dir=scoring_output_dir,
                        parallel=parallel,
                        source_checkpoint_dir=sorted_dir,
                        target_teacher_dir=teacher_dir,
                    )
                    for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
                        if key in diag_cfg:
                            scoring_cfg.scoring[key] = diag_cfg[key]
                    from ..plugins.automodel.solution_launch import launch_score_solutions_automodel

                    launch_score_solutions_automodel(scoring_cfg)
                    dist.barrier()
                    method_outputs[f"{axis}:{method}:{layer_name}"] = str(scoring_output_dir)

        summary = None
        if dist.is_master():
            score_rows = _collect_existing_rows(diag_root)
            summary = _write_summary(
                score_rows,
                artifacts_dir,
                requested_metric=requested_metric,
                comparison_tolerance=float(diag_cfg.get("comparison_tolerance", 0.0)),
            )
        dist.barrier()

    return complete_stage(
        config,
        manifest,
        outputs={
            "diagnostic_root": str(diag_root),
            "artifacts_dir": str(artifacts_dir),
            "methods": methods,
            "axes": axes,
            "ratios": ratios,
            "target_values": target_values,
            "non_sortable_axes": sorted(non_sortable_axes),
            "summary_path": str(artifacts_dir / "activation_diagnostic_summary.json"),
            "table_path": str(artifacts_dir / "activation_diagnostic_table.md"),
            "csv_path": str(artifacts_dir / "activation_diagnostic_scores.csv"),
        },
    )


def _validation_args_for_checkpoint(
    hydra_cfg: Any,
    config: dict[str, Any],
    diag_cfg: dict[str, Any],
    checkpoint_dir: Path,
) -> Any:
    """Build the small HF validation config used by sorted-teacher equivalence.

    This deliberately uses the same data knobs as replace-one-block scoring by
    default, so the diagnostic asks the same question as the later score stage:
    does this checkpoint have the same LM loss on the validation samples we will
    use for Puzzletron decisions?
    """

    scoring = _get(hydra_cfg, "scoring", {})
    data = dict(_get(config, "data", {}) or {})
    scoring_data = dict(data.get("scoring") or {})
    args = {
        "descriptor": _get(hydra_cfg, "descriptor", None),
        "model_name_or_path": str(checkpoint_dir),
        "tokenizer_name": str(_teacher_dir(config, hydra_cfg)),
        "trust_remote_code": bool(_get(_get(config, "model", {}), "trust_remote_code", True)),
        "model_dtype": diag_cfg.get("model_dtype", "torch.bfloat16"),
        "autocast_dtype": diag_cfg.get("autocast_dtype", "torch.bfloat16"),
        "dataset_path": _get(hydra_cfg, "dataset_path", None),
        "data_column": _get(scoring, "data_column", _get(hydra_cfg, "data_column", "messages")),
        "block_size": int(
            diag_cfg.get(
                "block_size",
                _get(
                    scoring, "block_size", _get(_get(hydra_cfg, "pruning", {}), "block_size", 2048)
                ),
            )
        ),
        "eval_samples": int(
            diag_cfg.get(
                "eval_samples",
                _get(scoring, "eval_samples", scoring_data.get("num_samples", 128)),
            )
        ),
        "micro_batch_size": int(
            diag_cfg.get(
                "micro_batch_size",
                _get(scoring, "micro_batch_size", scoring_data.get("micro_batch_size", 1)),
            )
        ),
        "seed": int(diag_cfg.get("seed", _get(scoring, "seed", 42))),
        "shuffle_seed": int(diag_cfg.get("shuffle_seed", _get(scoring, "shuffle_seed", 444))),
        "val_dataset_name": diag_cfg.get(
            "val_dataset_name",
            _get(scoring, "val_dataset_name", _get(hydra_cfg, "val_dataset_name", "train")),
        ),
        "source_datasets_to_discard": list(_get(scoring, "source_datasets_to_discard", [])),
        "bos_rate": float(_get(scoring, "bos_rate", 1.0)),
        "fim_rate": float(_get(scoring, "fim_rate", 0.0)),
        "fim_spm_rate": float(_get(scoring, "fim_spm_rate", 0.0)),
        "varlen": bool(_get(scoring, "varlen", False)),
        "load_dataset_fn": _get(scoring, "load_dataset_fn", "load_from_disk_fn"),
        "realized_dataset_cache_dir": _get(
            scoring,
            "realized_dataset_cache_dir",
            str(_puzzle_dir(config, hydra_cfg) / "dataset_cache"),
        ),
        "calc_losses_on_cpu": bool(diag_cfg.get("calc_losses_on_cpu", False)),
        "write_results": False,
        "activations_log_dir": None,
    }
    return OmegaConf.create(args)


def _avg_metric(losses: dict[str, Any] | None, key: str) -> float | None:
    if not losses or key not in losses:
        return None
    value = losses[key]
    if isinstance(value, dict) and "avg" in value:
        return float(value["avg"])
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _validate_checkpoint_lm_loss(
    hydra_cfg: Any,
    config: dict[str, Any],
    diag_cfg: dict[str, Any],
    checkpoint_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    from ..tools.validate_model import validate_model

    args = _validation_args_for_checkpoint(hydra_cfg, config, diag_cfg, checkpoint_dir)
    losses, _ = validate_model(args, hydra_cfg=hydra_cfg)
    summary = {
        key: _avg_metric(losses, key)
        for key in _PRIMARY_METRICS
        if _avg_metric(losses, key) is not None
    }
    return summary, losses


def _sort_equivalence_decision(
    *,
    delta: float,
    reverse_delta: float | None,
    tolerance: float,
    reverse_tolerance: float,
) -> dict[str, bool]:
    """Gate the production sort independently from its reverse-order control."""

    sorted_passed = abs(float(delta)) <= float(tolerance)
    reverse_passed = reverse_delta is None or abs(float(reverse_delta)) <= float(
        reverse_tolerance
    )
    return {
        "sorted_passed": sorted_passed,
        "reverse_passed": reverse_passed,
        "passed": sorted_passed and reverse_passed,
    }


def _sort_equivalence_tolerances(
    diag_cfg: dict[str, Any],
    descriptor: type[ModelDescriptor],
) -> tuple[float, float]:
    descriptor_tolerances = descriptor.checkpoint_equivalence_tolerances()
    tolerance = float(
        diag_cfg.get(
            "max_abs_lm_loss_delta",
            descriptor_tolerances.get("max_abs_lm_loss_delta", 1e-3),
        )
    )
    reverse_tolerance = float(
        diag_cfg.get("max_abs_reverse_lm_loss_delta", tolerance)
    )
    return tolerance, reverse_tolerance


def sort_equivalence_stage(config: dict[str, Any], manifest: StageManifest):
    """Evaluate teacher and sorted teacher with the chunked AutoModel scorer."""

    hydra_cfg = __import__(
        "modelopt.torch.puzzletron.pipeline_config",
        fromlist=["load_runtime_hydra_config"],
    ).load_runtime_hydra_config(config)
    diag_cfg = dict(config.get("sort_sanity") or {})
    teacher_dir = _teacher_dir(config, hydra_cfg)
    sorted_dir = _puzzle_dir(config, hydra_cfg) / "ckpts" / "sorted_teacher"
    include_reverse = bool(diag_cfg.get("include_reverse", True))
    reverse_dir = Path(
        diag_cfg.get("reverse_checkpoint_dir")
        or (_puzzle_dir(config, hydra_cfg) / "ckpts" / "reverse_sorted_teacher")
    )
    reverse_logs = Path(
        diag_cfg.get("reverse_activation_logs_dir")
        or (
            _puzzle_dir(config, hydra_cfg)
            / "pruning"
            / "pruning_scores"
            / "automodel"
            / "reverse_all_axes"
        )
    )
    artifacts_dir = _puzzle_dir(config, hydra_cfg) / "artifacts" / "sort_sanity"
    metric = str(diag_cfg.get("metric", "lm_loss"))
    descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
    tolerance, reverse_tolerance = _sort_equivalence_tolerances(
        diag_cfg,
        descriptor,
    )
    if not (sorted_dir / "config.json").is_file():
        raise FileNotFoundError(f"sorted checkpoint missing config.json: {sorted_dir}")

    diag_root = _puzzle_dir(config, hydra_cfg) / "diagnostics" / "sort_sanity"
    scoring_output_dir = diag_root / "single_sequence_replacement_solutions--validation"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    with _distributed(hydra_cfg):
        if include_reverse:
            needs_reverse_parent = _diagnostic_checkpoint_needs_rebuild(reverse_dir)
            if needs_reverse_parent and dist.is_master():
                if reverse_dir.exists():
                    shutil.rmtree(reverse_dir)
                _write_transformed_activation_logs(
                    _activations_log_dir(config, hydra_cfg),
                    reverse_logs,
                    method="reverse",
                    seed=int(diag_cfg.get("reverse_seed", 1234)),
                    axis="__all__",
                )
            dist.barrier()
            if needs_reverse_parent:
                sort_cfg = _get(hydra_cfg, "sort", {})
                embedding_cfg = _get(hydra_cfg, "embedding_pruning", {})
                build_sorted_teacher(
                    teacher_dir,
                    reverse_logs,
                    reverse_dir,
                    descriptor,
                    deferred_axes=tuple(_get(sort_cfg, "deferred_axes", ()) or ()),
                    mamba_state_score_key=str(
                        _get(sort_cfg, "mamba_state_score_key", "ssm_channel_contrib")
                    ),
                    embedding_widths=tuple(_get(embedding_cfg, "widths", ()) or ()),
                )
            dist.barrier()
            if _diagnostic_checkpoint_needs_rebuild(reverse_dir):
                raise FileNotFoundError(
                    f"global reverse-sorted parent is incomplete: {reverse_dir}"
                )
        if dist.is_master():
            if bool(diag_cfg.get("overwrite", True)) and diag_root.exists():
                shutil.rmtree(diag_root)
            diag_root.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        scoring_cfg = _scoring_cfg_for_method(
            hydra_cfg,
            method_dir=diag_root,
            scoring_output_dir=scoring_output_dir,
            parallel=_diagnostic_parallel(hydra_cfg, diag_cfg),
        )
        scoring_cfg.scoring.source_checkpoint_dir = str(sorted_dir)
        scoring_cfg.scoring.target_teacher_dir = str(teacher_dir)
        scoring_cfg.scoring.baseline_only = True
        scoring_cfg.scoring.skip_existing_solutions = False
        for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
            if key in diag_cfg:
                scoring_cfg.scoring[key] = diag_cfg[key]
        from ..plugins.automodel.solution_launch import launch_score_solutions_automodel

        launch_score_solutions_automodel(scoring_cfg)
        dist.barrier()

        reverse_output_dir = None
        if include_reverse:
            reverse_root = diag_root / "reverse"
            reverse_output_dir = reverse_root / "single_sequence_replacement_solutions--validation"
            if dist.is_master():
                reverse_root.mkdir(parents=True, exist_ok=True)
            dist.barrier()
            reverse_scoring_cfg = _scoring_cfg_for_method(
                hydra_cfg,
                method_dir=reverse_root,
                scoring_output_dir=reverse_output_dir,
                parallel=_diagnostic_parallel(hydra_cfg, diag_cfg),
            )
            reverse_scoring_cfg.scoring.source_checkpoint_dir = str(reverse_dir)
            reverse_scoring_cfg.scoring.target_teacher_dir = str(teacher_dir)
            reverse_scoring_cfg.scoring.baseline_only = True
            reverse_scoring_cfg.scoring.skip_existing_solutions = False
            for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
                if key in diag_cfg:
                    reverse_scoring_cfg.scoring[key] = diag_cfg[key]
            launch_score_solutions_automodel(reverse_scoring_cfg)
            dist.barrier()

    summary_path = artifacts_dir / "summary.json"
    md_path = artifacts_dir / "table.md"
    delta_value = None
    reverse_delta_value = None
    if dist.is_master():
        teacher_raw = json.loads((scoring_output_dir / "teacher.json").read_text())
        sorted_result_path = scoring_output_dir / "sliced_teacher.json"
        sorted_raw = json.loads(sorted_result_path.read_text())
        reverse_raw = (
            json.loads((reverse_output_dir / "sliced_teacher.json").read_text())
            if reverse_output_dir is not None
            else None
        )
        teacher_value = _metric_avg(teacher_raw, metric)
        sorted_value = _metric_avg(sorted_raw, metric)
        if teacher_value is None or sorted_value is None:
            raise RuntimeError(
                f"Could not find metric {metric!r}; teacher={teacher_raw} sorted={sorted_raw}"
            )
        delta = float(sorted_value) - float(teacher_value)
        delta_value = delta
        reverse_value = _metric_avg(reverse_raw, metric) if reverse_raw is not None else None
        if include_reverse and reverse_value is None:
            raise RuntimeError(
                f"Could not find metric {metric!r} for reverse-sorted checkpoint: {reverse_raw}"
            )
        reverse_delta = (
            float(reverse_value) - float(teacher_value) if reverse_value is not None else None
        )
        reverse_delta_value = reverse_delta
        decision = _sort_equivalence_decision(
            delta=delta,
            reverse_delta=reverse_delta,
            tolerance=tolerance,
            reverse_tolerance=reverse_tolerance,
        )
        sorted_passed = decision["sorted_passed"]
        reverse_passed = decision["reverse_passed"]
        passed = decision["passed"]
        summary = {
            "metric": metric,
            "teacher_dir": str(teacher_dir),
            "sorted_teacher_dir": str(sorted_dir),
            "teacher": {
                key: _metric_avg(teacher_raw, key)
                for key in _PRIMARY_METRICS
                if _metric_avg(teacher_raw, key) is not None
            },
            "sorted_teacher": {
                key: _metric_avg(sorted_raw, key)
                for key in _PRIMARY_METRICS
                if _metric_avg(sorted_raw, key) is not None
            },
            "reverse_sorted": (
                {
                    key: _metric_avg(reverse_raw, key)
                    for key in _PRIMARY_METRICS
                    if _metric_avg(reverse_raw, key) is not None
                }
                if reverse_raw is not None
                else None
            ),
            "delta": delta,
            "abs_delta": abs(delta),
            "sorted_passed": sorted_passed,
            "reverse_delta": reverse_delta,
            "reverse_abs_delta": abs(reverse_delta) if reverse_delta is not None else None,
            "reverse_passed": reverse_passed if include_reverse else None,
            "max_abs_delta": tolerance,
            "max_abs_reverse_delta": reverse_tolerance,
            "passed": passed,
            "findings": [],
            "verdict": "passed" if passed else "warning",
            "teacher_result": str(scoring_output_dir / "teacher.json"),
            "sorted_result": str(sorted_result_path),
            "reverse_sorted_dir": str(reverse_dir) if include_reverse else None,
            "reverse_result": (
                str(reverse_output_dir / "sliced_teacher.json")
                if reverse_output_dir is not None
                else None
            ),
        }
        summary_path.write_text(json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n")
        md_path.write_text(
            "\n".join(
                [
                    f"# Sorted Teacher Equivalence ({metric})",
                    "",
                    "| checkpoint | value | delta_vs_teacher |",
                    "| --- | --- | --- |",
                    f"| teacher | {teacher_value:.8g} | 0 |",
                    f"| sorted_teacher | {sorted_value:.8g} | {delta:.8g} |",
                    *(
                        [
                            f"| reverse_sorted_teacher | {reverse_value:.8g} | "
                            f"{reverse_delta:.8g} |"
                        ]
                        if reverse_value is not None and reverse_delta is not None
                        else []
                    ),
                    "",
                    f"pass: {'yes' if passed else 'no'} "
                    f"(max_abs_delta={tolerance:.3g}, "
                    f"max_abs_reverse_delta={reverse_tolerance:.3g})",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        mprint(md_path.read_text(encoding="utf-8"))
        findings = []
        if not passed:
            from ..diagnostics.sanity_verdict import (
                SanityVerdict,
                complete_sanity_stage,
                finding_from_message,
            )

            if not sorted_passed:
                findings.append(
                    finding_from_message(
                        stage="sort_sanity",
                        message=(
                            f"sorted teacher {metric} drift too large: delta={delta:.6g} "
                            f"tolerance={tolerance:.6g}"
                        ),
                        evidence={"metric": metric, "delta": delta, "tolerance": tolerance},
                    )
                )
            if include_reverse and not reverse_passed:
                findings.append(
                    finding_from_message(
                        stage="sort_sanity",
                        message=(
                            f"reverse-sorted teacher {metric} drift too large: "
                            f"reverse_delta={reverse_delta:.6g} "
                            f"tolerance={reverse_tolerance:.6g}"
                        ),
                        evidence={
                            "metric": metric,
                            "reverse_delta": reverse_delta,
                            "tolerance": reverse_tolerance,
                        },
                    )
                )
            summary["findings"] = findings
            summary_path.write_text(
                json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
            )
            dist.barrier()
            return complete_sanity_stage(
                config,
                manifest,
                outputs={
                    "summary_path": str(summary_path),
                    "table_path": str(md_path),
                    "metric": metric,
                    "delta": delta_value,
                    "reverse_delta": reverse_delta_value,
                },
                verdict=SanityVerdict(passed=False, findings=findings),
            )
    dist.barrier()
    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "summary_path": str(summary_path),
            "table_path": str(md_path),
            "metric": metric,
            "delta": delta_value,
            "reverse_delta": reverse_delta_value,
        },
        verdict=SanityVerdict(passed=True),
    )


def width_slice_equivalence_stage(config: dict[str, Any], manifest: StageManifest):
    """Compare content-identical physical/runtime slices on one canonical batch."""

    hydra_cfg = __import__(
        "modelopt.torch.puzzletron.pipeline_config",
        fromlist=["load_runtime_hydra_config"],
    ).load_runtime_hydra_config(config)
    stage_cfg = dict(config.get("slicing_sanity") or {})
    if stage_cfg.get("backend") == "distributed_parent_sweep":
        summary_path = _puzzle_dir(config, hydra_cfg) / "artifacts" / "slicing_sanity" / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                "distributed parent-sweep slicing summary is missing; run width_sanity "
                f"with physical_realization=true first: {summary_path}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not summary.get("rows") or not summary.get("axes"):
            raise RuntimeError(f"distributed slicing summary has no evidence: {summary_path}")
        findings = list(summary.get("findings") or ())
        passed = bool(summary.get("passed", not findings))
        from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

        return complete_sanity_stage(
            config,
            manifest,
            outputs={
                "summary_path": str(summary_path),
                "backend": "distributed_parent_sweep",
                "axes": list(summary.get("axes") or ()),
            },
            verdict=SanityVerdict(passed=passed, findings=findings),
        )
    diag_cfg = dict(config.get("width_slice_equivalence") or {})
    sorted_dir = _puzzle_dir(config, hydra_cfg) / "ckpts" / "sorted_teacher"
    if not (sorted_dir / "config.json").is_file():
        raise FileNotFoundError(f"sorted checkpoint missing config.json: {sorted_dir}")

    descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
    model_config = load_model_config(sorted_dir)
    configured_layers = tuple(diag_cfg.get("layer_indices") or ())
    sampled_layers = tuple(int(layer) for layer in configured_layers) or None
    if sampled_layers is not None and len(set(sampled_layers)) != 2:
        raise ValueError(
            "width-slice equivalence requires exactly two distinct sampled layers; "
            f"configured={list(sampled_layers)}"
        )

    tolerances = descriptor.width_slice_equivalence_tolerances()
    tolerances.update(dict(diag_cfg.get("tolerances") or {}))
    data_cfg = dict(config.get("scoring") or {})
    data_cfg.update(
        {
            key: value
            for key, value in diag_cfg.items()
            if key
            in {
                "block_size",
                "dataset",
                "dataset_path",
                "eval_samples",
                "micro_batch_size",
                "packed_token_cache_path",
                "realized_dataset_cache_dir",
                "seed",
                "tokenizer_name",
                "val_dataset_name",
                "varlen",
            }
        }
    )
    data_cfg.setdefault("model_name_or_path", str(sorted_dir))
    data_cfg.setdefault("teacher_dir", str(sorted_dir))
    data_cfg.setdefault("descriptor", _get(hydra_cfg, "descriptor", None))
    data_section = dict(config.get("data") or {})
    data_cfg.update(
        {
            key: value
            for key, value in data_section.items()
            if key
            in {
                "max_sample_length",
                "pack_size",
                "packing",
                "packing_ratio",
                "path",
            }
        }
    )
    layout = str(data_section.get("layout", "fixed"))
    modality = str(data_section.get("modality", "text"))
    from ..utils.data.dataloaders import (
        prepare_multimodal_validation_dataloader,
        prepare_validation_dataloader,
    )

    if modality == "multimodal":
        dataloader = prepare_multimodal_validation_dataloader(
            data_cfg,
            checkpoint_dir=sorted_dir,
            data_layout=layout,
        )
    else:
        dataloader = prepare_validation_dataloader(
            data_cfg,
            None,
            data_layout=layout,
        )
    raw_batch = next(iter(dataloader))
    if isinstance(raw_batch, dict):
        input_ids = raw_batch.get("input_ids")
        batch_size = (
            int(input_ids.shape[0]) if torch.is_tensor(input_ids) and input_ids.ndim > 1 else 1
        )
    else:
        batch_size = raw_batch.batch_size
    batch = normalize_width_slice_batch(
        raw_batch,
        descriptor=descriptor,
        checkpoint_config=model_config,
        layout=layout,
        sample_ids=tuple(f"width-slice-row-{index}" for index in range(batch_size)),
        source_metadata={
            "dataset": data_cfg.get("dataset_path", data_cfg.get("dataset")),
            "revision": data_cfg.get("revision", "materialized-manifest"),
            "checkpoint": str(sorted_dir),
        },
    )
    artifacts_dir = _puzzle_dir(config, hydra_cfg) / "artifacts" / "width_slice_equivalence"
    summary = evaluate_width_slice_equivalence(
        descriptor=descriptor,
        sorted_checkpoint_dir=sorted_dir,
        batch=batch,
        artifact_dir=artifacts_dir,
        tolerances=tolerances,
        alignment=int(diag_cfg.get("alignment", 1)),
        sampled_layers=sampled_layers,
    )
    validate_width_slice_artifacts(artifacts_dir, descriptor=descriptor)
    findings = [
        {
            "stage": "slicing_sanity",
            "message": f"width-slice equivalence failed for case {case.get('case_id')}",
            "evidence": {"case": case},
            "severity": "warning",
        }
        for case in summary.get("cases", ())
        if not case.get("passed", True)
    ]
    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "summary_path": str(artifacts_dir / "summary.json"),
            "artifact_manifest_path": str(artifacts_dir / "manifest.json"),
            "artifact_identity": summary["artifact_identity"],
            "passed": summary["passed"],
            "cases": summary["cases"],
            "tolerances": summary["tolerances"],
        },
        verdict=SanityVerdict(passed=bool(summary["passed"]), findings=findings),
    )


def _find_bypass_checkpoint(puzzle_dir: Path, diag_cfg: dict[str, Any]) -> Path:
    explicit = diag_cfg.get("bypass_checkpoint_dir")
    if explicit:
        path = Path(explicit)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"bypass_checkpoint_dir missing config.json: {path}")
        return path.resolve()
    candidates = sorted(
        (
            p.parent
            for p in (puzzle_dir / "bypass" / "bypass_runs").glob("*/final-step-*-ckpt/config.json")
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No bypass final checkpoints found under {puzzle_dir / 'bypass' / 'bypass_runs'}"
        )
    return candidates[0].resolve()


def _bypass_diagnostic_rows(diag_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_dir in sorted(diag_root.rglob("single_sequence_replacement_solutions--validation")):
        rel = output_dir.relative_to(diag_root).parts
        if len(rel) < 3:
            continue
        if rel[0] == "full_overlay":
            continue
        axis = rel[0]
        source = rel[1]
        for row in _extract_rows(source, output_dir):
            row["axis"] = row.get("axis") or axis
            row["source"] = source
            rows.append(row)
    return rows


def _bypass_full_overlay_solutions(
    block_configs: list[BlockConfig],
    layer_indices: list[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build no-prune one-block solutions for validating bypass checkpoint overlays.

    Nested bypass checkpoints train every block in one checkpoint.  Downstream
    replace-one-block diagnosis overlays only the candidate block, so before
    checking pruned slices we first verify that the full-size overlay itself is
    close to the sorted teacher.  These candidates intentionally keep the
    teacher block config unchanged; the only changed tensor source is the
    bypass checkpoint overlay.
    """

    teacher_entries = _teacher_replacements(block_configs)
    entries = list(teacher_entries.values())
    teacher_serialized = [_to_serializable(block_config) for block_config in block_configs]
    solutions: list[dict[str, Any]] = []
    for solution_id, layer_idx in enumerate(layer_indices):
        candidate = _entry(layer_idx, block_configs[layer_idx])
        candidate["diagnostic"] = {
            "axis": "bypass_full_overlay",
            "layer_idx": layer_idx,
            "teacher_value": 1,
            "target_value": 1,
            "ratio": 1.0,
            "changed_layers": [],
            "num_changed_layers": 0,
            "solution_id": solution_id,
        }
        chosen = [
            candidate if idx == layer_idx else teacher_entries[idx]
            for idx in range(len(block_configs))
        ]
        solution = {
            "single_sequence_replacement": candidate,
            "chosen_replacements": chosen,
            "block_configs": teacher_serialized,
            "diagnostic": candidate["diagnostic"],
        }
        entries.append(candidate)
        solutions.append(solution)
    return entries, solutions


def _write_bypass_full_overlay_summary(
    rows: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    requested_metric: str | None,
    tolerance: float,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    metric = _primary_metric(rows, requested_metric)
    table_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda x: int(x.get("layer_idx") or -1)):
        value = row.get(metric)
        passed = value is not None and math.isfinite(value) and float(value) <= tolerance
        table_rows.append(
            {
                "layer": row.get("layer_idx"),
                "value": value,
                "max_allowed": tolerance,
                "passed": passed,
                "solution_file": row.get("solution_file"),
            }
        )

    md_path = artifacts_dir / "bypass_full_overlay_table.md"
    lines = [
        f"# Bypass Full Overlay Check ({metric})",
        "",
        "| layer | full-overlay | max | ok |",
        "| --- | --- | --- | --- |",
    ]

    def fmt(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    for row in table_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["layer"]),
                    fmt(row["value"]),
                    fmt(row["max_allowed"]),
                    fmt(row["passed"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    passed = bool(table_rows) and all(bool(row["passed"]) for row in table_rows)
    summary = {
        "primary_metric": metric,
        "num_scores": len(rows),
        "max_allowed_delta": tolerance,
        "passed": passed,
        "markdown_path": str(md_path),
        "table": table_rows,
    }
    (artifacts_dir / "bypass_full_overlay_summary.json").write_text(
        json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
    )
    mprint("\n".join(lines[: min(len(lines), 80)]))
    return summary


def _write_bypass_diagnostic_summary(
    rows: list[dict[str, Any]],
    artifacts_dir: Path,
    *,
    requested_metric: str | None,
    tolerance: float,
) -> dict[str, Any]:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    csv_path = artifacts_dir / "bypass_diagnostic_scores.csv"
    metric = _primary_metric(rows, requested_metric)
    fieldnames = [
        "axis",
        "layer_idx",
        "ratio",
        "teacher_value",
        "target_value",
        "combo",
        "source",
        "method",
        "kept_kv_groups",
        "removed_kv_groups",
        "kv_group_order",
        "kept_query_heads_per_group",
        "removed_query_heads_per_group",
        "query_head_order_per_group",
        "changed_layers",
        "num_changed_layers",
        *_PRIMARY_METRICS,
        "solution_id",
        "solution_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda x: (
                str(x.get("axis")),
                int(x.get("layer_idx") or -1),
                float(x.get("ratio") or -1),
                str(x.get("source")),
            ),
        ):
            writer.writerow({key: row.get(key) for key in fieldnames})

    grouped: dict[tuple[Any, Any, Any], dict[str, float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is None or not math.isfinite(value):
            continue
        key = (
            row.get("axis"),
            row.get("layer_idx"),
            row.get("combo") or row.get("target_value"),
        )
        grouped.setdefault(key, {})[str(row.get("source") or row.get("method"))] = float(value)

    table_rows: list[dict[str, Any]] = []
    for (axis, layer_idx, target), values in sorted(
        grouped.items(),
        key=lambda item: (str(item[0][0]), int(item[0][1]), str(item[0][2])),
    ):
        pruned = values.get("pruned")
        bypassed = values.get("bypassed")
        delta = None if pruned is None or bypassed is None else bypassed - pruned
        passed = delta is not None and delta <= tolerance
        table_rows.append(
            {
                "axis": axis,
                "layer": layer_idx,
                "target": target,
                "pruned": pruned,
                "bypassed": bypassed,
                "bypassed_minus_pruned": delta,
                "expected_loss_order": passed,
            }
        )

    md_path = artifacts_dir / "bypass_diagnostic_table.md"
    lines = [
        f"# Bypass Diagnostic ({metric})",
        "",
        "| axis | layer | target | pruned | bypassed | bypass-pruned | ok |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in table_rows:

        def fmt(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, bool):
                return "yes" if value else "no"
            if isinstance(value, float):
                return f"{value:.6g}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["axis"]),
                    fmt(row["layer"]),
                    fmt(row["target"]),
                    fmt(row["pruned"]),
                    fmt(row["bypassed"]),
                    fmt(row["bypassed_minus_pruned"]),
                    fmt(row["expected_loss_order"]),
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    passed = bool(table_rows) and all(bool(row["expected_loss_order"]) for row in table_rows)
    summary = {
        "primary_metric": metric,
        "num_scores": len(rows),
        "num_comparisons": len(table_rows),
        "max_allowed_delta": tolerance,
        "passed": passed,
        "csv_path": str(csv_path),
        "markdown_path": str(md_path),
        "table": table_rows,
    }
    (artifacts_dir / "bypass_diagnostic_summary.json").write_text(
        json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
    )
    mprint("\n".join(lines[: min(len(lines), 80)]))
    return summary


def bypass_diagnostic_stage(config: dict[str, Any], manifest: StageManifest):
    """Compare sorted-prefix pruning against the same slice from a bypass checkpoint."""

    hydra_cfg = __import__(
        "modelopt.torch.puzzletron.pipeline_config",
        fromlist=["load_runtime_hydra_config"],
    ).load_runtime_hydra_config(config)
    diag_cfg = dict(config.get("bypass_diagnostic") or {})
    if not diag_cfg.get("enabled", True):
        return complete_stage(
            config,
            manifest,
            outputs={"skipped": True},
            status="skipped",
            message="Bypass diagnostic is disabled.",
        )

    puzzle_dir = _puzzle_dir(config, hydra_cfg)
    sorted_dir = puzzle_dir / "ckpts" / "sorted_teacher"
    if not (sorted_dir / "config.json").is_file():
        raise FileNotFoundError(f"sorted checkpoint missing config.json: {sorted_dir}")
    bypass_dir = _find_bypass_checkpoint(puzzle_dir, diag_cfg)

    methods = ["pruned", "bypassed"]
    axes = _enabled_diagnostic_axes(config, diag_cfg)
    ratios = [float(ratio) for ratio in diag_cfg.get("ratios", [0.25, 0.5, 0.75])]
    layer_count = int(diag_cfg.get("layer_count", 5))
    layer_indices = _as_int_list(diag_cfg.get("layer_indices"))
    requested_metric = diag_cfg.get("primary_metric")
    tolerance = float(diag_cfg.get("max_bypass_loss_delta", 0.0))
    mode = str(diag_cfg.get("mode", "block_combinations"))
    diag_root = puzzle_dir / "diagnostics" / "bypass"
    artifacts_dir = puzzle_dir / "artifacts" / "bypass_diagnostic"
    summary = None

    with _distributed(hydra_cfg):
        if dist.is_master() and bool(diag_cfg.get("overwrite", True)) and diag_root.exists():
            shutil.rmtree(diag_root)
        dist.barrier()

        descriptor = ModelDescriptorFactory.get(_get(hydra_cfg, "descriptor", None))
        model_config = load_model_config(
            sorted_dir, trust_remote_code=descriptor.requires_trust_remote_code()
        )
        lm = descriptor.get_language_model_config(model_config)
        block_configs = list(maybe_cast_block_configs(model_config.block_configs))
        head_dim = getattr(lm, "head_dim", None) or (lm.hidden_size // lm.num_attention_heads)
        overlay_layers = (
            layer_indices
            if layer_indices is not None
            else _select_layers(list(range(len(block_configs))), layer_count, None)
        )

        if bool(diag_cfg.get("full_overlay_check", False)):
            entries, solutions = _bypass_full_overlay_solutions(block_configs, overlay_layers)
            method_dir = diag_root / "full_overlay" / "bypassed"
            scoring_output_dir = method_dir / "single_sequence_replacement_solutions--validation"
            if dist.is_master():
                _write_library_and_solutions(method_dir, sorted_dir, entries, solutions)
                metadata = {
                    "method": "bypassed",
                    "axis": "bypass_full_overlay",
                    "source_checkpoint_dir": str(sorted_dir),
                    "target_teacher_dir": str(sorted_dir),
                    "bypass_checkpoint_dir": str(bypass_dir),
                    "overlay_checkpoint_dir": str(bypass_dir),
                    "layer_indices": overlay_layers,
                    "num_solutions": len(solutions),
                }
                (method_dir / "diagnostic_metadata.json").write_text(
                    json.dumps(canonicalize(metadata), indent=2, sort_keys=True) + "\n"
                )
            dist.barrier()
            scoring_cfg = _scoring_cfg_for_method(
                hydra_cfg,
                method_dir=method_dir,
                scoring_output_dir=scoring_output_dir,
                parallel=_diagnostic_parallel(hydra_cfg, diag_cfg),
            )
            scoring_cfg.scoring.source_checkpoint_dir = str(sorted_dir)
            scoring_cfg.scoring.target_teacher_dir = str(sorted_dir)
            scoring_cfg.scoring.bypass_checkpoint_dir = str(bypass_dir)
            scoring_cfg.scoring.skip_existing_solutions = False
            for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
                if key in diag_cfg:
                    scoring_cfg.scoring[key] = diag_cfg[key]
            from ..plugins.automodel.solution_launch import launch_score_solutions_automodel

            launch_score_solutions_automodel(scoring_cfg)
            dist.barrier()

        if mode in {"block", "blocks", "block_combinations", "both"}:
            entries, solutions = _configured_bypass_block_targets(
                block_configs,
                axes=axes,
                ratios=ratios,
                layer_count=layer_count,
                layer_indices=layer_indices,
                config=config,
            )
            _annotate_solution_selections(
                solutions=solutions,
                teacher_block_configs=block_configs,
                sorted_teacher_dir=sorted_dir,
            )
            entries = _entries_for_solutions(block_configs, solutions)
            for method in methods:
                source_dir = sorted_dir
                overlay_dir = bypass_dir if method == "bypassed" else None
                method_dir = diag_root / "block_combo" / method
                scoring_output_dir = (
                    method_dir / "single_sequence_replacement_solutions--validation"
                )
                if dist.is_master():
                    _write_library_and_solutions(method_dir, source_dir, entries, solutions)
                    metadata = {
                        "method": method,
                        "axis": "block_combo",
                        "source_checkpoint_dir": str(source_dir),
                        "target_teacher_dir": str(sorted_dir),
                        "bypass_checkpoint_dir": str(bypass_dir),
                        "overlay_checkpoint_dir": str(overlay_dir) if overlay_dir else None,
                        "ratios": ratios,
                        "layer_count": layer_count,
                        "layer_indices": layer_indices,
                        "num_solutions": len(solutions),
                    }
                    (method_dir / "diagnostic_metadata.json").write_text(
                        json.dumps(canonicalize(metadata), indent=2, sort_keys=True) + "\n"
                    )
                dist.barrier()
                scoring_cfg = _scoring_cfg_for_method(
                    hydra_cfg,
                    method_dir=method_dir,
                    scoring_output_dir=scoring_output_dir,
                    parallel=_diagnostic_parallel(hydra_cfg, diag_cfg),
                )
                scoring_cfg.scoring.source_checkpoint_dir = str(source_dir)
                scoring_cfg.scoring.target_teacher_dir = str(sorted_dir)
                if overlay_dir is not None:
                    scoring_cfg.scoring.bypass_checkpoint_dir = str(overlay_dir)
                scoring_cfg.scoring.skip_existing_solutions = False
                for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
                    if key in diag_cfg:
                        scoring_cfg.scoring[key] = diag_cfg[key]
                from ..plugins.automodel.solution_launch import launch_score_solutions_automodel

                launch_score_solutions_automodel(scoring_cfg)
                dist.barrier()

        if mode in {"axis", "axes", "per_axis", "both"}:
            for axis in axes:
                entries, solutions = _diagnostic_solutions(
                    block_configs,
                    axes=[axis],
                    ratios=ratios,
                    target_values=None,
                    layer_count=layer_count,
                    layer_indices=layer_indices,
                )
                _annotate_solution_selections(
                    solutions=solutions,
                    teacher_block_configs=block_configs,
                    sorted_teacher_dir=sorted_dir,
                )
                entries = _entries_for_solutions(block_configs, solutions)
                for method in methods:
                    source_dir = sorted_dir
                    overlay_dir = bypass_dir if method == "bypassed" else None
                    method_dir = diag_root / axis / method
                    scoring_output_dir = (
                        method_dir / "single_sequence_replacement_solutions--validation"
                    )
                    if dist.is_master():
                        _write_library_and_solutions(method_dir, source_dir, entries, solutions)
                        metadata = {
                            "method": method,
                            "axis": axis,
                            "source_checkpoint_dir": str(source_dir),
                            "target_teacher_dir": str(sorted_dir),
                            "bypass_checkpoint_dir": str(bypass_dir),
                            "overlay_checkpoint_dir": str(overlay_dir) if overlay_dir else None,
                            "ratios": ratios,
                            "layer_count": layer_count,
                            "layer_indices": layer_indices,
                            "num_solutions": len(solutions),
                        }
                        (method_dir / "diagnostic_metadata.json").write_text(
                            json.dumps(canonicalize(metadata), indent=2, sort_keys=True) + "\n"
                        )
                    dist.barrier()
                    scoring_cfg = _scoring_cfg_for_method(
                        hydra_cfg,
                        method_dir=method_dir,
                        scoring_output_dir=scoring_output_dir,
                        parallel=_diagnostic_parallel(hydra_cfg, diag_cfg),
                    )
                    scoring_cfg.scoring.source_checkpoint_dir = str(source_dir)
                    scoring_cfg.scoring.target_teacher_dir = str(sorted_dir)
                    if overlay_dir is not None:
                        scoring_cfg.scoring.bypass_checkpoint_dir = str(overlay_dir)
                    scoring_cfg.scoring.skip_existing_solutions = False
                    for key in ("eval_samples", "micro_batch_size", "block_size", "varlen"):
                        if key in diag_cfg:
                            scoring_cfg.scoring[key] = diag_cfg[key]
                    from ..plugins.automodel.solution_launch import launch_score_solutions_automodel

                    launch_score_solutions_automodel(scoring_cfg)
                    dist.barrier()

        summary = None
        if dist.is_master():
            full_overlay_summary = None
            if bool(diag_cfg.get("full_overlay_check", False)):
                full_rows = _extract_rows(
                    "bypassed",
                    diag_root
                    / "full_overlay"
                    / "bypassed"
                    / "single_sequence_replacement_solutions--validation",
                )
                full_overlay_summary = _write_bypass_full_overlay_summary(
                    full_rows,
                    artifacts_dir,
                    requested_metric=requested_metric,
                    tolerance=float(diag_cfg.get("max_full_overlay_loss", 0.05)),
                )
            rows = _bypass_diagnostic_rows(diag_root)
            summary = _write_bypass_diagnostic_summary(
                rows,
                artifacts_dir,
                requested_metric=requested_metric,
                tolerance=tolerance,
            )
            findings = list(summary.get("findings") or ())
            if full_overlay_summary is not None and not full_overlay_summary.get("passed", False):
                findings.append(
                    {
                        "stage": "bypass_diagnostic",
                        "message": (
                            "Bypass full-overlay check did not pass; see "
                            f"{artifacts_dir / 'bypass_full_overlay_table.md'}"
                        ),
                        "severity": "warning",
                    }
                )
            if not summary.get("passed", False):
                findings.append(
                    {
                        "stage": "bypass_diagnostic",
                        "message": (
                            "Bypass diagnostic comparison did not pass; see "
                            f"{artifacts_dir / 'bypass_diagnostic_table.md'}"
                        ),
                        "severity": "warning",
                    }
                )
            if findings:
                summary = dict(summary)
                summary["findings"] = findings
                summary["passed"] = False
                (artifacts_dir / "bypass_diagnostic_summary.json").write_text(
                    json.dumps(canonicalize(summary), indent=2, sort_keys=True) + "\n"
                )
        dist.barrier()

    from ..diagnostics.sanity_verdict import SanityVerdict, complete_sanity_stage

    findings = list((summary or {}).get("findings") or ())
    return complete_sanity_stage(
        config,
        manifest,
        outputs={
            "diagnostic_root": str(diag_root),
            "artifacts_dir": str(artifacts_dir),
            "bypass_checkpoint_dir": str(bypass_dir),
            "summary_path": str(artifacts_dir / "bypass_diagnostic_summary.json"),
            "table_path": str(artifacts_dir / "bypass_diagnostic_table.md"),
            "csv_path": str(artifacts_dir / "bypass_diagnostic_scores.csv"),
        },
        verdict=SanityVerdict(
            passed=bool((summary or {}).get("passed", True)),
            findings=findings,
        ),
    )
