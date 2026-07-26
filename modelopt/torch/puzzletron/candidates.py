# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Iterator

try:
    from omegaconf import DictConfig, ListConfig, OmegaConf
except ImportError:  # pragma: no cover - exercised in minimal utility environments.
    DictConfig = ListConfig = None  # type: ignore[assignment]
    OmegaConf = None  # type: ignore[assignment]

from .block_config import SUBBLOCK_CLS_DICT, BlockConfig, SubblockConfig
from .identity import Identity, cache_key, candidate_identity, canonicalize, stable_hash

__all__ = [
    "Candidate",
    "CandidateLibrary",
    "build_candidate_library",
    "build_candidate_library_from_checkpoint",
    "discover_bypass_checkpoints",
    "load_block_configs_from_checkpoint",
    "load_stats_identity_cache",
    "read_candidate_library",
    "write_candidate_library",
]


_AXIS_TO_TARGET: dict[str, tuple[str, str]] = {
    "ffn_intermediate": ("ffn", "intermediate_size"),
    "intermediate_size": ("ffn", "intermediate_size"),
    "moe_expert_intermediate": ("moe", "expert_intermediate_size"),
    "expert_intermediate": ("moe", "expert_intermediate_size"),
    "moe_shared_expert_intermediate": ("moe", "shared_expert_intermediate_size"),
    "moe_shared_intermediate": ("moe", "shared_expert_intermediate_size"),
    "shared_expert_intermediate": ("moe", "shared_expert_intermediate_size"),
    "moe_experts": ("moe", "num_experts"),
    "num_experts": ("moe", "num_experts"),
    "moe_latent_dim": ("moe", "latent_dim"),
    "latent_dim": ("moe", "latent_dim"),
    "query_heads": ("attention", "q_heads_per_group"),
    "q_heads_per_group": ("attention", "q_heads_per_group"),
    "num_query_heads": ("attention", "num_query_heads"),
    "kv_groups": ("attention", "num_kv_heads"),
    "kv_heads": ("attention", "num_kv_heads"),
    "num_kv_heads": ("attention", "num_kv_heads"),
    "qk_head_dim": ("attention", "qk_head_dim"),
    "sliding_window_size": ("attention", "sliding_window_size"),
    "moe_top_k": ("moe", "top_k"),
    "top_k": ("moe", "top_k"),
    "mamba_heads": ("mamba", "num_heads"),
    "mamba_num_heads": ("mamba", "num_heads"),
    "mamba_head_dim": ("mamba", "head_dim"),
    "mamba_state_dim": ("mamba", "state_dim"),
    "mamba_ssm_state": ("mamba", "state_dim"),
    "ssm_state_size": ("mamba", "state_dim"),
    "gdn_key_groups": ("mamba", "gdn_key_groups"),
    "gdn_value_heads_per_group": ("mamba", "gdn_value_heads_per_group"),
    "gdn_key_head_dim": ("mamba", "state_dim"),
    "gdn_value_head_dim": ("mamba", "head_dim"),
    "mla_q_lora_rank": ("mla", "q_lora_rank"),
    "mla_kv_lora_rank": ("mla", "kv_lora_rank"),
    "mla_heads": ("mla", "num_heads"),
}

_VARIANT_FIELDS = {"sliding_window_size", "top_k"}
_DEFAULT_STATS_FILENAMES = (
    "subblock_stats.json",
    "block_stats.json",
    "runtime_stats.json",
    "candidate_stats.json",
)


@dataclass(frozen=True, kw_only=True)
class Candidate:
    layer_idx: int
    block_config: BlockConfig
    source_kind: str
    parent_checkpoint_identity: str
    hidden_width: int | None = None
    source_identity: str | None = None
    score_identity: str | None = None
    cost_identity: str | None = None
    stats_identity: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> Identity:
        return candidate_identity(
            self.layer_idx,
            self.block_config,
            {
                "source_kind": self.source_kind,
                "source_identity": self.source_identity,
                "parent_checkpoint_identity": self.parent_checkpoint_identity,
                "hidden_width": self.hidden_width,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.identity.value,
            "layer_idx": self.layer_idx,
            "block_config": self.block_config.to_dict(),
            "source_kind": self.source_kind,
            "parent_checkpoint_identity": self.parent_checkpoint_identity,
            "hidden_width": self.hidden_width,
            "source_identity": self.source_identity,
            "score_identity": self.score_identity,
            "cost_identity": self.cost_identity,
            "stats_identity": self.stats_identity,
            "metadata": canonicalize(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        candidate = cls(
            layer_idx=int(data["layer_idx"]),
            block_config=BlockConfig(**data["block_config"]),
            source_kind=data["source_kind"],
            parent_checkpoint_identity=data["parent_checkpoint_identity"],
            hidden_width=(
                None if data.get("hidden_width") is None else int(data["hidden_width"])
            ),
            source_identity=data.get("source_identity"),
            score_identity=data.get("score_identity"),
            cost_identity=data.get("cost_identity"),
            stats_identity=data.get("stats_identity"),
            metadata=dict(data.get("metadata") or {}),
        )
        candidate_id = data.get("candidate_id")
        if candidate_id is not None and candidate_id != candidate.identity.value:
            raise ValueError(
                f"Candidate id mismatch for layer {candidate.layer_idx}: "
                f"file has {candidate_id!r}, canonical id is {candidate.identity.value!r}"
            )
        return candidate


@dataclass(frozen=True, kw_only=True)
class CandidateLibrary:
    candidates: tuple[Candidate, ...]
    parent_checkpoint_identity: str
    settings_identity: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        by_id = {candidate.identity.value: candidate for candidate in self.candidates}
        object.__setattr__(
            self,
            "candidates",
            tuple(by_id[candidate_id] for candidate_id in sorted(by_id)),
        )

    @property
    def identity(self) -> str:
        return stable_hash(self.to_dict(include_identity=False), prefix="candidate_library")

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        data = {
            "version": 1,
            "parent_checkpoint_identity": self.parent_checkpoint_identity,
            "settings_identity": self.settings_identity,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "metadata": canonicalize(self.metadata),
        }
        if include_identity:
            data["library_id"] = self.identity
        return data

    @classmethod
    def from_candidates(
        cls,
        candidates: Iterable[Candidate],
        *,
        parent_checkpoint_identity: str,
        settings: Any,
        metadata: dict[str, Any] | None = None,
    ) -> "CandidateLibrary":
        return cls(
            candidates=tuple(candidates),
            parent_checkpoint_identity=parent_checkpoint_identity,
            settings_identity=cache_key("library_settings", {}, settings).value,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateLibrary":
        return cls(
            candidates=tuple(Candidate.from_dict(item) for item in data.get("candidates", [])),
            parent_checkpoint_identity=data["parent_checkpoint_identity"],
            settings_identity=data["settings_identity"],
            metadata=dict(data.get("metadata") or {}),
        )


def write_candidate_library(path: str | Path, library: CandidateLibrary) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(library.to_dict(), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def read_candidate_library(path: str | Path) -> CandidateLibrary:
    return CandidateLibrary.from_dict(json.loads(Path(path).read_text()))


def load_block_configs_from_checkpoint(checkpoint_dir: str | Path) -> tuple[BlockConfig, ...]:
    """Load typed Puzzletron ``block_configs`` from a checkpoint ``config.json``."""
    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Checkpoint config not found: {config_path}")
    config = json.loads(config_path.read_text())
    raw_block_configs = _find_block_configs(config)
    if raw_block_configs is None:
        raise ValueError(f"{config_path} does not contain Puzzletron block_configs")
    return tuple(BlockConfig(**block_config) for block_config in raw_block_configs)


def build_candidate_library_from_checkpoint(
    checkpoint_dir: str | Path,
    *,
    search_space: dict[str, Any] | None = None,
    output_path: str | Path | None = None,
    puzzle_dir: str | Path | None = None,
    parent_checkpoint_identity: str | None = None,
    include_self: bool = True,
    include_noops: bool = True,
    include_bypass: bool = True,
    stats_paths: Iterable[str | Path] | None = None,
    metadata: dict[str, Any] | None = None,
    hidden_width: int | None = None,
) -> CandidateLibrary:
    """Build and optionally write the canonical identity-keyed candidate library.

    This is the Stage-5 virtual library builder. It does not materialize weights:
    sorted-slice entries describe smaller typed ``BlockConfig`` objects sourced
    from the parent checkpoint, no-op entries carry typed no-op subblocks, and
    bypass entries point at discovered bypass checkpoints.
    """
    checkpoint_dir = Path(checkpoint_dir)
    block_configs = load_block_configs_from_checkpoint(checkpoint_dir)
    parent_identity = parent_checkpoint_identity or _checkpoint_identity(checkpoint_dir).value
    effective_puzzle_dir = Path(puzzle_dir) if puzzle_dir is not None else checkpoint_dir.parent.parent
    stats_cache = load_stats_identity_cache(
        stats_paths if stats_paths is not None else _default_stats_paths(effective_puzzle_dir),
        hidden_width=hidden_width,
    )
    candidates = build_candidate_library(
        block_configs,
        search_space=search_space,
        parent_checkpoint_identity=parent_identity,
        include_self=include_self,
        include_noops=include_noops,
        stats_cache=stats_cache,
        hidden_width=hidden_width,
    )
    if include_bypass:
        candidates.extend(
            _build_bypass_candidates(
                effective_puzzle_dir,
                parent_block_configs=block_configs,
                parent_checkpoint_identity=parent_identity,
                stats_cache=stats_cache,
            )
        )
    library = CandidateLibrary.from_candidates(
        candidates,
        parent_checkpoint_identity=parent_identity,
        settings={
            "search_space": search_space or {},
            "include_self": include_self,
            "include_noops": include_noops,
            "include_bypass": include_bypass,
            "stats_cache_entries": len(stats_cache),
            "hidden_width": hidden_width,
        },
        metadata={
            "format": "puzzletron_candidate_library",
            "checkpoint_dir": str(checkpoint_dir),
            "puzzle_dir": str(effective_puzzle_dir),
            "num_layers": len(block_configs),
            "hidden_width": hidden_width,
            **dict(metadata or {}),
        },
    )
    if output_path is not None:
        write_candidate_library(output_path, library)
    return library


def build_candidate_library(
    block_configs: Iterable[BlockConfig | dict[str, Any]],
    *,
    search_space: dict[str, Any] | None = None,
    parent_checkpoint_identity: str,
    include_self: bool = True,
    include_noops: bool = True,
    stats_cache: dict[tuple[int | None, str], str] | None = None,
    hidden_width: int | None = None,
) -> list[Candidate]:
    """Enumerate self, no-op, and sorted-slice candidates for typed blocks."""
    typed_block_configs = tuple(
        block_config if isinstance(block_config, BlockConfig) else BlockConfig(**block_config)
        for block_config in block_configs
    )
    axis_specs = _normalise_axis_specs(search_space or {})
    no_op_specs = _normalise_no_op_specs(search_space or {})
    candidates: list[Candidate] = []

    for layer_idx, base_config in enumerate(typed_block_configs):
        numeric_candidates: list[Candidate] = []
        if include_self:
            numeric_candidates.append(
                _candidate_with_stats(
                    layer_idx=layer_idx,
                    block_config=base_config,
                    source_kind="self",
                    parent_checkpoint_identity=parent_checkpoint_identity,
                    source_identity=parent_checkpoint_identity,
                    metadata={"dense": True, "source": "parent_checkpoint"},
                    stats_cache=stats_cache,
                )
            )
        numeric_candidates.extend(
            _sorted_slice_candidates(
                layer_idx,
                base_config,
                axis_specs,
                parent_checkpoint_identity=parent_checkpoint_identity,
                stats_cache=stats_cache,
            )
        )
        candidates.extend(numeric_candidates)
        if include_noops:
            if no_op_specs.get("cartesian"):
                for numeric_candidate in numeric_candidates:
                    candidates.extend(
                        _cartesian_no_op_candidates(
                            numeric_candidate,
                            no_op_specs,
                            parent_checkpoint_identity=parent_checkpoint_identity,
                            stats_cache=stats_cache,
                        )
                    )
            else:
                candidates.extend(
                    _no_op_candidates(
                        layer_idx,
                        base_config,
                        no_op_specs,
                        parent_checkpoint_identity=parent_checkpoint_identity,
                        stats_cache=stats_cache,
                    )
                )

    deduplicated = list(_dedupe_candidates(candidates))
    if hidden_width is not None:
        if int(hidden_width) <= 0:
            raise ValueError("hidden_width must be positive")
        deduplicated = [
            replace(candidate, hidden_width=int(hidden_width))
            for candidate in deduplicated
        ]
    return deduplicated


def discover_bypass_checkpoints(puzzle_dir: str | Path) -> tuple[Path, ...]:
    """Return bypass checkpoint dirs from ``ckpts/bypass*`` and ``ckpts/bypass*/symlinks``."""
    ckpts_dir = Path(puzzle_dir) / "ckpts"
    if not ckpts_dir.exists():
        return ()
    candidates: set[Path] = set()
    for bypass_root in ckpts_dir.glob("bypass*"):
        if (bypass_root / "config.json").exists():
            candidates.add(bypass_root)
        symlinks_dir = bypass_root / "symlinks"
        if symlinks_dir.exists():
            for child in symlinks_dir.iterdir():
                if (child / "config.json").exists():
                    candidates.add(child)
        if bypass_root.is_dir():
            for child in bypass_root.iterdir():
                if child.name == "symlinks":
                    continue
                if (child / "config.json").exists():
                    candidates.add(child)
    return tuple(sorted(candidates, key=lambda path: str(path)))


def load_stats_identity_cache(
    stats_paths: Iterable[str | Path] | str | Path | None,
    *,
    hidden_width: int | None = None,
) -> dict[tuple[int | None, str], str]:
    """Index existing stats by ``(layer_idx, subblock_config_identity)``.

    The returned ids are stable hashes of the stats payloads. They intentionally
    do not include file paths, so moving a puzzle directory does not invalidate
    otherwise identical runtime or memory measurements.
    """
    if stats_paths is None:
        return {}
    if isinstance(stats_paths, (str, Path)):
        stats_paths = (stats_paths,)
    cache: dict[tuple[int | None, str], str] = {}
    for stats_path in stats_paths:
        path = Path(stats_path)
        if not path.exists():
            continue
        raw = json.loads(path.read_text())
        for stats_entry in _iter_stats_entries(raw):
            args = stats_entry.get("args") if isinstance(stats_entry, dict) else None
            entry_width = (args or {}).get("n_embd")
            if hidden_width is not None and (
                entry_width is None or int(entry_width) != int(hidden_width)
            ):
                continue
            for subblock_entry in _iter_subblock_stats(stats_entry):
                subblock_config = _stats_subblock_config(subblock_entry)
                if subblock_config is None:
                    continue
                layer_idx = subblock_entry.get("parent_layer_index")
                layer_key = int(layer_idx) if layer_idx is not None else None
                subblock_id = stable_hash(subblock_config, prefix="subblock_config")
                stats_id = stable_hash(
                    {
                        "args": args or {},
                        "subblock": subblock_entry,
                    },
                    prefix="stats",
                )
                cache[(layer_key, subblock_id)] = stats_id
                cache[(None, subblock_id)] = stats_id
    return cache


def _find_block_configs(config: dict[str, Any]) -> list[dict[str, Any]] | None:
    if isinstance(config.get("block_configs"), list):
        return config["block_configs"]
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and isinstance(text_config.get("block_configs"), list):
        return text_config["block_configs"]
    return None


def _checkpoint_identity(checkpoint_dir: Path) -> Identity:
    config = json.loads((checkpoint_dir / "config.json").read_text())
    return cache_key(
        "checkpoint_config",
        {"block_configs": _find_block_configs(config), "architectures": config.get("architectures")},
        {
            "base_architecture": config.get("base_architecture"),
            "model_type": config.get("model_type"),
        },
    )


def _default_stats_paths(puzzle_dir: Path) -> tuple[Path, ...]:
    return tuple(puzzle_dir / filename for filename in _DEFAULT_STATS_FILENAMES)


def _normalise_axis_specs(search_space: dict[str, Any]) -> dict[str, dict[str, Any]]:
    search_space = _plain_config(search_space)
    raw_axes = search_space.get("axes") if isinstance(search_space.get("axes"), dict) else search_space
    specs: dict[str, dict[str, Any]] = {}
    for axis_id, raw_spec in dict(raw_axes or {}).items():
        if axis_id == "no_op" or not isinstance(raw_spec, dict):
            continue
        target = _AXIS_TO_TARGET.get(axis_id)
        if target is None:
            continue
        values = tuple(raw_spec.get("sizes") or raw_spec.get("values") or ())
        ratios = tuple(raw_spec.get("ratios") or ())
        enabled = bool(raw_spec.get("enabled", bool(values or ratios)))
        if not enabled:
            continue
        specs[axis_id] = {
            "subblock_kind": target[0],
            "field": target[1],
            "values": values,
            "ratios": ratios,
        }
    return specs


def _normalise_no_op_specs(search_space: dict[str, Any]) -> dict[str, Any]:
    search_space = _plain_config(search_space)
    raw = search_space.get("no_op") or {}
    if not isinstance(raw, dict):
        return {}
    legacy_subblocks = tuple(
        kind for kind, enabled in raw.items() if kind in SUBBLOCK_CLS_DICT and bool(enabled)
    )
    if "subblocks" in raw or "whole_block" in raw:
        return {
            "subblocks": tuple(raw.get("subblocks") or legacy_subblocks),
            "whole_block": bool(raw.get("whole_block", False)),
            "cartesian": bool(raw.get("cartesian", False)),
        }
    return {
        "subblocks": legacy_subblocks,
        "whole_block": bool(raw.get("whole_block", False)),
        "cartesian": bool(raw.get("cartesian", False)),
    }


def _plain_config(value: Any) -> Any:
    if DictConfig is not None and ListConfig is not None and isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _sorted_slice_candidates(
    layer_idx: int,
    base_config: BlockConfig,
    axis_specs: dict[str, dict[str, Any]],
    *,
    parent_checkpoint_identity: str,
    stats_cache: dict[tuple[int | None, str], str] | None,
) -> Iterator[Candidate]:
    per_axis_options: list[tuple[str, str, str, tuple[Any, ...]]] = []
    for axis_id, spec in axis_specs.items():
        subblock = base_config.get_subblock(spec["subblock_kind"])
        if subblock is None or subblock.no_op:
            continue
        base_value = _axis_base_value(subblock, spec["field"])
        if base_value is None:
            continue
        values = _axis_values(base_value, spec)
        if not values:
            continue
        per_axis_options.append((axis_id, spec["subblock_kind"], spec["field"], values))

    for edits in _axis_edit_product(per_axis_options):
        candidate_config, changed_axes = _apply_axis_edits(base_config, edits)
        if not changed_axes or candidate_config.to_dict() == base_config.to_dict():
            continue
        yield _candidate_with_stats(
            layer_idx=layer_idx,
            block_config=candidate_config,
            source_kind="sorted_slice",
            parent_checkpoint_identity=parent_checkpoint_identity,
            source_identity=stable_hash(
                {"parent": parent_checkpoint_identity, "layer_idx": layer_idx, "axes": changed_axes},
                prefix="sorted_slice",
            ),
            metadata={"slice_axes": changed_axes, "virtual": True},
            stats_cache=stats_cache,
        )
def _axis_values(base_value: Any, spec: dict[str, Any]) -> tuple[Any, ...]:
    values: list[Any] = []
    for raw_value in spec["values"]:
        values.append(_resolve_axis_value(base_value, raw_value, spec["field"]))
    for ratio in spec["ratios"]:
        values.append(_resolve_ratio_value(base_value, ratio))
    values.append(base_value)
    return tuple(
        dict.fromkeys(
            value
            for value in values
            if _valid_axis_value(base_value, value, spec["field"])
        )
    )


def _resolve_axis_value(base_value: Any, raw_value: Any, field_name: str) -> Any:
    if field_name not in _VARIANT_FIELDS and isinstance(raw_value, float) and 0 < raw_value <= 1:
        return _resolve_ratio_value(base_value, raw_value)
    return int(raw_value) if isinstance(base_value, int) and isinstance(raw_value, float) else raw_value


def _resolve_ratio_value(base_value: Any, ratio: Any) -> Any:
    if not isinstance(base_value, int):
        return base_value
    return max(1, min(base_value, int(round(base_value * float(ratio)))))


def _valid_axis_value(base_value: Any, value: Any, field_name: str) -> bool:
    if field_name == "sliding_window_size":
        if value == "full":
            return base_value == "full"
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            return False
        return base_value == "full" or (
            isinstance(base_value, int)
            and not isinstance(base_value, bool)
            and value <= base_value
        )
    if isinstance(base_value, int):
        return isinstance(value, int) and 0 < value <= base_value
    return value is not None


def _axis_base_value(subblock: SubblockConfig, field_name: str) -> Any:
    if field_name == "q_heads_per_group":
        q = getattr(subblock, "num_query_heads", None)
        kv = getattr(subblock, "num_kv_heads", None)
        if q is None or kv is None:
            return None
        return int(q) // int(kv)
    if field_name == "gdn_key_groups":
        return getattr(subblock, "num_groups", None)
    if field_name == "gdn_value_heads_per_group":
        heads = getattr(subblock, "num_heads", None)
        groups = getattr(subblock, "num_groups", None)
        return None if heads is None or groups is None else int(heads) // int(groups)
    return getattr(subblock, field_name, None)


def _axis_edit_product(
    per_axis_options: list[tuple[str, str, str, tuple[Any, ...]]],
) -> Iterator[tuple[tuple[str, str, str, Any], ...]]:
    if not per_axis_options:
        return
    edits: list[tuple[str, str, str, Any]] = []

    def _walk(index: int) -> Iterator[tuple[tuple[str, str, str, Any], ...]]:
        if index == len(per_axis_options):
            yield tuple(edits)
            return
        axis_id, subblock_kind, field_name, values = per_axis_options[index]
        for value in values:
            edits.append((axis_id, subblock_kind, field_name, value))
            yield from _walk(index + 1)
            edits.pop()

    yield from _walk(0)


def _apply_axis_edits(
    block_config: BlockConfig,
    edits: tuple[tuple[str, str, str, Any], ...],
) -> tuple[BlockConfig, dict[str, Any]]:
    edits_by_kind: dict[str, dict[str, Any]] = {}
    axis_values: dict[str, Any] = {}
    for axis_id, subblock_kind, field_name, value in edits:
        subblock = block_config.get_subblock(subblock_kind)
        if subblock is None or _axis_base_value(subblock, field_name) == value:
            continue
        if subblock_kind == "mamba" and axis_id in {
            "gdn_key_groups",
            "gdn_value_heads_per_group",
        }:
            field_edits = edits_by_kind.setdefault(subblock_kind, {})
            teacher_groups = int(getattr(subblock, "num_groups"))
            teacher_ratio = int(getattr(subblock, "num_heads")) // teacher_groups
            target_groups = int(value) if axis_id == "gdn_key_groups" else int(
                field_edits.get("num_groups", teacher_groups)
            )
            target_ratio = int(value) if axis_id == "gdn_value_heads_per_group" else teacher_ratio
            field_edits["num_groups"] = target_groups
            field_edits["num_heads"] = target_groups * target_ratio
        elif subblock_kind == "attention" and field_name == "q_heads_per_group":
            field_edits = edits_by_kind.setdefault(subblock_kind, {})
            kv = field_edits.get("num_kv_heads", getattr(subblock, "num_kv_heads", None))
            if kv is None:
                continue
            field_edits["num_query_heads"] = int(kv) * int(value)
        elif subblock_kind == "attention" and field_name == "num_kv_heads":
            q = getattr(subblock, "num_query_heads", None)
            kv = getattr(subblock, "num_kv_heads", None)
            field_edits = edits_by_kind.setdefault(subblock_kind, {})
            field_edits["num_kv_heads"] = value
            if q is not None and kv is not None and "num_query_heads" not in field_edits:
                field_edits["num_query_heads"] = int(value) * (int(q) // int(kv))
        else:
            edits_by_kind.setdefault(subblock_kind, {})[field_name] = value
        axis_values[axis_id] = value

    attention_edits = edits_by_kind.get("attention")
    if attention_edits is not None and "num_kv_heads" in attention_edits:
        subblock = block_config.get_subblock("attention")
        if subblock is not None:
            target_hpg = next(
                (
                    int(value)
                    for _, edit_kind, field_name, value in edits
                    if edit_kind == "attention" and field_name == "q_heads_per_group"
                ),
                None,
            )
            if target_hpg is None:
                q = getattr(subblock, "num_query_heads", None)
                kv = getattr(subblock, "num_kv_heads", None)
                if q is not None and kv is not None:
                    target_hpg = int(q) // int(kv)
            if target_hpg is not None:
                attention_edits["num_query_heads"] = int(attention_edits["num_kv_heads"]) * target_hpg

    gdn_edits = [edit for edit in edits if edit[0] in {"gdn_key_groups", "gdn_value_heads_per_group"}]
    if gdn_edits:
        subblock = block_config.get_subblock("mamba")
        teacher_groups = int(getattr(subblock, "num_groups"))
        teacher_ratio = int(getattr(subblock, "num_heads")) // teacher_groups
        target_groups = next(
            (int(value) for axis, _, _, value in gdn_edits if axis == "gdn_key_groups"),
            teacher_groups,
        )
        target_ratio = next(
            (
                int(value)
                for axis, _, _, value in gdn_edits
                if axis == "gdn_value_heads_per_group"
            ),
            teacher_ratio,
        )
        edits_by_kind.setdefault("mamba", {}).update(
            num_groups=target_groups,
            num_heads=target_groups * target_ratio,
        )

    candidate_config = block_config
    changed_axes: dict[str, Any] = {}
    for subblock_kind, field_edits in edits_by_kind.items():
        subblock = candidate_config.get_subblock(subblock_kind)
        if subblock is None:
            continue
        try:
            replacement = replace(subblock, **field_edits)
        except ValueError:
            return block_config, {}
        candidate_config = candidate_config.with_subblock(replacement)
        changed_axes.update(
            {
                axis_id: value
                for axis_id, edit_kind, field_name, value in edits
                if edit_kind == subblock_kind
                and (
                    field_name in field_edits
                    or field_name == "q_heads_per_group"
                    or axis_id in {"gdn_key_groups", "gdn_value_heads_per_group"}
                )
                and axis_values.get(axis_id) == value
            }
        )
    return candidate_config, changed_axes


def _no_op_candidates(
    layer_idx: int,
    base_config: BlockConfig,
    no_op_specs: dict[str, Any],
    *,
    parent_checkpoint_identity: str,
    stats_cache: dict[tuple[int | None, str], str] | None,
) -> Iterator[Candidate]:
    enabled_subblocks = set(no_op_specs.get("subblocks") or ())
    if no_op_specs.get("whole_block"):
        no_op_block = BlockConfig(
            subblock_configs=tuple(_make_no_op_subblock(subblock) for subblock in base_config.subblock_configs)
        )
        yield _candidate_with_stats(
            layer_idx=layer_idx,
            block_config=no_op_block,
            source_kind="no_op",
            parent_checkpoint_identity=parent_checkpoint_identity,
            source_identity=stable_hash(
                {"parent": parent_checkpoint_identity, "layer_idx": layer_idx, "scope": "block"},
                prefix="no_op",
            ),
            metadata={"no_op_scope": "block", "virtual": True},
            stats_cache=stats_cache,
        )
    for subblock in base_config.subblock_configs:
        if subblock.kind not in enabled_subblocks:
            continue
        no_op_block = base_config.with_subblock(_make_no_op_subblock(subblock))
        yield _candidate_with_stats(
            layer_idx=layer_idx,
            block_config=no_op_block,
            source_kind="no_op",
            parent_checkpoint_identity=parent_checkpoint_identity,
            source_identity=stable_hash(
                {
                    "parent": parent_checkpoint_identity,
                    "layer_idx": layer_idx,
                    "scope": "subblock",
                    "subblock": subblock.name,
                },
                prefix="no_op",
            ),
            metadata={"no_op_scope": "subblock", "subblock": subblock.name, "virtual": True},
            stats_cache=stats_cache,
        )


def _cartesian_no_op_candidates(
    numeric_candidate: Candidate,
    no_op_specs: dict[str, Any],
    *,
    parent_checkpoint_identity: str,
    stats_cache: dict[tuple[int | None, str], str] | None,
) -> Iterator[Candidate]:
    """Apply every configured non-empty no-op subset to one numeric candidate."""
    enabled = set(no_op_specs.get("subblocks") or ())
    eligible = [
        subblock
        for subblock in numeric_candidate.block_config.subblock_configs
        if subblock.kind in enabled and not subblock.no_op
    ]
    active_kinds = {
        subblock.kind
        for subblock in numeric_candidate.block_config.subblock_configs
        if not subblock.no_op
    }
    for mask in range(1, 1 << len(eligible)):
        block_config = numeric_candidate.block_config
        disabled = []
        for index, subblock in enumerate(eligible):
            if not mask & (1 << index):
                continue
            block_config = block_config.with_subblock(_make_no_op_subblock(subblock))
            disabled.append(subblock.kind)
        if not no_op_specs.get("whole_block") and set(disabled) == active_kinds:
            continue
        source_identity = stable_hash(
            {
                "parent": parent_checkpoint_identity,
                "layer_idx": numeric_candidate.layer_idx,
                "block_config": block_config.to_dict(),
            },
            prefix="no_op_cartesian",
        )
        yield _candidate_with_stats(
            layer_idx=numeric_candidate.layer_idx,
            block_config=block_config,
            source_kind="no_op",
            parent_checkpoint_identity=parent_checkpoint_identity,
            source_identity=source_identity,
            metadata={
                "no_op_scope": "cartesian",
                "disabled_subblocks": sorted(disabled),
                "numeric_source": numeric_candidate.identity.value,
                "virtual": True,
            },
            stats_cache=stats_cache,
        )


def _make_no_op_subblock(subblock: SubblockConfig) -> SubblockConfig:
    cls = SUBBLOCK_CLS_DICT[subblock.kind]
    return cls(kind=subblock.kind, name=subblock.name, no_op=True)


def _build_bypass_candidates(
    puzzle_dir: Path,
    *,
    parent_block_configs: tuple[BlockConfig, ...],
    parent_checkpoint_identity: str,
    stats_cache: dict[tuple[int | None, str], str] | None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for checkpoint_dir in discover_bypass_checkpoints(puzzle_dir):
        try:
            block_configs = load_block_configs_from_checkpoint(checkpoint_dir)
        except (FileNotFoundError, ValueError):
            continue
        source_identity = _checkpoint_identity(checkpoint_dir).value
        changed_layers = [
            layer_idx
            for layer_idx, block_config in enumerate(block_configs)
            if layer_idx >= len(parent_block_configs)
            or block_config.to_dict() != parent_block_configs[layer_idx].to_dict()
        ]
        layer_indices = changed_layers or list(range(len(block_configs)))
        for layer_idx in layer_indices:
            candidates.append(
                _candidate_with_stats(
                    layer_idx=layer_idx,
                    block_config=block_configs[layer_idx],
                    source_kind="bypass",
                    parent_checkpoint_identity=parent_checkpoint_identity,
                    source_identity=source_identity,
                    metadata={
                        "checkpoint_dir": str(checkpoint_dir),
                        "changed_from_parent": layer_idx in changed_layers,
                    },
                    stats_cache=stats_cache,
                )
            )
    return candidates


def _candidate_with_stats(
    *,
    layer_idx: int,
    block_config: BlockConfig,
    source_kind: str,
    parent_checkpoint_identity: str,
    source_identity: str | None,
    metadata: dict[str, Any],
    stats_cache: dict[tuple[int | None, str], str] | None,
) -> Candidate:
    stats_ids = _stats_ids_for_block(layer_idx, block_config, stats_cache or {})
    stats_identity = (
        stable_hash({"layer_idx": layer_idx, "stats_ids": stats_ids}, prefix="stats_bundle")
        if stats_ids
        else None
    )
    candidate_metadata = dict(metadata)
    if stats_ids:
        candidate_metadata["stats_identities"] = stats_ids
    return Candidate(
        layer_idx=layer_idx,
        block_config=block_config,
        source_kind=source_kind,
        parent_checkpoint_identity=parent_checkpoint_identity,
        source_identity=source_identity,
        cost_identity=stats_identity,
        stats_identity=stats_identity,
        metadata=candidate_metadata,
    )


def _stats_ids_for_block(
    layer_idx: int,
    block_config: BlockConfig,
    stats_cache: dict[tuple[int | None, str], str],
) -> dict[str, str]:
    stats_ids: dict[str, str] = {}
    for subblock_ref in block_config.subblocks():
        subblock_id = stable_hash(subblock_ref.config, prefix="subblock_config")
        stats_id = stats_cache.get((layer_idx, subblock_id)) or stats_cache.get((None, subblock_id))
        if stats_id is not None:
            stats_ids[subblock_ref.name] = stats_id
    return stats_ids


def _dedupe_candidates(candidates: Iterable[Candidate]) -> Iterator[Candidate]:
    seen: set[str] = set()
    for candidate in candidates:
        candidate_id = candidate.identity.value
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        yield candidate


def _iter_stats_entries(raw: Any) -> Iterator[dict[str, Any]]:
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                yield item
    elif isinstance(raw, dict):
        if "subblocks" in raw:
            yield raw
        for key in ("entries", "stats", "runs"):
            value = raw.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item


def _iter_subblock_stats(stats_entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    subblocks = stats_entry.get("subblocks")
    if isinstance(subblocks, list):
        for subblock_entry in subblocks:
            if isinstance(subblock_entry, dict):
                yield subblock_entry


def _stats_subblock_config(entry: dict[str, Any]) -> SubblockConfig | None:
    raw_config = entry.get("subblock_config")
    if isinstance(raw_config, SubblockConfig):
        return raw_config
    if not isinstance(raw_config, dict):
        return None
    kind = raw_config.get("kind")
    if kind not in SUBBLOCK_CLS_DICT:
        return None
    return SUBBLOCK_CLS_DICT[kind](**raw_config)
