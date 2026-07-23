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
"""Build a v2 virtual replacement library from one sorted-teacher checkpoint."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from modelopt.torch.utils import json_dump

from ..anymodel.model_descriptor import ModelDescriptor, ModelDescriptorFactory
from ..block_config import (
    SUBBLOCK_CLS_DICT,
    AttentionConfig,
    BlockConfig,
    FFNConfig,
    SubblockConfig,
    maybe_cast_block_configs,
)
from ..candidates import build_candidate_library
from ..mip.utils import sort_replacements
from ..tools.checkpoint_utils_hf import load_model_config
from ..tools.logger import mprint
from ..utils.parsing import format_global_config
from .replacement_utils import is_replacement_identical_to_teacher, parse_layer_replacement
from .subblock_scoring import build_subblock_replacement_payload

__all__ = [
    "build_replacement_library_from_sorted_teacher",
    "launch_build_replacement_library",
]


SORTED_TEACHER_DIR_NAME = "sorted_teacher"
ELASTIC_SORTED_TEACHER_DIR_NAME = "elastic_sorted_teacher"
CHECKPOINTS_DIR_NAME = "ckpts"
SUBBLOCK_KINDS = ("attention", "mla", "mamba", "ffn", "moe")
ATTENTION_LIKE_KINDS = frozenset(("attention", "mla", "mamba"))
FFN_LIKE_KINDS = frozenset(("ffn", "moe"))


def _replace_subblock(
    block_config: BlockConfig,
    subblock_config: SubblockConfig,
    *,
    replace_kinds: frozenset[str],
) -> BlockConfig:
    return block_config.with_subblock(subblock_config, replace_kinds=replace_kinds)


def _get_subblock_for_group(block_config: BlockConfig, kinds: frozenset[str]) -> SubblockConfig | None:
    for subblock in block_config.subblock_configs:
        if subblock.kind in kinds:
            return subblock
    return None


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


def _target_ints(values: Any) -> list[int]:
    return [int(value) for value in list(values or []) if value is not None]


def _target_pairs(values: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for value in list(values or []):
        if value is None:
            continue
        q, kv = value
        pairs.append((int(q), int(kv)))
    return pairs


def _entry(layer_idx: int, block_config: BlockConfig) -> dict[str, Any]:
    return {
        "weight_paths": [],
        "parent_layer_indices": [layer_idx],
        "child_block_configs": [_to_serializable(block_config)],
    }


def _block_entries_for_layer(
    *,
    layer_idx: int,
    block_config: BlockConfig,
    teacher_ffn_size: int | None,
    teacher_q: int,
    teacher_kv: int,
    ffn_targets: list[int],
    attn_targets: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], int]:
    """Build identity + pruned virtual candidates for one layer."""
    attn_like = _get_subblock_for_group(block_config, ATTENTION_LIKE_KINDS)
    ffn_like = _get_subblock_for_group(block_config, FFN_LIKE_KINDS)
    ffn_prunable = isinstance(ffn_like, FFNConfig)
    attn_prunable = isinstance(attn_like, AttentionConfig)

    lyr_ffn = getattr(ffn_like, "intermediate_size", None) or teacher_ffn_size
    lyr_q = getattr(attn_like, "num_query_heads", None) or teacher_q
    lyr_kv = getattr(attn_like, "num_kv_heads", None) or teacher_kv

    ffn_values = [lyr_ffn]
    if ffn_prunable and lyr_ffn is not None:
        ffn_values = sorted({lyr_ffn, *[target for target in ffn_targets if target <= lyr_ffn]})

    attn_values = [(lyr_q, lyr_kv)]
    if attn_prunable:
        attn_values = sorted(
            {(lyr_q, lyr_kv), *[(q, kv) for q, kv in attn_targets if q <= lyr_q and kv <= lyr_kv]}
        )

    entries: list[dict[str, Any]] = []
    student_count = 0
    for ffn_size in ffn_values:
        for q, kv in attn_values:
            child_cfg = block_config
            if ffn_prunable and ffn_size != lyr_ffn:
                child_cfg = _replace_subblock(
                    child_cfg,
                    dataclasses.replace(ffn_like, intermediate_size=ffn_size),
                    replace_kinds=FFN_LIKE_KINDS,
                )
            if attn_prunable and (q != lyr_q or kv != lyr_kv):
                child_attn = dataclasses.replace(
                    attn_like,
                    num_query_heads=(None if q == teacher_q else q),
                    num_kv_heads=kv,
                )
                child_cfg = _replace_subblock(
                    child_cfg,
                    child_attn,
                    replace_kinds=ATTENTION_LIKE_KINDS,
                )
            if child_cfg != block_config:
                student_count += 1
            entries.append(_entry(layer_idx, child_cfg))
    return entries, student_count


def _no_op_entries_for_layer(layer_idx: int, block_config: BlockConfig) -> list[dict[str, Any]]:
    """Add one attention-like and one FFN-like no-op candidate for depth/grid budgeting."""
    attn_like = _get_subblock_for_group(block_config, ATTENTION_LIKE_KINDS)
    ffn_like = _get_subblock_for_group(block_config, FFN_LIKE_KINDS)
    no_attn_kind = attn_like.kind if attn_like is not None else "attention"
    no_ffn_kind = ffn_like.kind if ffn_like is not None else "ffn"
    no_attn = _replace_subblock(
        block_config,
        SUBBLOCK_CLS_DICT[no_attn_kind](no_op=True),
        replace_kinds=ATTENTION_LIKE_KINDS,
    )
    no_ffn = _replace_subblock(
        block_config,
        SUBBLOCK_CLS_DICT[no_ffn_kind](no_op=True),
        replace_kinds=FFN_LIKE_KINDS,
    )
    return [_entry(layer_idx, no_ffn), _entry(layer_idx, no_attn)]


def _no_op_block_variants(block_config: BlockConfig) -> list[BlockConfig]:
    """Expand no-op over the full numeric mixer/FFN grid, including joint no-op."""
    attn_like = _get_subblock_for_group(block_config, ATTENTION_LIKE_KINDS)
    ffn_like = _get_subblock_for_group(block_config, FFN_LIKE_KINDS)
    variants: list[BlockConfig] = []
    no_attn = None
    no_ffn = None
    if attn_like is not None and not attn_like.no_op:
        no_attn = _replace_subblock(
            block_config,
            SUBBLOCK_CLS_DICT[attn_like.kind](kind=attn_like.kind, name=attn_like.name, no_op=True),
            replace_kinds=ATTENTION_LIKE_KINDS,
        )
        variants.append(no_attn)
    if ffn_like is not None and not ffn_like.no_op:
        no_ffn = _replace_subblock(
            block_config,
            SUBBLOCK_CLS_DICT[ffn_like.kind](kind=ffn_like.kind, name=ffn_like.name, no_op=True),
            replace_kinds=FFN_LIKE_KINDS,
        )
        variants.append(no_ffn)
    if no_attn is not None and ffn_like is not None and not ffn_like.no_op:
        variants.append(
            _replace_subblock(
                no_attn,
                SUBBLOCK_CLS_DICT[ffn_like.kind](
                    kind=ffn_like.kind,
                    name=ffn_like.name,
                    no_op=True,
                ),
                replace_kinds=FFN_LIKE_KINDS,
            )
        )
    return variants


def _build_single_sequence_replacement_solutions(
    layer_replacements: list[dict],
    teacher_checkpoint_dir: Path,
    descriptor: ModelDescriptor,
) -> list[dict]:
    teacher_model_config = load_model_config(
        teacher_checkpoint_dir, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    n_layer = descriptor.get_language_model_config(teacher_model_config).num_hidden_layers

    teacher_replacements: dict[int, dict] = {}
    student_replacements = []
    for layer_replacement in layer_replacements:
        if is_replacement_identical_to_teacher(layer_replacement, teacher_model_config):
            block_idx = layer_replacement["parent_layer_indices"][0]
            teacher_replacements[block_idx] = layer_replacement
        else:
            student_replacements.append(layer_replacement)

    represented = sorted(teacher_replacements)
    if represented != list(range(n_layer)):
        raise ValueError(f"Replacement library is missing teacher entries: {n_layer=}, {represented=}")

    solutions = []
    for layer_replacement in sort_replacements(student_replacements):
        missing = sorted(set(range(n_layer)) - set(layer_replacement["parent_layer_indices"]))
        chosen_replacements = sort_replacements(
            [layer_replacement] + [teacher_replacements[block_idx] for block_idx in missing]
        )
        block_configs = [
            block_config
            for replacement in chosen_replacements
            for block_config in replacement["child_block_configs"]
        ]
        solutions.append(
            {
                "single_sequence_replacement": layer_replacement,
                "chosen_replacements": chosen_replacements,
                "block_configs": block_configs,
            }
        )
    return solutions


def build_replacement_library_from_sorted_teacher(
    master_puzzle_dir: Path | str,
    sorted_teacher_dir: Path | str,
    descriptor: ModelDescriptor,
    ffn_targets: list[int] | None = None,
    attn_targets: list[tuple[int, int]] | None = None,
    search_space: dict[str, Any] | None = None,
    include_noops: bool = True,
    hidden_width: int | None = None,
) -> None:
    """Build ``replacement_library.json`` and one-replacement scoring solutions.

    The library is virtual: every entry has empty ``weight_paths`` and is
    materialized from ``sorted_teacher_dir`` by slicing or no-op replacement.
    """
    master_puzzle_dir = Path(master_puzzle_dir)
    sorted_teacher_dir = Path(sorted_teacher_dir).resolve()
    master_puzzle_dir.mkdir(parents=True, exist_ok=True)

    teacher_config = load_model_config(
        sorted_teacher_dir, trust_remote_code=descriptor.requires_trust_remote_code()
    )
    lm = descriptor.get_language_model_config(teacher_config)
    teacher_hidden_width = int(lm.hidden_size)
    if hidden_width is None:
        hidden_width = teacher_hidden_width
    hidden_width = int(hidden_width)
    if not 0 < hidden_width <= teacher_hidden_width:
        raise ValueError(
            f"hidden_width must be in [1, {teacher_hidden_width}], got {hidden_width}"
        )
    block_configs = maybe_cast_block_configs(teacher_config.block_configs)
    teacher_ffn_size = getattr(lm, "intermediate_size", None)
    teacher_q = int(lm.num_attention_heads)
    teacher_kv = int(getattr(lm, "num_key_value_heads", teacher_q))

    entries_serializable: list[dict[str, Any]] = []
    student_count = 0
    if search_space:
        candidates = build_candidate_library(
            block_configs,
            search_space=search_space,
            parent_checkpoint_identity=str(sorted_teacher_dir),
            include_self=True,
            include_noops=include_noops,
            stats_cache=None,
            hidden_width=hidden_width,
        )
        seen_entries: set[str] = set()
        for candidate in candidates:
            candidate_config = candidate.block_config
            entry = _entry(candidate.layer_idx, candidate_config)
            key = json.dumps(_to_serializable(entry), sort_keys=True)
            if key in seen_entries:
                continue
            seen_entries.add(key)
            entries_serializable.append(entry)
            if candidate_config.to_dict() != block_configs[candidate.layer_idx].to_dict():
                student_count += 1
    else:
        for layer_idx, block_config in enumerate(block_configs):
            entries, layer_student_count = _block_entries_for_layer(
                layer_idx=layer_idx,
                block_config=block_config,
                teacher_ffn_size=teacher_ffn_size,
                teacher_q=teacher_q,
                teacher_kv=teacher_kv,
                ffn_targets=ffn_targets or [],
                attn_targets=attn_targets or [],
            )
            entries_serializable.extend(entries)
            if include_noops:
                entries_serializable.extend(_no_op_entries_for_layer(layer_idx, block_config))
            student_count += layer_student_count

    library_v2 = {
        "version": 2,
        "sorted_teacher_dir": str(sorted_teacher_dir),
        "hidden_width": hidden_width,
        "teacher_hidden_width": teacher_hidden_width,
        "scenario": f"width-{hidden_width:04d}",
        "entries": entries_serializable,
    }
    json_dump(library_v2, master_puzzle_dir / "replacement_library.json")

    parsed_entries = [parse_layer_replacement(entry) for entry in entries_serializable]
    solutions = _build_single_sequence_replacement_solutions(
        parsed_entries,
        teacher_checkpoint_dir=sorted_teacher_dir,
        descriptor=descriptor,
    )
    for solution in solutions:
        solution["hidden_width"] = hidden_width
        solution["teacher_hidden_width"] = teacher_hidden_width
        solution["scenario"] = f"width-{hidden_width:04d}"
    json_dump(solutions, master_puzzle_dir / "single_sequence_replacement_solutions.json")

    subblock_manifest, subblock_solutions = build_subblock_replacement_payload(
        library_v2,
        block_configs,
    )
    subblock_solutions_path = master_puzzle_dir / "single_subblock_replacement_solutions.json"
    subblock_manifest_path = master_puzzle_dir / "subblock_replacement_manifest.json"
    subblock_manifest.update(
        {
            "replacement_library": str(
                (master_puzzle_dir / "replacement_library.json").resolve()
            ),
            "teacher_dir": str(sorted_teacher_dir),
            "solutions": str(subblock_solutions_path.resolve()),
        }
    )
    json_dump(subblock_solutions, subblock_solutions_path)
    json_dump(subblock_manifest, subblock_manifest_path)

    mprint(
        f"Sorted-teacher replacement library: {len(entries_serializable)} entries, "
        f"{student_count} pruned entries, {len(solutions)} one-replacement solutions, "
        f"{len(subblock_solutions)} one-subblock solutions, "
        f"hidden_width={hidden_width}, sorted_teacher_dir={sorted_teacher_dir}"
    )


def _resolve_sorted_teacher_source(
    puzzle_dir: Path,
    source_checkpoint_dir: str | Path | None = None,
) -> Path:
    if source_checkpoint_dir:
        source = Path(source_checkpoint_dir).resolve()
        if not (source / "config.json").exists():
            raise FileNotFoundError(
                "Configured build_replacement_library.source_checkpoint_dir is not a "
                f"complete HF checkpoint: {source}"
            )
        mprint(f"Using configured sorted/distilled supernet for the library: {source}")
        return source
    ckpts = puzzle_dir / CHECKPOINTS_DIR_NAME
    elastic = ckpts / ELASTIC_SORTED_TEACHER_DIR_NAME
    if (elastic / "config.json").exists():
        mprint(f"Using elastic sorted teacher for the library: {elastic}")
        return elastic
    sorted_teacher = ckpts / SORTED_TEACHER_DIR_NAME
    if not (sorted_teacher / "config.json").exists():
        raise FileNotFoundError(
            f"Sorted teacher not found at {sorted_teacher}. Run the sort stage before build_library."
        )
    return sorted_teacher


def launch_build_replacement_library(cfg: DictConfig) -> None:
    """Build the modern virtual replacement library."""
    mprint(f"Building replacement library for puzzle directory: {cfg.puzzle_dir}")
    mprint(
        "Build replacement library config: "
        f"{format_global_config(cfg.build_replacement_library, title='Build replacement library')}"
    )

    pruning_output = cfg.pruning.get("output", "sorted_teacher") if hasattr(cfg, "pruning") else "sorted_teacher"
    if pruning_output != "sorted_teacher":
        raise ValueError(
            "Only pruning.output=sorted_teacher is supported. "
            "Explicit-checkpoint replacement libraries have been removed."
        )

    descriptor = ModelDescriptorFactory.get(cfg.descriptor)
    ffn_targets = _target_ints(cfg.pruning.get("intermediate_size_list", []))
    raw_attn = cfg.pruning.get("attn_heads_list", None) or cfg.pruning.get("attention_groups_list", [])
    attn_targets = _target_pairs(raw_attn)
    source_dir = _resolve_sorted_teacher_source(
        Path(cfg.puzzle_dir),
        cfg.build_replacement_library.get("source_checkpoint_dir", None),
    )
    build_replacement_library_from_sorted_teacher(
        master_puzzle_dir=cfg.puzzle_dir,
        sorted_teacher_dir=source_dir,
        descriptor=descriptor,
        ffn_targets=ffn_targets,
        attn_targets=attn_targets,
        search_space=cfg.get("search_space", None),
        include_noops=bool(cfg.build_replacement_library.get("include_noops", True)),
        hidden_width=cfg.build_replacement_library.get("hidden_width", None),
    )
