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

"""Derive replace-one-subblock scoring solutions from a full block library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

from ..block_config import BlockConfig, SubblockConfig
from .replacement_utils import parse_layer_replacement

__all__ = [
    "build_subblock_replacement_payload",
    "build_subblock_replacement_solutions",
]


SubblockIdentity = tuple[str, str]


def build_subblock_replacement_payload(
    replacement_library: dict[str, Any],
    teacher_block_configs: Sequence[BlockConfig],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build an annotated replace-one-subblock catalog and its manifest."""

    entries = replacement_library.get("entries")
    if replacement_library.get("version") != 2 or not isinstance(entries, list):
        raise ValueError("subblock scoring requires a v2 replacement library with entries")
    solutions = build_subblock_replacement_solutions(entries, teacher_block_configs)
    annotation = {
        key: replacement_library[key]
        for key in ("hidden_width", "teacher_hidden_width", "scenario")
        if key in replacement_library
    }
    for solution in solutions:
        solution.update(annotation)
    manifest = {
        "version": 1,
        "mode": "replace_one_subblock",
        "canonical_entry_count": len(entries),
        "subblock_solution_count": len(solutions),
        "teacher_layer_count": len(teacher_block_configs),
        "full_search_space_preserved": True,
        **annotation,
    }
    return manifest, solutions


def _subblocks_by_identity(block: BlockConfig) -> dict[SubblockIdentity, SubblockConfig]:
    by_identity = {(subblock.kind, subblock.name): subblock for subblock in block.subblock_configs}
    if len(by_identity) != len(block.subblock_configs):
        raise ValueError("BlockConfig contains duplicate (kind, name) subblock identities")
    return by_identity


def _one_subblock_variant(
    teacher: BlockConfig,
    candidate_by_identity: dict[SubblockIdentity, SubblockConfig],
    changed_identity: SubblockIdentity,
) -> BlockConfig:
    return BlockConfig(
        subblock_configs=tuple(
            candidate_by_identity[(subblock.kind, subblock.name)]
            if (subblock.kind, subblock.name) == changed_identity
            else subblock
            for subblock in teacher.subblock_configs
        )
    )


def _replacement(layer_idx: int, block: BlockConfig) -> dict[str, Any]:
    return {
        "weight_paths": [],
        "parent_layer_indices": [layer_idx],
        "child_block_configs": [block.to_dict()],
    }


def _parse_one_block_replacement(raw_replacement: dict[str, Any]) -> tuple[int, BlockConfig]:
    replacement = parse_layer_replacement(raw_replacement)
    layers = replacement["parent_layer_indices"]
    blocks = replacement["child_block_configs"]
    if len(layers) != 1 or len(blocks) != 1:
        raise ValueError("subblock scoring requires one-layer, one-block library entries")
    return int(layers[0]), blocks[0]


def build_subblock_replacement_solutions(
    layer_replacements: Iterable[dict[str, Any]],
    teacher_block_configs: Sequence[BlockConfig],
) -> list[dict[str, Any]]:
    """Build deterministic solutions that each alter exactly one named subblock.

    The input remains the canonical full-block library. Every distinct subblock
    value observed in that library is paired with teacher values for all companion
    subblocks. This preserves arbitrary subblock kinds and multiple named
    subblocks of the same kind without encoding model-family assumptions.
    """

    teachers = tuple(teacher_block_configs)
    teacher_maps = tuple(_subblocks_by_identity(block) for block in teachers)
    variants: dict[tuple[int, BlockConfig], SubblockIdentity] = {}
    for raw_replacement in layer_replacements:
        layer_idx, candidate = _parse_one_block_replacement(raw_replacement)
        if not 0 <= layer_idx < len(teachers):
            raise ValueError(
                f"replacement layer {layer_idx} is outside teacher depth {len(teachers)}"
            )
        teacher = teachers[layer_idx]
        teacher_by_identity = teacher_maps[layer_idx]
        candidate_by_identity = _subblocks_by_identity(candidate)
        if candidate_by_identity.keys() != teacher_by_identity.keys():
            raise ValueError(
                f"layer {layer_idx} candidate subblock identities do not match the teacher: "
                f"candidate={sorted(candidate_by_identity)}, teacher={sorted(teacher_by_identity)}"
            )
        for identity, candidate_subblock in candidate_by_identity.items():
            if candidate_subblock == teacher_by_identity[identity]:
                continue
            variant = _one_subblock_variant(teacher, candidate_by_identity, identity)
            variants[(layer_idx, variant)] = identity

    teacher_replacements = [
        _replacement(layer_idx, block) for layer_idx, block in enumerate(teachers)
    ]
    solutions: list[dict[str, Any]] = []
    for (layer_idx, block), identity in sorted(
        variants.items(), key=lambda item: (item[0][0], str(item[0][1]), item[1])
    ):
        single_replacement = _replacement(layer_idx, block)
        chosen_replacements = list(teacher_replacements)
        chosen_replacements[layer_idx] = single_replacement
        solutions.append(
            {
                "single_sequence_replacement": single_replacement,
                "chosen_replacements": chosen_replacements,
                "block_configs": [
                    replacement["child_block_configs"][0] for replacement in chosen_replacements
                ],
                "subblock_replacement": {
                    "layer_idx": layer_idx,
                    "kind": identity[0],
                    "name": identity[1],
                },
            }
        )
    return solutions
