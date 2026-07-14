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

"""Compose complete block metrics from isolated subblock measurements."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from ..block_config import BlockConfig, SubblockConfig
from .replacement_utils import parse_layer_replacement

__all__ = [
    "ComposedScoreRecord",
    "ComposedScoreTable",
    "compose_full_block_metrics",
    "composed_table_to_gathered_metrics",
]

SubblockIdentity = tuple[str, str]


def _subblocks(block: BlockConfig) -> dict[SubblockIdentity, SubblockConfig]:
    result = {(subblock.kind, subblock.name): subblock for subblock in block.subblock_configs}
    if len(result) != len(block.subblock_configs):
        raise ValueError("duplicate subblock identity in block config")
    return result


def _subblock_key(subblock: SubblockConfig) -> str:
    return json.dumps(subblock.to_dict(), sort_keys=True, separators=(",", ":"))


def _block_key(block: BlockConfig) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        sorted(
            (subblock.kind, subblock.name, _subblock_key(subblock))
            for subblock in block.subblock_configs
        )
    )


def _replacement(payload: Mapping[str, Any]) -> tuple[int, BlockConfig]:
    replacement = payload["single_sequence_replacement"]
    layers = replacement["parent_layer_indices"]
    blocks = replacement["child_block_configs"]
    if len(layers) != 1 or len(blocks) != 1:
        raise ValueError("score composition requires one-layer, one-block replacements")
    block = blocks[0] if isinstance(blocks[0], BlockConfig) else BlockConfig(**blocks[0])
    return int(layers[0]), block


def _metrics(result: Mapping[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for name, value in result.items():
        if isinstance(value, Mapping) and isinstance(value.get("avg"), int | float):
            scalar = float(value["avg"])
            if not math.isfinite(scalar):
                raise ValueError(f"non-finite score metric {name}={scalar}")
            metrics[name] = scalar
    if not metrics:
        raise ValueError("score result has no finite average metrics")
    return metrics


@dataclass(frozen=True)
class ComposedScoreRecord:
    solution_index: int
    layer_idx: int
    block_config: BlockConfig
    metrics: Mapping[str, float]
    provenance: str
    source_result_ids: tuple[str, ...]


@dataclass(frozen=True)
class ComposedScoreTable:
    records: tuple[ComposedScoreRecord, ...]


def composed_table_to_gathered_metrics(
    table: ComposedScoreTable,
    canonical_solutions: Sequence[Mapping[str, Any]],
    teacher_records: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Convert a composed table to the established multi-layer MIP schema."""

    if len(table.records) != len(canonical_solutions):
        raise ValueError(
            "composed score cardinality does not match canonical solutions: "
            f"{len(table.records)} != {len(canonical_solutions)}"
        )
    gathered: dict[str, dict[str, Any]] = {}
    for record in table.records:
        raw_replacement = canonical_solutions[record.solution_index]["single_sequence_replacement"]
        replacement = parse_layer_replacement(dict(raw_replacement))
        layer_idx = int(replacement["parent_layer_indices"][0])
        if layer_idx != record.layer_idx:
            raise ValueError(f"composed record layer mismatch: {record.layer_idx} != {layer_idx}")
        metrics = {name: float(value) for name, value in record.metrics.items()}
        metrics.update({f"one_minus_{name}": 1.0 - value for name, value in metrics.items()})
        gathered[f"replacement_{record.solution_index}"] = {
            "block_config": record.block_config,
            "parent_layer_indices": [layer_idx],
            "metrics": metrics,
            "layer_replacement": replacement,
            "is_teacher": False,
            "score_provenance": {
                "granularity": "subblock",
                "method": record.provenance,
                "source_result_ids": list(record.source_result_ids),
            },
        }
    for layer_idx, teacher_record in enumerate(teacher_records):
        gathered[f"teacher_{layer_idx}"] = teacher_record
    return gathered


def compose_full_block_metrics(
    canonical_solutions: Iterable[Mapping[str, Any]],
    subblock_results: Iterable[Mapping[str, Any]],
    *,
    teacher_blocks: Sequence[BlockConfig],
    teacher_baseline: Mapping[str, float],
    exact_results: Iterable[Mapping[str, Any]] = (),
) -> ComposedScoreTable:
    """Compose every canonical full block from isolated component deltas.

    Exact full-block measurements, when supplied, take precedence over additive
    estimates for the same semantic block identity.
    """

    baseline = {name: float(value) for name, value in teacher_baseline.items()}
    if not baseline or any(not math.isfinite(value) for value in baseline.values()):
        raise ValueError("teacher baseline must contain finite metrics")

    components: dict[tuple[int, SubblockIdentity, str], tuple[str, dict[str, float]]] = {}
    for result in subblock_results:
        puzzle_solution = result["puzzle_solution"]
        marker = puzzle_solution["subblock_replacement"]
        layer_idx, block = _replacement(puzzle_solution)
        identity = (str(marker["kind"]), str(marker["name"]))
        subblock = _subblocks(block).get(identity)
        if subblock is None:
            raise ValueError(f"subblock score marker {identity!r} is absent from its block")
        key = (layer_idx, identity, _subblock_key(subblock))
        result_id = str(result.get("request_id", result.get("i_solution", "unknown")))
        if key in components:
            raise ValueError(f"duplicate subblock score for {key!r}")
        components[key] = (result_id, _metrics(result))

    exact: dict[tuple[int, tuple[tuple[str, str, str], ...]], tuple[str, dict[str, float]]] = {}
    for result in exact_results:
        layer_idx, block = _replacement(result["puzzle_solution"])
        key = (layer_idx, _block_key(block))
        if key in exact:
            raise ValueError(f"duplicate exact block score for layer {layer_idx}")
        exact[key] = (str(result.get("request_id", "exact")), _metrics(result))

    records: list[ComposedScoreRecord] = []
    for solution_index, solution in enumerate(canonical_solutions):
        layer_idx, block = _replacement(solution)
        teacher = teacher_blocks[layer_idx]
        exact_row = exact.get((layer_idx, _block_key(block)))
        if exact_row is not None:
            records.append(
                ComposedScoreRecord(
                    solution_index=solution_index,
                    layer_idx=layer_idx,
                    block_config=block,
                    metrics=exact_row[1],
                    provenance="exact_block",
                    source_result_ids=(exact_row[0],),
                )
            )
            continue

        teacher_by_identity = _subblocks(teacher)
        candidate_by_identity = _subblocks(block)
        if candidate_by_identity.keys() != teacher_by_identity.keys():
            raise ValueError(f"layer {layer_idx} candidate and teacher subblock identities differ")
        changed = [
            identity
            for identity in teacher_by_identity
            if candidate_by_identity[identity] != teacher_by_identity[identity]
        ]
        metrics = dict(baseline)
        source_ids: list[str] = []
        for identity in changed:
            key = (layer_idx, identity, _subblock_key(candidate_by_identity[identity]))
            component = components.get(key)
            if component is None:
                raise ValueError(
                    f"missing subblock score for layer={layer_idx}, identity={identity}, "
                    f"config={candidate_by_identity[identity].to_dict()}"
                )
            result_id, component_metrics = component
            if component_metrics.keys() != baseline.keys():
                raise ValueError(
                    f"metric mismatch for subblock score {result_id}: "
                    f"expected={sorted(baseline)}, present={sorted(component_metrics)}"
                )
            source_ids.append(result_id)
            for name in metrics:
                metrics[name] += component_metrics[name] - baseline[name]
        if any(not math.isfinite(value) for value in metrics.values()):
            raise ValueError(f"non-finite composed metrics for solution {solution_index}")
        records.append(
            ComposedScoreRecord(
                solution_index=solution_index,
                layer_idx=layer_idx,
                block_config=block,
                metrics=metrics,
                provenance="additive_subblock",
                source_result_ids=tuple(source_ids),
            )
        )
    return ComposedScoreTable(records=tuple(records))
