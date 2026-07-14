#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a capped deterministic replacement-scoring view for one width."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

from modelopt.torch.puzzletron.anymodel.registry import resolve_descriptor_from_pretrained
from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
from modelopt.torch.puzzletron.candidates import Candidate, build_candidate_library
from modelopt.torch.puzzletron.distributed_eval.config import checkpoint_identity
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.sampling.sparse import (
    SparseSamplingPolicy,
    sample_replacement_candidates,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils import load_model_config


def _canonical_block_key(block: dict[str, Any]) -> str:
    """Key a block by semantics, independent of serialized subblock order."""

    def without_optional_nulls(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_optional_nulls(item)
                for key, item in value.items()
                if item is not None
            }
        if isinstance(value, list):
            return [without_optional_nulls(item) for item in value]
        return value

    normalized = without_optional_nulls(deepcopy(block))
    subblocks = normalized.get("subblock_configs")
    if isinstance(subblocks, list):
        normalized["subblock_configs"] = sorted(
            subblocks,
            key=lambda item: (
                str(item.get("kind", "")),
                str(item.get("name", "")),
                json.dumps(item, sort_keys=True, separators=(",", ":")),
            ),
        )
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _replacement_key(solution: dict[str, Any]) -> tuple[int, str]:
    replacement = solution["single_sequence_replacement"]
    layers = replacement["parent_layer_indices"]
    blocks = replacement["child_block_configs"]
    if len(layers) != 1 or len(blocks) != 1:
        raise ValueError("sparse replacement scoring requires one layer and one block")
    return int(layers[0]), _canonical_block_key(blocks[0])


def prepare_sparse_replacement_payload(
    candidates: Iterable[Candidate],
    solutions: Iterable[dict[str, Any]],
    *,
    hidden_width: int,
    policy: SparseSamplingPolicy | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the sample manifest and annotated solution subset for one width."""

    candidates = list(candidates)
    if any(candidate.hidden_width != hidden_width for candidate in candidates):
        raise ValueError(f"candidate library mixes widths; expected only {hidden_width}")
    manifest = sample_replacement_candidates(candidates, policy=policy)
    by_key: dict[tuple[int, str], dict[str, Any]] = {}
    for solution in solutions:
        key = _replacement_key(solution)
        if key in by_key:
            raise ValueError(f"duplicate canonical replacement solution: {key}")
        by_key[key] = solution

    selected: list[dict[str, Any]] = []
    for row in manifest.selected:
        key = (
            row.layer_idx,
            _canonical_block_key(row.block_config),
        )
        if key not in by_key:
            raise ValueError(
                f"selected candidate {row.candidate_id} has no canonical replacement solution"
            )
        solution = deepcopy(by_key[key])
        solution.update(
            {
                "hidden_width": hidden_width,
                "sparse_sample_id": row.sample_id,
                "candidate_id": row.candidate_id,
            }
        )
        selected.append(solution)

    payload = manifest.to_dict()
    payload["hidden_width"] = hidden_width
    if len(selected) > manifest.policy.replacement_cap:
        raise ValueError(
            f"replacement cap violated: {len(selected)} > {manifest.policy.replacement_cap}"
        )
    return payload, selected


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenario-dir", required=True)
    parser.add_argument("--replacement-cap", type=int, default=50)
    parser.add_argument("--max-pairwise-per-family", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = pipeline_config_from_path(args.config)
    scenario_dir = Path(args.scenario_dir)
    teacher = scenario_dir / "ckpts" / "sorted_teacher"
    scenario = json.loads((scenario_dir / "scenario_manifest.json").read_text())
    width = int(scenario["hidden_width"])
    model_cfg = cfg.get("model") or {}
    descriptor = resolve_descriptor_from_pretrained(
        str(teacher),
        trust_remote_code=bool(model_cfg.get("trust_remote_code", False)),
        descriptor_override=model_cfg.get("descriptor_override"),
    ).descriptor
    model_config = load_model_config(
        teacher,
        trust_remote_code=descriptor.requires_trust_remote_code(),
    )
    candidates = build_candidate_library(
        list(maybe_cast_block_configs(model_config.block_configs)),
        search_space=cfg.get("search_space") or {},
        parent_checkpoint_identity=checkpoint_identity(teacher)["fingerprint"],
        include_self=True,
        include_noops=bool(
            (cfg.get("build_replacement_library") or {}).get(
                "include_noops", False
            )
        ),
        hidden_width=width,
    )
    canonical_solutions = json.loads(
        (scenario_dir / "single_sequence_replacement_solutions.json").read_text()
    )
    manifest, selected = prepare_sparse_replacement_payload(
        candidates,
        canonical_solutions,
        hidden_width=width,
        policy=SparseSamplingPolicy(
            max_pairwise_per_family=args.max_pairwise_per_family,
            replacement_cap=args.replacement_cap,
            seed=args.seed,
        ),
    )
    manifest.update(
        {
            "model_id": Path(cfg["puzzle_dir"]).name,
            "scenario_dir": str(scenario_dir.resolve()),
            "parent_checkpoint_identity": checkpoint_identity(teacher),
            "candidate_count": len(candidates),
            "canonical_solution_count": len(canonical_solutions),
        }
    )
    manifest_path = scenario_dir / "sparse_replacement_samples.json"
    solutions_path = scenario_dir / "sparse_replacement_solutions.json"
    _write_json(manifest_path, manifest)
    _write_json(solutions_path, selected)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "solutions": str(solutions_path),
                "identity": manifest["identity"],
                "eligible": len(manifest["eligible"]),
                "selected": len(selected),
                "excluded": len(manifest["excluded"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
