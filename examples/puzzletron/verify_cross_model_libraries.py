#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Verify exact Cartesian library identities before sparse campaign sampling."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.block_config import BlockConfig
from modelopt.torch.puzzletron.candidates import build_candidate_library


_REQUIRED_RUNTIME_METRICS = frozenset(
    {
        "runtime_ms",
        "prefill_runtime_ms",
        "decode_runtime_ms",
        "decode_runtime_ms_per_token",
        "weight_memory_mib",
        "kv_cache_bytes_per_token",
        "state_cache_bytes_per_sequence",
        "prefill_flops",
        "decode_flops",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _teacher_checkpoint(root: Path) -> Path:
    for candidate in (
        root / "ckpts" / "sorted_teacher",
        root / "ckpts" / "elastic_sorted_teacher",
    ):
        if (candidate / "config.json").is_file():
            return candidate
    raise FileNotFoundError(f"no sorted teacher checkpoint under {root}")


def verify_model_library(
    model_id: str,
    root: str | Path,
    config: dict[str, Any],
    *,
    require_runtime: bool,
) -> dict[str, Any]:
    """Require a library to equal the descriptor-generated Cartesian space."""

    root = Path(root)
    teacher_config = json.loads((_teacher_checkpoint(root) / "config.json").read_text())
    teacher_blocks = [BlockConfig(**block) for block in teacher_config["block_configs"]]
    library_path = root / "replacement_library.json"
    library = json.loads(library_path.read_text())
    entries = library.get("entries", library) if isinstance(library, dict) else library
    hidden_width = library.get("hidden_width") if isinstance(library, dict) else None
    expected = build_candidate_library(
        teacher_blocks,
        search_space=config.get("search_space") or {},
        parent_checkpoint_identity=(
            library.get("parent_checkpoint_identity", "library_parent")
            if isinstance(library, dict)
            else "library_parent"
        ),
        include_self=True,
        include_noops=bool(
            (config.get("build_replacement_library") or {}).get("include_noops", False)
        ),
        hidden_width=hidden_width,
    )

    def entry_key(entry):
        layers = entry.get("parent_layer_indices") or []
        blocks = entry.get("child_block_configs") or []
        if len(layers) != 1 or len(blocks) != 1:
            raise ValueError(f"replacement entry is not one layer/one block: {entry}")
        return int(layers[0]), _canonical(blocks[0])

    actual_keys = [entry_key(entry) for entry in entries]
    duplicates = sorted(key for key, count in Counter(actual_keys).items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate replacement layer/config identities: {duplicates[:5]}")
    expected_keys = {
        (candidate.layer_idx, _canonical(candidate.block_config.to_dict()))
        for candidate in expected
    }
    actual_key_set = set(actual_keys)
    if actual_key_set != expected_keys:
        missing = sorted(expected_keys - actual_key_set)
        extra = sorted(actual_key_set - expected_keys)
        raise ValueError(
            f"library does not match exact Cartesian space: missing={missing[:5]} extra={extra[:5]}"
        )

    teacher_keys = {
        (layer_idx, _canonical(block.to_dict()))
        for layer_idx, block in enumerate(teacher_blocks)
    }
    teacher_entries = len(actual_key_set & teacher_keys)
    if teacher_entries != len(teacher_blocks):
        raise ValueError(
            f"library has {teacher_entries} teacher entries, expected {len(teacher_blocks)}"
        )
    for entry in entries:
        layer_idx = int(entry["parent_layer_indices"][0])
        parent_subblocks = {
            (subblock.kind, subblock.name): subblock
            for subblock in teacher_blocks[layer_idx].subblock_configs
        }
        for block in entry["child_block_configs"]:
            for subblock in block.get("subblock_configs", []):
                parent = parent_subblocks.get((subblock.get("kind"), subblock.get("name")))
                if subblock.get("no_op", False) and not getattr(parent, "no_op", True):
                    raise ValueError(
                        "replacement library unexpectedly disables an active teacher subblock: "
                        f"layer={layer_idx} subblock={subblock.get('name')}"
                    )

    solutions = json.loads((root / "single_sequence_replacement_solutions.json").read_text())
    expected_solution_count = len(entries) - len(teacher_blocks)
    if len(solutions) != expected_solution_count:
        raise ValueError(
            f"solution/library cardinality mismatch: {len(solutions)} != {expected_solution_count}"
        )

    runtime_entries = 0
    stats_filename = (
        (config.get("calc_subblock_stats") or {}).get(
            "subblock_stats_filename", "subblock_stats.json"
        )
    )
    stats_path = root / stats_filename
    if stats_path.is_file():
        stats = json.loads(stats_path.read_text())
        for record_index, record in enumerate(stats):
            if not record.get("args", {}).get("runtime_stats"):
                continue
            for row_index, row in enumerate(record.get("subblocks", [])):
                if row.get("runtime_ms") is None:
                    continue
                runtime_entries += 1
                if require_runtime:
                    missing = sorted(_REQUIRED_RUNTIME_METRICS - row.keys())
                    provenance = row.get("additive_metric_provenance") or {}
                    missing_provenance = sorted(_REQUIRED_RUNTIME_METRICS - provenance.keys())
                    if missing or missing_provenance:
                        raise ValueError(
                            "additive runtime metrics are incomplete in "
                            f"{stats_path} record={record_index} row={row_index}: "
                            f"missing={missing} missing_provenance={missing_provenance}"
                        )
    if require_runtime and runtime_entries == 0:
        raise ValueError(f"runtime measurements are missing or empty in {stats_path}")

    per_layer = Counter(str(layer_idx) for layer_idx, _ in actual_keys)
    return {
        "model_id": model_id,
        "root": str(root.resolve()),
        "hidden_width": hidden_width,
        "num_layers": len(teacher_blocks),
        "entries": len(entries),
        "teacher_entries": teacher_entries,
        "solutions": len(solutions),
        "per_layer": dict(sorted(per_layer.items(), key=lambda item: int(item[0]))),
        "runtime_entries": runtime_entries,
        "status": "complete",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-runtime", action="store_true")
    args = parser.parse_args()
    import yaml

    summary = verify_model_library(
        args.model_id,
        args.root,
        yaml.safe_load(Path(args.config).read_text()),
        require_runtime=args.require_runtime,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
