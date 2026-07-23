#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare a replace-one-subblock scoring view from a canonical block library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.block_config import maybe_cast_block_configs
from modelopt.torch.puzzletron.replacement_library.subblock_scoring import (
    build_subblock_replacement_payload,
)
from modelopt.torch.puzzletron.tools.checkpoint_utils_hf import load_model_config

# Backward-compatible import for callers of the example helper.
prepare_subblock_replacement_payload = build_subblock_replacement_payload


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle-dir", required=True)
    parser.add_argument("--replacement-library")
    parser.add_argument("--teacher-dir")
    parser.add_argument("--solutions-output")
    parser.add_argument("--manifest-output")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    puzzle_dir = Path(args.puzzle_dir)
    library_path = Path(args.replacement_library or puzzle_dir / "replacement_library.json")
    teacher_dir = Path(args.teacher_dir or puzzle_dir / "ckpts" / "sorted_teacher")
    solutions_path = Path(
        args.solutions_output or puzzle_dir / "single_subblock_replacement_solutions.json"
    )
    manifest_path = Path(
        args.manifest_output or puzzle_dir / "subblock_replacement_manifest.json"
    )
    replacement_library = json.loads(library_path.read_text())
    model_config = load_model_config(teacher_dir, trust_remote_code=args.trust_remote_code)
    teacher_blocks = list(maybe_cast_block_configs(model_config.block_configs))
    manifest, solutions = prepare_subblock_replacement_payload(
        replacement_library,
        teacher_blocks,
    )
    manifest.update(
        {
            "replacement_library": str(library_path.resolve()),
            "teacher_dir": str(teacher_dir.resolve()),
            "solutions": str(solutions_path.resolve()),
        }
    )
    _write_json(solutions_path, solutions)
    _write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
