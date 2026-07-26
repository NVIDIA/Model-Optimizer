#!/usr/bin/env python3
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

"""Materialize a bounded first-class Puzzletron dataset for offline cluster reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from modelopt.torch.puzzletron.dataset.acquisition import (
    VLM_HEADER_SUBSETS,
    TextAcquisitionSpec,
    VlmAcquisitionSpec,
    materialize_nemotron_vlm_dataset,
    materialize_puzzle_kd_dataset,
)


def _subset_row(value: str) -> tuple[str, int]:
    name, separator, raw_rows = value.partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("subset rows must use NAME=ROWS")
    try:
        rows = int(raw_rows)
    except ValueError as error:
        raise argparse.ArgumentTypeError("subset row count must be an integer") from error
    if rows <= 0:
        raise argparse.ArgumentTypeError("subset row count must be positive")
    return name, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="adapter", required=True)

    text = subparsers.add_parser("puzzle_kd_v2", help="Materialize Puzzle-KD text data.")
    text.add_argument("--output", type=Path, required=True)
    text.add_argument("--train-samples", type=int, default=8192)
    text.add_argument("--validation-samples", type=int, default=1024)
    text.add_argument("--seed", type=int, default=408)
    text.add_argument("--revision")

    vlm = subparsers.add_parser(
        "nemotron_vlm_v2",
        help="Materialize image-text rows from bounded Nemotron-VLM media shards.",
    )
    vlm.add_argument("--output", type=Path, required=True)
    vlm.add_argument("--subsets", nargs="+", default=list(VLM_HEADER_SUBSETS))
    vlm.add_argument(
        "--subset-rows",
        nargs="+",
        type=_subset_row,
        help="Ordered source row counts as NAME=ROWS; enables proportional sampling.",
    )
    vlm.add_argument("--num-samples", type=int, default=512)
    vlm.add_argument("--seed", type=int, default=42)
    vlm.add_argument("--max-shards-per-subset", type=int, default=1)
    vlm.add_argument("--revision")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.adapter == "puzzle_kd_v2":
        return materialize_puzzle_kd_dataset(
            TextAcquisitionSpec(
                output_dir=args.output,
                train_samples=args.train_samples,
                validation_samples=args.validation_samples,
                seed=args.seed,
                revision=args.revision,
            )
        )
    if args.adapter == "nemotron_vlm_v2":
        return materialize_nemotron_vlm_dataset(
            VlmAcquisitionSpec(
                output_dir=args.output,
                subsets=tuple(args.subsets),
                subset_rows=tuple(args.subset_rows or ()),
                num_samples=args.num_samples,
                seed=args.seed,
                max_shards_per_subset=args.max_shards_per_subset,
                revision=args.revision,
            )
        )
    raise ValueError(f"unsupported dataset adapter {args.adapter!r}")


def main() -> None:
    args = build_parser().parse_args()
    manifest = run(args)
    print(
        json.dumps(
            {
                "status": "ready",
                "adapter": args.adapter,
                "output": str(args.output.resolve()),
                "manifest": manifest,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
