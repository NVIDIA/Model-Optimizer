# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Prepare a dataset from the Nemotron Post-Training V3 collection for two purposes:
https://huggingface.co/collections/nvidia/nemotron-post-training-v3

  generate (default)
    Input conversations for synthetic data generation.  The last assistant turn is
    stripped so a target model can generate a fresh response.  Datasets marked
    ``augment: true`` in the config get an augmented copy appended (language-redirect
    and style variants).  Multilingual datasets (``augment: false``) are included
    as-is without augmentation.

  train
    Full conversations for direct SFT training.  All turns are kept and normalized
    to clean OpenAI message format (role + content only).  No augmentation is applied.
    The ``augment`` flag in the dataset config is ignored.

Output format
-------------
Every output row is a JSON object with a single ``messages`` key whose value is a list
of ``{"role": ..., "content": ...}`` dicts — standard OpenAI chat format.

Dataset config
--------------
``nemotron_ptv3_datasets.yaml`` lists every dataset in the mix with:
  repo_id        — HuggingFace repo ID or local path
  splits         — list of splits to load
  cap_per_split  — max rows per split (null = no cap)
  augment        — false for multilingual splits (generate mode only)

Edit the YAML to add, remove, or re-weight datasets without touching this script.

Expected output size (approximate, default settings from nemotron_ptv3_datasets.yaml)
------------------------------------------------------------------------------------
Datasets without tool turns (included in both modes):
  Math-v2 200K  |  SFT-Math-v3 200K  |  Math-Proofs 50K           →   450K
  Comp-Prog-v1 capped 300K  |  SFT-Comp-Prog-v2 capped 200K       →   500K
  Science 226K  |  IF-Chat-v1 288K  |  IF-Chat-v2 ~2K             →  ~516K
  Safety 45K  |  Finance capped 100K                               →   145K
  Multilingual ×18 splits, each capped at 10K                      →   180K
  Subtotal (base)                                                   → ~1.79M

Datasets with tool turns (generate mode: excluded; train mode: included):
  SWE-v1 51K  |  SFT-SWE-v2 256K  |  OpenCode ~459K               →  ~766K
  Agentic-v1 335K  |  SFT-Agentic-v2 ~992K                        → ~1.33M
  Tool-turn subtotal                                                → ~2.09M

  generate mode:  base ~1.61M (excl. multilingual)  ×2 aug  +  multilingual 180K  ≈  3.4M rows
  train mode:     base ~1.61M  +  tool datasets ~2.09M  +  multilingual 180K       ≈  3.9M rows

Usage
-----
    # Synthetic data generation (default):
    python make_nemotron_ptv3_dataset.py --output-dir /tmp/ptv3_gen

    # Direct SFT training mix:
    python make_nemotron_ptv3_dataset.py --mode train --output-dir /tmp/ptv3_train

    # Custom dataset list:
    python make_nemotron_ptv3_dataset.py --datasets-config my_datasets.yaml \\
        --output-dir /tmp/ptv3_gen
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from conversation_utils import (
    has_tool_turns,
    load_augmentations,
    make_augment_fn,
    normalize_messages,
    strip_assistant_turns,
)
from datasets import concatenate_datasets, load_dataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

_DEFAULT_DATASETS_CONFIG = Path(__file__).parent / "nemotron_ptv3_datasets.yaml"
_DEFAULT_AUGMENTATIONS_CONFIG = Path(__file__).parent / "augmentations.yaml"


# -------------------------------------------------------------------
# Dataset spec
# -------------------------------------------------------------------


@dataclass
class DatasetSpec:
    repo_id: str
    splits: list[str]
    cap_per_split: int | None = None
    augment: bool = True


def load_dataset_specs(config_path: Path) -> list[DatasetSpec]:
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [DatasetSpec(**entry) for entry in data.get("datasets", [])]


# -------------------------------------------------------------------
# Loading
# -------------------------------------------------------------------


def load_split(repo_id: str, split: str, cap: int | None, num_proc: int, mode: str):
    """Load one split, apply cap, normalize for the requested mode."""
    logger.info("    split '%s' ...", split)
    ds = load_dataset(repo_id, split=split)
    if cap is not None and len(ds) > cap:
        ds = ds.select(range(cap))
        logger.info("      capped to %d rows", cap)
    if mode == "generate":
        ds = ds.filter(lambda ex: not has_tool_turns(ex), num_proc=num_proc)
        ds = ds.map(strip_assistant_turns, with_indices=True, num_proc=num_proc)
        ds = ds.filter(lambda ex: len(ex["messages"]) > 0, num_proc=num_proc)
    else:
        ds = ds.map(normalize_messages, with_indices=True, num_proc=num_proc)
        # Train mode: drop prompt-only rows (no assistant turn = nothing to train on).
        ds = ds.filter(
            lambda ex: any(m["role"] == "assistant" for m in ex["messages"]),
            num_proc=num_proc,
        )
    logger.info("      %d rows", len(ds))
    return ds


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset from the Nemotron Post-Training V3 collection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["generate", "train"],
        default="generate",
        help=(
            "generate: strip last assistant turn and apply augmentations — "
            "produces multi-turn input conversations for synthetic data generation. "
            "train: keep all turns in clean OpenAI format — "
            "produces a dataset ready for direct SFT training."
        ),
    )
    parser.add_argument(
        "--datasets-config",
        type=Path,
        default=_DEFAULT_DATASETS_CONFIG,
        help="YAML file listing datasets, splits, and caps.",
    )
    parser.add_argument(
        "--augmentations-config",
        type=Path,
        default=_DEFAULT_AUGMENTATIONS_CONFIG,
        help="YAML file listing augmentation specs (generate mode only).",
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/ptv3_train",
        help="Directory where output JSONL files will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel workers for dataset.map().",
    )
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Skip augmentation even in generate mode.",
    )
    parser.add_argument(
        "--no-subsets",
        action="store_true",
        help="Skip writing the 1K / 10K / 100K prefix subsets.",
    )
    return parser.parse_args()


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Mode: %s", args.mode)

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    dataset_specs = load_dataset_specs(args.datasets_config)
    logger.info("Loaded %d dataset spec(s) from %s.", len(dataset_specs), args.datasets_config)

    aug_fn = None
    if args.mode == "generate" and not args.no_augmentation:
        aug_specs = load_augmentations(args.augmentations_config)
        aug_fn = make_augment_fn(aug_specs)

    # ------------------------------------------------------------------
    # Load all datasets
    # In generate mode, partition into augmentable vs. non-augmentable.
    # In train mode, all datasets are treated identically.
    # ------------------------------------------------------------------
    augmentable_parts: list[Any] = []
    non_augmentable_parts: list[Any] = []

    for spec in dataset_specs:
        logger.info("Loading %s  (augment=%s)", spec.repo_id, spec.augment)
        for split in spec.splits:
            ds = load_split(spec.repo_id, split, spec.cap_per_split, args.num_proc, args.mode)
            if args.mode == "generate" and not spec.augment:
                non_augmentable_parts.append(ds)
            else:
                augmentable_parts.append(ds)

    augmentable = concatenate_datasets(augmentable_parts) if augmentable_parts else None
    non_augmentable = concatenate_datasets(non_augmentable_parts) if non_augmentable_parts else None
    if augmentable is not None:
        logger.info("Augmentable rows: %d", len(augmentable))
    if non_augmentable is not None:
        logger.info("Non-augmentable (multilingual) rows: %d", len(non_augmentable))

    # ------------------------------------------------------------------
    # Augmentation (generate mode only)
    # ------------------------------------------------------------------
    parts_to_combine: list[Any] = []
    if augmentable is not None:
        parts_to_combine.append(augmentable)

    if aug_fn is not None and augmentable is not None:
        logger.info("Augmenting %d rows with %d variant(s)...", len(augmentable), len(aug_specs))
        augmented = augmentable.map(aug_fn, with_indices=True, num_proc=args.num_proc)
        logger.info("Augmented dataset: %d rows", len(augmented))
        parts_to_combine.append(augmented)

    if non_augmentable is not None:
        parts_to_combine.append(non_augmentable)

    if not parts_to_combine:
        raise ValueError("No data to combine — all rows were filtered out.")

    combined = concatenate_datasets(parts_to_combine)
    logger.info("Combined (pre-shuffle): %d rows", len(combined))

    # ------------------------------------------------------------------
    # Shuffle and save
    # ------------------------------------------------------------------
    combined = combined.shuffle(seed=args.seed)
    logger.info("Shuffled with seed=%d", args.seed)

    full_path = output_dir / "default.jsonl"
    logger.info("Writing %d rows to %s", len(combined), full_path)
    combined.to_json(str(full_path), num_proc=args.num_proc)

    if not args.no_subsets:
        for n, name in [(1_000, "1K"), (10_000, "10K"), (100_000, "100K")]:
            if len(combined) < n:
                logger.warning("Fewer than %d rows — skipping %s subset.", n, name)
                continue
            subset_path = output_dir / f"sample-{name}.jsonl"
            logger.info("Writing %s subset to %s", name, subset_path)
            combined.select(range(n)).to_json(str(subset_path), num_proc=args.num_proc)

    logger.info("Done. Output files are in %s", output_dir)


if __name__ == "__main__":
    main()
