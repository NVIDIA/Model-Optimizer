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
Prepare a dataset from nvidia/Nemotron-Post-Training-Dataset-v2 for two purposes:

  generate (default)
    Input conversations for synthetic data generation.  The last assistant turn is
    stripped so a target model can generate a fresh response.  An augmented copy of
    the major splits is appended (language-redirect and style variants) to diversify
    the prompts seen during generation.  Multilingual splits are included but not
    augmented (they are already non-English).

  train
    Full conversations for direct SFT training.  All turns are kept and normalized
    to clean OpenAI message format (role + content only).  No augmentation is applied.

Output format
-------------
Every output row is a JSON object with a single ``messages`` key whose value is a list
of ``{"role": ..., "content": ...}`` dicts — standard OpenAI chat format.

Expected output size (approximate, default settings)
----------------------------------------------------
Split sizes (Nemotron-Post-Training-Dataset-v2):
  stem 355K  |  chat 628K  |  math 239K  |  code 175K  →  major total ~1.40M
  multilingual ×5 splits, each capped at 100K          →  500K

  generate mode:  major ~1.40M  +  augmented ~1.40M  +  multilingual 500K  ≈  3.3M rows
  train mode:     major ~1.40M  +  multilingual 500K                        ≈  1.9M rows

Usage
-----
    # Synthetic data generation (default):
    python make_nemotron_ptv2_dataset.py --output-dir /tmp/ptv2_gen

    # Direct SFT training mix:
    python make_nemotron_ptv2_dataset.py --mode train --output-dir /tmp/ptv2_train

    # Custom augmentation config:
    python make_nemotron_ptv2_dataset.py --augmentations-config my_augs.yaml \\
        --output-dir /tmp/ptv2_gen
"""

import argparse
import logging
import os
from pathlib import Path

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

MAJOR_SPLITS = ["stem", "chat", "math", "code"]

MULTILINGUAL_SPLITS = [
    "multilingual_ja",
    "multilingual_de",
    "multilingual_it",
    "multilingual_es",
    "multilingual_fr",
]

_DEFAULT_AUGMENTATIONS_CONFIG = Path(__file__).parent / "augmentations.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a dataset from Nemotron-Post-Training-Dataset-v2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="nvidia/Nemotron-Post-Training-Dataset-v2",
        help="HuggingFace Hub repo ID or local path to the dataset.",
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
        "--output-dir",
        default="/tmp/ptv2_gen",
        help="Directory where output JSONL files will be written.",
    )
    parser.add_argument(
        "--multilingual-cap",
        type=int,
        default=100_000,
        help="Maximum number of rows to take from each multilingual split.",
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
        "--augmentations-config",
        type=Path,
        default=_DEFAULT_AUGMENTATIONS_CONFIG,
        help="Path to a YAML file listing augmentation specs (generate mode only).",
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


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Mode: %s", args.mode)

    # ------------------------------------------------------------------
    # Augmentation (generate mode only)
    # ------------------------------------------------------------------
    aug_fn = None
    if args.mode == "generate" and not args.no_augmentation:
        aug_specs = load_augmentations(args.augmentations_config)
        aug_fn = make_augment_fn(aug_specs)

    combined = None

    # ------------------------------------------------------------------
    # Major splits
    # ------------------------------------------------------------------
    logger.info("Loading major splits: %s", MAJOR_SPLITS)
    for split in MAJOR_SPLITS:
        logger.info("  loading split '%s'", split)
        ds = load_dataset(args.dataset, split=split)
        if args.mode == "generate":
            ds = ds.filter(lambda ex: not has_tool_turns(ex), num_proc=args.num_proc)
            ds = ds.map(strip_assistant_turns, with_indices=True, num_proc=args.num_proc)
            ds = ds.filter(lambda ex: len(ex["messages"]) > 0, num_proc=args.num_proc)
        else:
            ds = ds.map(normalize_messages, with_indices=True, num_proc=args.num_proc)
            # Train mode: drop prompt-only rows (no assistant turn = nothing to train on).
            ds = ds.filter(
                lambda ex: any(m["role"] == "assistant" for m in ex["messages"]),
                num_proc=args.num_proc,
            )
        logger.info("  %d rows", len(ds))
        combined = ds if combined is None else concatenate_datasets([combined, ds])

    assert combined is not None
    logger.info("Major splits total: %d rows", len(combined))

    # ------------------------------------------------------------------
    # Augmented copy (generate mode only, applied to major splits)
    # ------------------------------------------------------------------
    if aug_fn is not None:
        logger.info("Augmenting with %d variant(s)...", len(aug_specs))
        augmented = combined.map(aug_fn, with_indices=True, num_proc=args.num_proc)
        combined = concatenate_datasets([combined, augmented])
        logger.info("After major + augmented: %d rows", len(combined))

    # ------------------------------------------------------------------
    # Multilingual splits (not augmented — already non-English)
    # ------------------------------------------------------------------
    logger.info("Loading multilingual splits: %s", MULTILINGUAL_SPLITS)
    for split in MULTILINGUAL_SPLITS:
        logger.info("  loading split '%s'", split)
        ds = load_dataset(args.dataset, split=split)
        if len(ds) > args.multilingual_cap:
            ds = ds.select(range(args.multilingual_cap))
        if args.mode == "generate":
            ds = ds.filter(lambda ex: not has_tool_turns(ex), num_proc=args.num_proc)
            ds = ds.map(strip_assistant_turns, with_indices=True, num_proc=args.num_proc)
            ds = ds.filter(lambda ex: len(ex["messages"]) > 0, num_proc=args.num_proc)
        else:
            ds = ds.map(normalize_messages, with_indices=True, num_proc=args.num_proc)
            ds = ds.filter(
                lambda ex: any(m["role"] == "assistant" for m in ex["messages"]),
                num_proc=args.num_proc,
            )
        logger.info("  %d rows", len(ds))
        combined = concatenate_datasets([combined, ds])

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
