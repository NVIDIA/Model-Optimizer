#!/usr/bin/env python3
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
"""Download datasets for QAD training (OpenScience, Nemotron-v2)."""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from tqdm import tqdm

SEED = 42
TRAIN_RATIO, VALID_RATIO = 0.95, 0.025
_TOKENIZER = None


def init_tokenizer(name: str) -> None:
    """Load HuggingFace tokenizer for chat template."""
    global _TOKENIZER
    if name:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(name, trust_remote_code=True)


def format_text(messages: list[dict], reasoning: str = "") -> str:
    """Format messages to text using tokenizer chat template or simple format."""
    # Add reasoning as thinking block if provided
    if reasoning.strip():
        messages = messages.copy()
        for i, m in enumerate(messages):
            if m.get("role") == "assistant" and i == len(messages) - 1:
                messages[i] = {
                    "role": "assistant",
                    "content": f"<think>\n{reasoning}\n</think>\n{m.get('content', '')}",
                }

    if _TOKENIZER:
        try:
            return _TOKENIZER.apply_chat_template(messages, tokenize=False)
        except Exception:
            pass

    # Fallback
    return "\n\n".join(f"{m['role'].title()}: {m['content']}" for m in messages if m.get("content"))


def split_and_save(examples: list[dict], output_dir: str, prefix: str) -> dict[str, int]:
    """Shuffle, split into train/valid/test, and save as JSONL."""
    random.seed(SEED)
    random.shuffle(examples)

    n = len(examples)
    train_end = int(n * TRAIN_RATIO)
    valid_end = train_end + int(n * VALID_RATIO)

    splits = {
        "train": examples[:train_end],
        "validation": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    counts = {}
    for name, data in splits.items():
        path = os.path.join(output_dir, f"{prefix}_{name}.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in data)
        counts[name] = len(data)
        print(f"  {name}: {len(data):,}")

    return counts


def download_openscience(output_dir: str, use_chat: bool) -> dict[str, Any]:
    """Download nvidia/OpenScience dataset."""
    from datasets import load_dataset

    print("\nDownloading nvidia/OpenScience...")
    ds = load_dataset("nvidia/OpenScience", "OS-Q3-235B-4")
    data = ds["train"] if "train" in ds else ds[next(iter(ds.keys()))]

    print(f"Processing {len(data)} examples...")
    suffix = "_chat" if use_chat else ""
    examples = []
    for ex in tqdm(data.shuffle(seed=SEED), desc="openscience"):
        msgs = [
            {"role": "user", "content": ex.get("input", "")},
            {"role": "assistant", "content": ex.get("output", "")},
        ]
        examples.append({"text": format_text(msgs)})

    counts = split_and_save(examples, output_dir, f"openscience{suffix}")
    return {"dataset": "openscience", "total": len(examples), **counts}


def download_nemotron_v2(
    output_dir: str, splits: list[str], sample_pct: float, suffix: str, include_reasoning: bool
) -> list[dict[str, Any]]:
    """Download nvidia/Nemotron-Post-Training-Dataset-v2 splits."""
    from datasets import load_dataset

    print(f"\nDownloading Nemotron-v2 ({', '.join(splits)}) @ {sample_pct}%...")
    results = []

    for split in splits:
        print(f"\n{split}:")
        ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split, streaming=True)

        examples = []
        for ex in tqdm(ds, desc=split):
            msgs = ex.get("messages", [])
            reasoning = ex.get("reasoning", "") if include_reasoning else ""
            text = format_text(msgs, reasoning)
            if text.strip():
                examples.append({"text": text})

        # Sample if needed
        if sample_pct < 100:
            random.seed(SEED)
            target = int(len(examples) * sample_pct / 100)
            examples = random.sample(examples, min(target, len(examples)))
            print(f"  Sampled to {len(examples):,}")

        if not examples:
            continue

        split_dir = os.path.join(output_dir, split)
        counts = split_and_save(examples, split_dir, f"{split}_{suffix}")
        results.append({"split_name": split, "total": len(examples), **counts})

    return results


def main():
    p = argparse.ArgumentParser(description="Download QAD datasets")
    p.add_argument("--dataset", required=True, choices=["openscience", "nemotron-v2", "all"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tokenizer", help="HuggingFace tokenizer for chat template")
    p.add_argument("--splits", default="stem,math,code,chat", help="Nemotron-v2 splits")
    p.add_argument("--sample-percent", type=float, default=30.0)
    p.add_argument(
        "--include-reasoning", action="store_true", help="Include COT for Thinking models"
    )
    args = p.parse_args()

    if args.tokenizer:
        init_tokenizer(args.tokenizer)

    # Build suffix
    suffix = f"{int(args.sample_percent)}pct"
    if args.include_reasoning:
        suffix += "_cot"
    if args.tokenizer:
        suffix += "_chat"

    results = []

    if args.dataset in ["openscience", "all"]:
        info = download_openscience(
            os.path.join(args.output_dir, "openscience_splits"), args.tokenizer is not None
        )
        results.append(info)

    if args.dataset in ["nemotron-v2", "all"]:
        infos = download_nemotron_v2(
            os.path.join(args.output_dir, "nemotron_v2"),
            [s.strip() for s in args.splits.split(",")],
            args.sample_percent,
            suffix,
            args.include_reasoning,
        )
        results.extend(infos)

    print("\n" + "=" * 50)
    print("Download complete!")
    for r in results:
        name = r.get("dataset") or r.get("split_name")
        print(f"  {name}: {r['total']:,} (train={r['train']:,})")
    print("=" * 50)


if __name__ == "__main__":
    main()
