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

"""
Unified dataset downloader for QAD training.

Supports:
  - nvidia/OpenScience (OS-Q3-235B-4)
  - nvidia/Nemotron-Post-Training-Dataset-v2 (stem, math, code, chat)

Usage:
    # Download OpenScience
    python download_dataset.py --dataset openscience --output-dir /path/to/data --tokenizer Qwen/Qwen3-8B

    # Download Nemotron-v2 (all English splits)
    python download_dataset.py --dataset nemotron-v2 --output-dir /path/to/data --tokenizer Qwen/Qwen3-8B

    # Download specific Nemotron-v2 splits
    python download_dataset.py --dataset nemotron-v2 --splits stem,math --sample-percent 30 ...

NOTE: Nemotron-v2 is GATED. You need:
  1. Request access at: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2
  2. Login with: huggingface-cli login
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any

from tqdm import tqdm

# Constants
TRAIN_RATIO = 0.95
VALID_RATIO = 0.025
TEST_RATIO = 0.025
RANDOM_SEED = 42

DATASET_CONFIGS: dict[str, dict[str, Any]] = {
    "openscience": {
        "hf_name": "nvidia/OpenScience",
        "hf_config": "OS-Q3-235B-4",
        "format": "input_output",  # Has input/output fields
        "gated": False,
    },
    "nemotron-v2": {
        "hf_name": "nvidia/Nemotron-Post-Training-Dataset-v2",
        "hf_config": None,  # Uses split names directly
        "format": "messages",  # Has messages field
        "gated": True,
        "default_splits": ["stem", "math", "code", "chat"],
        "all_splits": [
            "stem",
            "math",
            "code",
            "chat",
            "multilingual_ja",
            "multilingual_de",
            "multilingual_it",
            "multilingual_es",
            "multilingual_fr",
        ],
    },
}

# Global tokenizer
_TOKENIZER = None


def init_tokenizer(tokenizer_name: str) -> None:
    """Initialize tokenizer for chat template formatting."""
    global _TOKENIZER
    if tokenizer_name:
        from transformers import AutoTokenizer

        print(f"Loading tokenizer: {tokenizer_name}")
        _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)


def format_input_output(input_text: str, output_text: str) -> str:
    """Format input/output pair (OpenScience format)."""
    global _TOKENIZER

    if _TOKENIZER is not None:
        messages = [
            {"role": "user", "content": input_text},
            {"role": "assistant", "content": output_text},
        ]
        try:
            return _TOKENIZER.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")

    return f"User: {input_text}\n\nAssistant: {output_text}"


def format_messages(messages: list, reasoning: str | None = None) -> str:
    """Format messages list (Nemotron-v2 format)."""
    global _TOKENIZER

    # Optionally prepend reasoning as thinking block
    if reasoning and reasoning.strip():
        messages_with_cot = []
        for i, msg in enumerate(messages):
            if msg.get("role") == "assistant" and i == len(messages) - 1:
                thinking_content = f"<think>\n{reasoning}\n</think>\n{msg.get('content', '')}"
                messages_with_cot.append({"role": "assistant", "content": thinking_content})
            else:
                messages_with_cot.append(msg)
        messages = messages_with_cot

    if _TOKENIZER is not None:
        try:
            return _TOKENIZER.apply_chat_template(messages, tokenize=False)
        except Exception as e:
            print(f"Warning: Chat template failed: {e}")

    # Fallback: simple format
    text_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            text_parts.append(f"System: {content}")
        elif role == "user":
            text_parts.append(f"User: {content}")
        elif role == "assistant":
            text_parts.append(f"Assistant: {content}")

    return "\n\n".join(text_parts)


def download_openscience(output_dir: str, datablend_dir: str, use_chat: bool) -> dict[str, Any]:
    """Download and split OpenScience dataset."""
    from datasets import load_dataset

    config = DATASET_CONFIGS["openscience"]
    chat_suffix = "_chat" if use_chat else ""

    print(f"\nDownloading {config['hf_name']}...")
    dataset = load_dataset(config["hf_name"], config["hf_config"])

    # Get the data
    if "train" in dataset:
        full_data = dataset["train"]
    else:
        first_split = next(iter(dataset.keys()))
        print(f"Using '{first_split}' split")
        full_data = dataset[first_split]

    print(f"Shuffling {len(full_data)} examples...")
    shuffled_data = full_data.shuffle(seed=RANDOM_SEED)

    # Split
    total = len(shuffled_data)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    splits = {
        "train": shuffled_data.select(range(train_end)),
        "validation": shuffled_data.select(range(train_end, valid_end)),
        "test": shuffled_data.select(range(valid_end, total)),
    }

    print(
        f"Splits: train={len(splits['train'])}, valid={len(splits['validation'])}, test={len(splits['test'])}"
    )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    saved_files = {}

    for split_name, split_data in splits.items():
        output_file = os.path.join(output_dir, f"openscience{chat_suffix}_{split_name}.jsonl")

        with open(output_file, "w", encoding="utf-8") as f:
            for example in tqdm(split_data, desc=split_name):
                text = format_input_output(example.get("input", ""), example.get("output", ""))
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

        saved_files[split_name] = output_file
        print(f"Saved {split_name}: {len(split_data)} examples")

    # Datablend config
    preprocessed_dir = output_dir.replace("openscience_splits", "openscience_splits_preprocessed")
    blend_file = os.path.join(datablend_dir, f"datablend_openscience{chat_suffix}.json")
    blend_config = {
        "train": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_train_text_document"],
        "valid": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_validation_text_document"],
        "test": [1.0, f"{preprocessed_dir}/openscience{chat_suffix}_test_text_document"],
    }
    os.makedirs(datablend_dir, exist_ok=True)
    with open(blend_file, "w") as f:
        json.dump(blend_config, f, indent=2)
    print(f"Created datablend: {blend_file}")

    return {
        "dataset": "openscience",
        "total": total,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "files": saved_files,
        "datablend": blend_file,
    }


def download_nemotron_v2_split(
    split_name: str,
    output_dir: str,
    datablend_dir: str,
    sample_percent: float,
    suffix: str,
    include_reasoning: bool,
) -> dict[str, Any] | None:
    """Download a single Nemotron-v2 split."""
    from datasets import load_dataset, load_dataset_builder

    config = DATASET_CONFIGS["nemotron-v2"]
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)

    # Get split size
    try:
        builder = load_dataset_builder(config["hf_name"], split_name)
        available = builder.info.splits[split_name].num_examples if builder.info.splits else None
        if available:
            target = int(available * sample_percent / 100)
            print(f"\n{split_name}: downloading {target:,} of {available:,} ({sample_percent}%)")
        else:
            target = None
            print(f"\n{split_name}: size unknown, downloading all then sampling")
    except Exception as e:
        if "gated" in str(e).lower() or "access" in str(e).lower():
            print("\nACCESS DENIED - Request access at:")
            print(f"  https://huggingface.co/datasets/{config['hf_name']}")
            print("Then login with: huggingface-cli login")
            raise
        target = None
        print(f"\n{split_name}: could not get size ({e})")

    # Download
    examples = []
    dataset = load_dataset(config["hf_name"], split=split_name, streaming=True)

    count = 0
    for example in tqdm(dataset, desc=split_name, total=target):
        if target is not None and count >= target:
            break

        messages = example.get("messages", [])
        reasoning = example.get("reasoning", "") if include_reasoning else ""
        text = format_messages(messages, reasoning)

        if text.strip():
            examples.append({"text": text})
            count += 1

    print(f"Collected {count:,} examples")

    # If downloaded all, sample
    if target is None and sample_percent < 100:
        random.seed(RANDOM_SEED)
        target_count = int(len(examples) * sample_percent / 100)
        examples = random.sample(examples, target_count)
        print(f"Sampled to {len(examples):,}")

    if not examples:
        print(f"Warning: No examples from {split_name}")
        return None

    # Shuffle and split
    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    total = len(examples)
    train_end = int(total * TRAIN_RATIO)
    valid_end = train_end + int(total * VALID_RATIO)

    splits = {
        "train": examples[:train_end],
        "validation": examples[train_end:valid_end],
        "test": examples[valid_end:],
    }

    # Save
    saved_files = {}
    for data_split, data in splits.items():
        output_file = os.path.join(split_dir, f"{split_name}_{suffix}_{data_split}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(ex, ensure_ascii=False) + "\n" for ex in data)
        saved_files[data_split] = output_file
        print(f"  {data_split}: {len(data):,} examples")

    # Datablend config
    preprocessed_dir = output_dir.replace("nemotron_v2", "nemotron_v2_preprocessed")
    split_preprocessed_dir = os.path.join(preprocessed_dir, split_name)

    blend_file = os.path.join(datablend_dir, f"datablend_nemotron_v2_{split_name}_{suffix}.json")
    blend_config = {
        "train": [1.0, f"{split_preprocessed_dir}/{split_name}_{suffix}_train_text_document"],
        "valid": [1.0, f"{split_preprocessed_dir}/{split_name}_{suffix}_validation_text_document"],
        "test": [1.0, f"{split_preprocessed_dir}/{split_name}_{suffix}_test_text_document"],
    }
    os.makedirs(datablend_dir, exist_ok=True)
    with open(blend_file, "w") as f:
        json.dump(blend_config, f, indent=2)
    print(f"  Datablend: {blend_file}")

    return {
        "split_name": split_name,
        "total": total,
        "train": len(splits["train"]),
        "validation": len(splits["validation"]),
        "test": len(splits["test"]),
        "files": saved_files,
        "datablend": blend_file,
    }


def download_nemotron_v2(
    output_dir: str,
    datablend_dir: str,
    splits: list[str],
    sample_percent: float,
    suffix: str,
    include_reasoning: bool,
) -> list[dict[str, Any]]:
    """Download Nemotron-v2 dataset (multiple splits)."""
    print(f"\nDownloading Nemotron-v2: {splits}")
    print(f"Sample: {sample_percent}%, Reasoning: {include_reasoning}")
    print("=" * 60)
    print("NOTE: This dataset is GATED. You need HuggingFace access.")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(datablend_dir, exist_ok=True)

    all_infos = []
    for split_name in splits:
        info = download_nemotron_v2_split(
            split_name=split_name,
            output_dir=output_dir,
            datablend_dir=datablend_dir,
            sample_percent=sample_percent,
            suffix=suffix,
            include_reasoning=include_reasoning,
        )
        if info:
            all_infos.append(info)

    # Create combined datablend
    if all_infos:
        preprocessed_dir = output_dir.replace("nemotron_v2", "nemotron_v2_preprocessed")
        total_train = sum(info["train"] for info in all_infos)

        train_blend = []
        valid_blend = []
        test_blend = []

        for info in all_infos:
            sn = info["split_name"]
            weight = info["train"] / total_train if total_train > 0 else 1.0 / len(all_infos)
            split_path = os.path.join(preprocessed_dir, sn)

            train_blend.extend([weight, f"{split_path}/{sn}_{suffix}_train_text_document"])
            valid_blend.extend([weight, f"{split_path}/{sn}_{suffix}_validation_text_document"])
            test_blend.extend([weight, f"{split_path}/{sn}_{suffix}_test_text_document"])

        combined_file = os.path.join(datablend_dir, f"datablend_nemotron_v2_combined_{suffix}.json")
        with open(combined_file, "w") as f:
            json.dump({"train": train_blend, "valid": valid_blend, "test": test_blend}, f, indent=2)
        print(f"\nCombined datablend: {combined_file}")

    return all_infos


def main():
    parser = argparse.ArgumentParser(description="Download datasets for QAD training")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["openscience", "nemotron-v2", "all"],
        help="Dataset to download",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--datablend-dir",
        type=str,
        default=None,
        help="Datablend config directory (default: output-dir)",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None, help="HuggingFace tokenizer for chat template"
    )

    # Nemotron-v2 specific
    parser.add_argument(
        "--splits",
        type=str,
        default="stem,math,code,chat",
        help="Nemotron-v2 splits to download (comma-separated)",
    )
    parser.add_argument(
        "--sample-percent",
        type=float,
        default=30.0,
        help="Percentage of data to sample (default: 30)",
    )
    parser.add_argument(
        "--include-reasoning",
        action="store_true",
        help="Include chain-of-thought reasoning (for Thinking models)",
    )

    args = parser.parse_args()

    output_dir = args.output_dir
    datablend_dir = args.datablend_dir or output_dir
    use_chat = args.tokenizer is not None

    if args.tokenizer:
        init_tokenizer(args.tokenizer)

    # Build suffix
    pct_str = f"{int(args.sample_percent)}pct"
    cot_str = "_cot" if args.include_reasoning else ""
    chat_str = "_chat" if use_chat else ""
    suffix = f"{pct_str}{cot_str}{chat_str}"

    results = []

    if args.dataset in ["openscience", "all"]:
        os_dir = os.path.join(output_dir, "openscience_splits")
        info = download_openscience(os_dir, datablend_dir, use_chat)
        results.append(info)

    if args.dataset in ["nemotron-v2", "all"]:
        nv2_dir = os.path.join(output_dir, "nemotron_v2")
        splits = [s.strip() for s in args.splits.split(",")]
        infos = download_nemotron_v2(
            output_dir=nv2_dir,
            datablend_dir=datablend_dir,
            splits=splits,
            sample_percent=args.sample_percent,
            suffix=suffix,
            include_reasoning=args.include_reasoning,
        )
        results.extend(infos)

    # Summary
    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)
    for r in results:
        name = r.get("dataset") or r.get("split_name")
        print(f"  {name}: {r['total']:,} samples (train={r['train']:,})")
    print("=" * 60)


if __name__ == "__main__":
    main()
