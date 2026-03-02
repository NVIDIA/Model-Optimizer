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

"""Add Nemotron-Post-Training-Dataset-v2 chat conversations to a conversation dataset.

Dataset: https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2

Note: This dataset requires agreeing to the terms of use on Hugging Face.
      Make sure you are logged in with `huggingface-cli login` before running this script.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import hf_hub_url, list_repo_files
from tqdm import tqdm
from utils import (
    dataset_splits_explanation,
    id_for_conversation,
    update_dataset_file_with_conversations,
)

NEMOTRON_DATASET_ID = "nvidia/Nemotron-Post-Training-Dataset-v2"
NEMOTRON_CHAT_SPLIT = "chat"


def get_split_parquet_urls(dataset_id: str, split_name: str) -> list[str]:
    """Return the HuggingFace Hub download URLs for parquet shards of *split_name* only.

    Files in this dataset follow the flat naming pattern:
    ``data/{split_name}-XXXXX-of-XXXXX.parquet`` (no per-split subdirectory).
    We resolve them to full Hub URLs so they can be passed to
    ``load_dataset("parquet", data_files=…)`` which skips the repo's split
    metadata validation and downloads only the requested shards.

    Raises:
        ValueError: If no parquet files are found for the requested split.
    """
    all_files = list(list_repo_files(dataset_id, repo_type="dataset"))
    split_files = [
        f
        for f in all_files
        if f.endswith(".parquet") and Path(f).name.startswith(f"{split_name}-")
    ]
    if not split_files:
        err_msg = (
            f"No parquet files found for split '{split_name}' in {dataset_id}. "
            f"Available files: {[f for f in all_files if f.endswith('.parquet')]}"
        )
        raise ValueError(err_msg)
    return [hf_hub_url(dataset_id, f, repo_type="dataset") for f in split_files]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load Nemotron-Post-Training-Dataset-v2 chat conversations."
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help=(
            "Maximum number of samples to load from the dataset. "
            f"The chat split contains ~627,720 samples. "
            "If not provided, all samples are loaded."
        ),
    )

    parser.add_argument(
        "--mapping-file",
        type=Path,
        default=None,
        help=(
            "Path to a CSV file containing one 0-based dataset row index per line, "
            "e.g.:\n  429801\n  271023\n  432477\n"
            "Rows are loaded in the order they appear in the file. "
            "When provided, the dataset is downloaded (not streamed) to allow random access "
            "and --max-samples is ignored."
        ),
    )

    parser.add_argument(
        "--output-split-name",
        type=str,
        default="nemotron-chat",
        help=dataset_splits_explanation("nemotron-chat"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save the conversations file(s) into. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


def parse_nemotron_conversation(raw_conversations: list) -> list[dict] | None:
    """Parse a Nemotron conversation into a list of messages with standardized roles.

    Args:
        raw_conversations: List of message dicts from the Nemotron dataset.

    Returns:
        List of parsed message dicts with 'role' and 'content' keys, or None if the
        conversation should be skipped.
    """
    msgs = []
    for msg in raw_conversations:
        # Resolve role field (datasets may use "from" or "role")
        if "from" in msg:
            role = msg["from"].lower()
        elif "role" in msg:
            role = msg["role"].lower()
        else:
            continue

        # Normalize role names to standard values
        if role in ("human", "user"):
            role = "user"
        elif role in ("gpt", "assistant"):
            role = "assistant"
        elif role == "system":
            # Skip system messages; they are metadata not part of the conversation turns
            continue
        else:
            # Skip unrecognized roles rather than failing, as the dataset may evolve
            continue

        # Resolve content field
        if "value" in msg:
            content = msg["value"]
        elif "content" in msg:
            content = msg["content"]
        elif "text" in msg:
            content = msg["text"]
        else:
            continue

        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})

    return msgs if msgs else None


async def main(args: argparse.Namespace) -> None:
    # Resolve Hub download URLs for chat parquet shards only.
    # Using load_dataset("parquet", data_files=urls) bypasses the repo's split
    # metadata validation, so only the chat shards are ever downloaded.
    print(f"Resolving chat parquet URLs in {NEMOTRON_DATASET_ID}...")
    chat_parquet_urls = get_split_parquet_urls(NEMOTRON_DATASET_ID, NEMOTRON_CHAT_SPLIT)
    print(f"Found {len(chat_parquet_urls)} parquet shard(s) for the '{NEMOTRON_CHAT_SPLIT}' split.")

    if args.mapping_file is not None:
        # --- Mapping mode: download (and cache) only the chat parquet shards,
        #     then use ds.select() for fast index-based random access.
        #     The chat data is cached after the first run; ds.select() is near-instant
        #     on subsequent runs. ---
        if not args.mapping_file.exists():
            err_msg = f"Mapping file {args.mapping_file} does not exist."
            raise FileNotFoundError(err_msg)

        with args.mapping_file.open("r", encoding="utf-8") as f:
            ordered_source_indices: list[int] = [
                int(line.strip()) for line in f if line.strip()
            ]
        print(f"Mapping file loaded: {len(ordered_source_indices)} entries.")

        ds = load_dataset(
            "parquet",
            data_files={"train": chat_parquet_urls},
            split="train",
            streaming=False,
        )
        # ds.select() preserves the given order of indices and operates on the
        # locally cached Arrow data, so it is fast even for large index lists.
        ds_subset = ds.select(ordered_source_indices)
        iterable = enumerate(
            tqdm(ds_subset, desc="Processing mapped samples", total=len(ordered_source_indices))
        )
    else:
        # --- Streaming mode: fetch chat shards on demand, no upfront download ---
        print(f"Streaming {NEMOTRON_DATASET_ID} (split={NEMOTRON_CHAT_SPLIT})...")
        ds = load_dataset(
            "parquet",
            data_files={"train": chat_parquet_urls},
            split="train",
            streaming=True,
        )
        stream = itertools.islice(ds, args.max_samples)  # islice(ds, None) = full split
        iterable = enumerate(tqdm(stream, desc="Loading Nemotron chat", total=args.max_samples))

    input_conversations: list[dict] = []
    skipped = 0
    for i, sample in iterable:
        raw_conversations = sample.get("conversations") or sample.get("messages") or []
        if not raw_conversations:
            skipped += 1
            continue

        msgs = parse_nemotron_conversation(raw_conversations)
        if not msgs:
            skipped += 1
            continue

        # Build a unique conversation ID, incorporating any source ID if available.
        source_id = sample.get("id") or sample.get("conversation_id") or f"{i:06}"
        conv_hash = id_for_conversation(msgs)
        cid = f"nemotron-chat-{source_id}_{conv_hash}"

        input_conversations.append({"conversation_id": cid, "conversations": msgs})

    print(
        f"Loaded {len(input_conversations)} conversations from Nemotron chat "
        f"(skipped {skipped} empty/invalid entries)."
    )

    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
