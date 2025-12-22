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

"""Utility functions."""

import hashlib
import json
import random
from pathlib import Path

import aiohttp


async def download_file(url: str, destination: Path) -> None:
    """Download a file from a URL to a specified destination."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session, session.get(url) as response:
        if response.status != 200:
            msg = f"Failed to download {url}: {response.status}"
            raise RuntimeError(msg)
        content = await response.read()
        destination.write_bytes(content)
        print(f"Downloaded {url} to {destination}")


def id_for_conversation(conversation: list) -> str:
    """Generate a unique ID for a conversation based on its content."""
    json_str = json.dumps(conversation, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    json_bytes = json_str.encode("utf-8")
    return hashlib.sha256(json_bytes).hexdigest()


def add_conversations_to_split(conversations: list, dataset_dir: Path, split: str) -> None:
    """Add conversations to a specific split in the dataset."""
    if len(conversations) == 0:
        return

    # Open the dataset file for the specified split, or create it if it doesn't exist
    dataset_file = dataset_dir / f"{split}.jsonl"
    all_conversations = []
    if dataset_file.exists():
        # load the existing conversations
        with dataset_file.open("r", encoding="utf-8") as f:
            all_conversations.extend([json.loads(line) for line in f if line.strip()])

    if any(not entry.get("conversation_id") for entry in all_conversations):
        msg = "All existing conversations must have a 'conversation_id' field."
        raise ValueError(msg)

    existing_ids = {entry["conversation_id"] for entry in all_conversations}
    num_new_entries = 0
    num_duplicates = 0
    for entry in conversations:
        if entry.get("conversation_id") is None:
            raise ValueError("Each conversation must have a 'conversation_id' field.")
        if entry["conversation_id"] not in existing_ids:
            all_conversations.append(
                {
                    "conversation_id": entry["conversation_id"],
                    "conversations": entry["conversations"],
                }
            )
            num_new_entries += 1
        else:
            num_duplicates += 1

    if num_duplicates > 0:
        print(
            f"Added {num_new_entries} new conversations to {dataset_file}, "
            f"skipped {num_duplicates} existing entries."
        )
    else:
        print(f"Added {num_new_entries} new conversations to {dataset_file}.")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    with dataset_file.open("w", encoding="utf-8") as f:
        for entry in all_conversations:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def mix_conversations_and_add_to_splits(
    conversations: list,
    dataset_dir: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> None:
    """Mix the conversations and add to the dataset's train, val, and test splits."""
    if train_ratio + val_ratio + test_ratio != 1.0:
        msg = "Ratios must sum to 1.0"
        raise ValueError(msg)
    if any(ratio < 0 for ratio in [train_ratio, val_ratio, test_ratio]):
        msg = "Ratios must be non-negative"
        raise ValueError(msg)

    total_conversations = len(conversations)
    train_count = int(total_conversations * train_ratio)
    val_count = int(total_conversations * val_ratio)

    if shuffle:
        random.seed(seed)
        random.shuffle(conversations)

    train_conversations = conversations[:train_count]
    val_conversations = conversations[train_count : train_count + val_count]
    test_conversations = conversations[train_count + val_count :]
    add_conversations_to_split(train_conversations, dataset_dir, "train")
    add_conversations_to_split(val_conversations, dataset_dir, "val")
    add_conversations_to_split(test_conversations, dataset_dir, "test")


def update_dataset_file_with_conversations(
    conversations: list, dataset_dir: Path, dataset_split: str
) -> None:
    """
    Update a dataset file with new conversations. The conversations are added to the specified
    split in the dataset. If the split is 'mix' or 'mix_test', the conversations are mixed and
    distributed into train, val, and test splits according to predefined ratios.
    """
    if dataset_split == "mix":
        print("Mixing conversations and adding to train, val, and test splits.")
        mix_conversations_and_add_to_splits(
            conversations,
            dataset_dir,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )
    elif dataset_split == "mix_test":
        print("Mixing conversations and adding to val and test splits.")
        mix_conversations_and_add_to_splits(
            conversations,
            dataset_dir,
            train_ratio=0.0,
            val_ratio=0.5,
            test_ratio=0.5,
        )
    else:
        add_conversations_to_split(conversations, dataset_dir, dataset_split)


def dataset_splits_explanation(default_split: str) -> str:
    """Return an explanation string for the dataset split argument."""
    return f"""Split to assign the processed conversations to.
        Can be any name, or one of ['mix', 'mix_test'].
        Default is '{default_split}'.

        If the provided split name matches an existing file in the dataset directory,
        the new conversations will be added to that file,
        avoiding duplicates based on conversation IDs.

        Special split names:
        - 'mix': Conversations will be randomly mixed and distributed into
            'train' (80%%), 'val' (10%%), and 'test' (10%%) splits.
        - 'mix_test': Conversations will be randomly mixed and distributed into
            'val' (50%%) and 'test' (50%%) splits.
        """
