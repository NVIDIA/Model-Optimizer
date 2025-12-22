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

"""Add ShareGPT conversations to a conversation dataset."""

import argparse
import json
from pathlib import Path

from tqdm import tqdm
from utils import (
    dataset_splits_explanation,
    download_file,
    id_for_conversation,
    update_dataset_file_with_conversations,
)

SHAREGPT_DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load ShareGPT conversations.")

    parser.add_argument(
        "--sharegpt-file",
        type=Path,
        required=False,
        help="""Path to the ShareGPT JSON file containing conversations.
        If not provided, it will be downloaded and saved to ~/.cache/""",
    )

    parser.add_argument(
        "--output-split-name",
        type=str,
        default="sharegpt",
        help=dataset_splits_explanation("sharegpt"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save the conversations file(s) into. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


def parse_sharegpt_conversation(sharegpt_conv: dict) -> list[dict] | None:
    """Parse a ShareGPT conversation into a list of messages."""
    msgs = []
    for turn in sharegpt_conv.get("conversations", []):
        if turn.get("from") in ["human", "user"]:
            role = "user"
        elif turn.get("from") in ["gpt", "chatgpt", "bard"]:
            role = "assistant"
        elif turn.get("from") == "system":
            # ShareGPT system messages are metadata, skip them
            continue
        elif turn.get("from") == "bing":
            # Bing conversations are skipped for training, omit it
            return None
        else:
            err_msg = f"Unknown role in conversation: {turn.get('from')}"
            raise ValueError(err_msg)

        value = turn.get("value", "").strip()
        if value:
            msgs.append({"role": role, "content": value})

    return msgs


async def main(args: argparse.Namespace) -> None:
    # Download the ShareGPT dataset if not provided
    if not args.sharegpt_file:
        args.sharegpt_file = Path("~/.cache/sharegpt.json").expanduser().resolve()
        if not args.sharegpt_file.exists():
            print("Downloading ShareGPT dataset...")
            await download_file(SHAREGPT_DATASET_URL, args.sharegpt_file)
        else:
            print(f"Using existing ShareGPT file at {args.sharegpt_file}")

    # Error if we failed to download the file or if it was provided but does not exist
    if not args.sharegpt_file.exists():
        err_msg = f"ShareGPT file {args.sharegpt_file} does not exist."
        raise FileNotFoundError(err_msg)

    with args.sharegpt_file.open("r", encoding="utf-8") as f:
        sharegpt_raw = json.load(f)

    input_conversations: list[dict] = []
    for source_conv in tqdm(sharegpt_raw, desc="Loading ShareGPT", total=len(sharegpt_raw)):
        msgs = parse_sharegpt_conversation(source_conv)
        if not msgs:
            continue
        cid = source_conv.get("id")
        conv_id = id_for_conversation(msgs)
        if cid:
            cid = f"{cid}_{conv_id}"
        else:
            cid = conv_id
        cid = f"sharegpt-{cid}"

        input_conversations.append({"conversation_id": cid, "conversations": msgs})

    print(f"Loaded {len(input_conversations)} filtered conversations from ShareGPT.")

    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
