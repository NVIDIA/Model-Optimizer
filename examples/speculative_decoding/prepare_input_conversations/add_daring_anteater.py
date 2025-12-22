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

"""Add Daring-Anteater conversations to a conversation dataset."""

import argparse
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from utils import (
    dataset_splits_explanation,
    id_for_conversation,
    update_dataset_file_with_conversations,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load Daring-Anteater conversations.")

    parser.add_argument(
        "--output-split-name",
        type=str,
        default="daring-anteater",
        help=dataset_splits_explanation("daring-anteater"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save the conversations file(s) into. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    ds = load_dataset("nvidia/Daring-Anteater", split="train", streaming=False)
    input_conversations = []
    for i in tqdm(
        range(len(ds)),
        desc="Loading Daring-Anteater dataset",
        total=len(ds),
    ):
        conversations = ds[i]["conversations"]
        if conversations and isinstance(conversations, list):
            prompt_id = f"daring-anteater-{i:05}_" + id_for_conversation(conversations)
            processed_conversations = []
            for msg in conversations:
                if "from" in msg:
                    role = msg["from"].lower()
                elif "role" in msg:
                    role = msg["role"].lower()
                else:
                    continue
                if role == "human":
                    role = "user"
                elif role == "gpt":
                    role = "assistant"

                if "value" in msg:
                    content = msg["value"]
                elif "text" in msg:
                    content = msg["text"]
                elif "content" in msg:
                    content = msg["content"]
                else:
                    continue
                content = content.strip()
                if content:
                    processed_conversations.append({"role": role, "content": content})

            input_conversations.append(
                {"conversation_id": prompt_id, "conversations": processed_conversations}
            )

    print(f"Loaded {len(input_conversations)} prompts from Daring-Anteater.")

    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
