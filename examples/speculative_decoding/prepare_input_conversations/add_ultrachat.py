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

"""Add UltraChat conversations to a conversation dataset."""

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
    parser = argparse.ArgumentParser(description="Load UltraChat conversations.")

    parser.add_argument(
        "--ultrachat-split",
        type=str,
        default="train_sft",
        help="Split of the HuggingFace UltraChat dataset to load. Default is 'train_sft'.",
    )

    parser.add_argument(
        "--output-split-name",
        type=str,
        default="ultrachat",
        help=dataset_splits_explanation("ultrachat"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save the conversations file(s) into. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=args.ultrachat_split, streaming=False)
    input_conversations = []
    for i in tqdm(
        range(len(ds)),
        desc=f"Loading UltraChat split {args.ultrachat_split}",
        total=len(ds),
    ):
        prompt = ds[i]["prompt"].strip()
        prompt_id = ds[i]["prompt_id"].strip()
        if prompt and prompt_id:
            msgs = [{"role": "user", "content": prompt}]
            prompt_id = (
                f"ultrachat-{args.ultrachat_split}_{i:06}-{prompt_id}_" + id_for_conversation(msgs)
            )
            input_conversations.append({"conversation_id": prompt_id, "conversations": msgs})

    print(f"Loaded {len(input_conversations)} prompts from UltraChat.")

    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
