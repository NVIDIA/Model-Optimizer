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

"""Add MTBench conversations to a conversation dataset."""

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

MTBENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load MTBench conversations.")

    parser.add_argument(
        "--mtbench-questions-file",
        type=Path,
        required=False,
        help="""Path to the MTBench questions.jsonl file.
        If not provided, it will be downloaded and saved to ~/.cache/""",
    )

    parser.add_argument(
        "--output-split-name",
        type=str,
        default="mtbench",
        help=dataset_splits_explanation("mtbench"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("input_conversations/"),
        help="Path to save the conversations file(s) into. Default is 'input_conversations/'.",
    )

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    # Download the MTBench questions file if not provided
    if not args.mtbench_questions_file:
        args.mtbench_questions_file = (
            Path("~/.cache/mtbench_questions.jsonl").expanduser().resolve()
        )
        if not args.mtbench_questions_file.exists():
            print("Downloading MTBench questions dataset...")
            await download_file(MTBENCH_QUESTIONS_URL, args.mtbench_questions_file)
        else:
            print(f"Using existing MTBench questions file {args.mtbench_questions_file}")

    # Error if we failed to download the file or if it was provided but does not exist
    if not args.mtbench_questions_file.exists():
        err_msg = f"MTBench questions file {args.mtbench_questions_file} does not exist."
        raise FileNotFoundError(err_msg)

    with args.mtbench_questions_file.open("r", encoding="utf-8") as f:
        mtbench_raw = [json.loads(line) for line in f]

    input_conversations: list[dict] = []
    for entry in tqdm(mtbench_raw, desc="Loading MTBench", total=len(mtbench_raw)):
        if not entry:
            continue
        prompt = entry.get("turns", [""])[0]
        if not prompt:
            continue
        prompt_id = f"mtbench-{entry['question_id']:03}_" + id_for_conversation(prompt)
        input_conversations.append(
            {"conversation_id": prompt_id, "conversations": [{"role": "user", "content": prompt}]}
        )

    print(f"Loaded {len(input_conversations)} filtered conversations from MTBench.")

    update_dataset_file_with_conversations(
        input_conversations, args.output_dir, args.output_split_name
    )


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
