# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Materialize the tutorial's QAD blend as chat jsonl (train.jsonl / validation.jsonl).

``distill.py --sft_hf_dataset`` reads one source per split and has no blend weights, so the
weighted mixture is written out here. Each source contributes ``total * weight / sum(weights)``
rows, taken from the start of its Hugging Face split. Rows come in three shapes:

* ``messages``: used as-is.
* ``text`` with ``<extra_id_1>User`` / ``Assistant`` markers (SFT-General): parsed into messages.
* ``text`` without markers (SFT-Code, SFT-MATH): one assistant turn, so the whole text is trained.

Rows Megatron-Bridge does not accept as text-only chat are skipped. Run inside the NeMo container.
"""

import argparse
import json
import os
import random
import re

from datasets import load_dataset
from megatron.bridge.data.conversation_processing import is_text_only_chat_example

# (weight, repo, config, split, field)
SOURCES = [
    (5, "nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-Code", "train", "text"),
    (20, "nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-General", "train", "text"),
    (5, "nvidia/Nemotron-Pretraining-SFT-v1", "Nemotron-SFT-MATH", "train", "text"),
    (10, "nvidia/Nemotron-Math-v2", "default", "high_part00", "messages"),
    (17, "nvidia/Nemotron-SFT-Math-v3", "default", "train", "messages"),
    (
        15,
        "nvidia/Nemotron-Competitive-Programming-v1",
        "default",
        "competitive_coding_python_part00",
        "messages",
    ),
    (
        5,
        "nvidia/Nemotron-Competitive-Programming-v1",
        "default",
        "competitive_coding_cpp_part00",
        "messages",
    ),
    (8, "nvidia/Nemotron-Post-Training-Dataset-v1", "default", "stem", "messages"),
    (3, "nvidia/Nemotron-Science-v1", "default", "MCQ", "messages"),
    (2, "nvidia/Nemotron-Science-v1", "default", "RQA", "messages"),
    (3, "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2", "default", "reasoning_on", "messages"),
    (
        2,
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
        "default",
        "reasoning_off",
        "messages",
    ),
    (5, "nvidia/Nemotron-Agentic-v1", "default", "tool_calling", "messages"),
]
TURN = re.compile(
    r"<extra_id_\d+>(User|Assistant|System)\n?(.*?)(?=<extra_id_\d+>(?:User|Assistant|System)|\Z)",
    re.DOTALL,
)
ROLE = {"User": "user", "Assistant": "assistant", "System": "system"}


def to_chat(row: dict, field: str) -> dict | None:
    if field == "messages":
        return {"messages": row["messages"]} if row.get("messages") else None
    text = row.get("text") or ""
    if not text:
        return None
    turns = [{"role": ROLE[r], "content": c.strip()} for r, c in TURN.findall(text)]
    return {"messages": turns or [{"role": "assistant", "content": text}]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total", type=int, default=40000, help="Rows to request across sources.")
    parser.add_argument("--num_validation", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    total_weight = sum(w for w, *_ in SOURCES)
    rows = []
    for weight, repo, config, split, field in SOURCES:
        want = max(1, round(args.total * weight / total_weight))
        try:
            ds = load_dataset(repo, config, split=f"{split}[:{want * 3}]")
        except Exception as e:  # e.g. a gated or renamed split
            print(f"{repo} {split}: skipped ({type(e).__name__}: {e})")
            continue
        got, dropped = [], 0
        for row in ds:
            chat = to_chat(row, field)
            if chat is None or not is_text_only_chat_example(chat):
                dropped += 1
                continue
            got.append(chat)
            if len(got) == want:
                break
        rows.extend(got)
        print(f"{repo} {split}: {len(got)}/{want} rows ({dropped} dropped)")

    random.Random(args.seed).shuffle(rows)
    os.makedirs(args.output_dir, exist_ok=True)
    splits = {"validation": rows[: args.num_validation], "train": rows[args.num_validation :]}
    for name, chunk in splits.items():
        path = os.path.join(args.output_dir, f"{name}.jsonl")
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in chunk)
        print(f"wrote {path}: {len(chunk)} rows")


if __name__ == "__main__":
    main()
