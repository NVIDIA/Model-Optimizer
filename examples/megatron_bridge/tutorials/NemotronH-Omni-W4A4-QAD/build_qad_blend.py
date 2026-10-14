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
rows: the first ones of its Hugging Face split, streamed in order, that Megatron-Bridge accepts
as text-only chat. Rows come in three shapes:

* ``messages``: used as-is.
* ``text`` with ``<extra_id_1>User`` / ``Assistant`` markers (SFT-General): parsed into messages,
  dropping empty turns.
* ``text`` without markers (SFT-Code, SFT-MATH): one assistant turn, so the whole text is trained.

A source that fails to load or yields fewer rows than its share stops the run, so the mixture
cannot change silently.

Each source is cached under ``<output_dir>/parts/``, written atomically, so an interrupted run
resumes and an unreadable part is rebuilt. Run inside the NeMo container.
"""

import argparse
import json
import os
import random
import re

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
    (
        2,
        "nvidia/Nemotron-SFT-Instruction-Following-Chat-v2",
        "default",
        "reasoning_off",
        "messages",
    ),
]
TURN = re.compile(
    r"<extra_id_\d+>(User|Assistant|System)\n?(.*?)(?=<extra_id_\d+>|\Z)",
    re.DOTALL,
)
ROLE = {"User": "user", "Assistant": "assistant", "System": "system"}


# ``datasets`` and Megatron-Bridge are imported inside the two functions below, not at module
# level, so the unit tests can load this script without either installed.


def stream(repo: str, config: str, split: str):
    """Yield the rows of one Hugging Face split, in order, without downloading it."""
    from datasets import load_dataset

    yield from load_dataset(repo, config, split=split, streaming=True)


def is_usable(chat: dict) -> bool:
    """Whether Megatron-Bridge's chat path accepts the conversation as text-only chat."""
    from megatron.bridge.data.conversation_processing import is_text_only_chat_example

    return is_text_only_chat_example(chat)


def to_chat(row: dict, field: str) -> dict | None:
    if field == "messages":
        return {"messages": row["messages"]} if row.get("messages") else None
    text = row.get("text") or ""
    if not text:
        return None
    if "<extra_id_" in text:
        turns = [
            {"role": ROLE[r], "content": c.strip()} for r, c in TURN.findall(text) if c.strip()
        ]
        # Markers that parse to no turns make the row malformed, not plain text: falling through
        # would train the raw <extra_id_*> delimiters as assistant content.
        return {"messages": turns} if turns else None
    return {"messages": [{"role": "assistant", "content": text}]}


def take(repo: str, config: str, split: str, field: str, want: int) -> tuple[list[dict], int]:
    """Stream the split and keep the first ``want`` usable rows; give up after ``10 * want`` rejects."""
    got, dropped = [], 0
    for row in stream(repo, config, split):
        chat = to_chat(row, field)
        if chat is None or not is_usable(chat):
            dropped += 1
            if dropped > want * 10:
                break
            continue
        got.append(chat)
        if len(got) >= want:
            break
    return got, dropped


def write_jsonl(path: str, rows: list[dict]) -> None:
    """Write ``rows`` to ``path`` atomically: an interrupted write never leaves a partial file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    os.replace(tmp, path)


def read_part(path: str) -> list[dict] | None:
    """Read a cached part, or ``None`` if it is missing or unreadable (it is then rebuilt)."""
    try:
        with open(path) as f:
            return [json.loads(line) for line in f]
    except (OSError, ValueError):
        return None


def build(output_dir: str, total: int, seed: int) -> dict[str, int]:
    """Build ``train.jsonl`` and ``validation.jsonl`` under ``output_dir``; return the split sizes."""
    parts = os.path.join(output_dir, "parts")
    os.makedirs(parts, exist_ok=True)
    total_weight = sum(w for w, *_ in SOURCES)
    rows = []
    for weight, repo, config, split, field in SOURCES:
        want = max(1, round(total * weight / total_weight))
        part = os.path.join(parts, f"{repo.split('/')[-1]}__{config}__{split}.jsonl")
        got = read_part(part)
        if got is not None and len(got) >= want:
            print(f"{repo} {split}: {want} rows (cached)", flush=True)
        else:
            try:
                got, dropped = take(repo, config, split, field, want)
            except (
                Exception
            ) as e:  # e.g. a gated split, or a schema the streaming reader cannot cast
                raise RuntimeError(f"{repo} {split}: failed to load") from e
            if len(got) < want:
                raise RuntimeError(f"{repo} {split}: only {len(got)}/{want} usable rows")
            write_jsonl(part, got)
            print(f"{repo} {split}: {want} rows ({dropped} dropped)", flush=True)
        rows.extend(got[:want])

    random.Random(seed).shuffle(rows)
    num_validation = min(1000, len(rows) // 20)
    splits = {"validation": rows[:num_validation], "train": rows[num_validation:]}
    if not splits["train"] or not splits["validation"]:
        raise RuntimeError(f"too few rows ({len(rows)}) for a train/validation split")
    for name, chunk in splits.items():
        write_jsonl(os.path.join(output_dir, f"{name}.jsonl"), chunk)
        print(f"wrote {name}.jsonl: {len(chunk)} rows", flush=True)
    return {name: len(chunk) for name, chunk in splits.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--total", type=int, default=36800, help="Rows to request across sources.")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    build(args.output_dir, args.total, args.seed)


if __name__ == "__main__":
    main()
