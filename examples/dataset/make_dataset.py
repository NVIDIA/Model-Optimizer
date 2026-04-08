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

"""
Using a YAML file as an outline, initialize one or more conversation dataset files,
each as a JSONL file containing a list of conversations sampled from multiple source datasets.

Each source dataset is specified in the YAML file with its name, and which splits
from that dataset to include in the output conversation dataset files, as well as
bounds on the number of conversations to include from each split.

A global limit can also be placed on the total number of conversations in each output dataset file.

The dataset choices available are:
- "mtbench"
- "sharegpt"
- "ultrachat"
- "daring-anteater"
- "magpie"
- "nemotron-post-training-v2"

Here is an example YAML file:

```
outputs:
  - filename: "mixed_conversation.jsonl"
    global_limit: 5000 # downsample to 5000 total samples
    sources:
      - name: "mtbench"
        splits: ["all"]
      - name: "ultrachat"
        splits:
          train_gen: 0.5 # 50% of examples from train_gen split
          train_sft: 100 # 100 examples from train_sft split
          test_gen: "all" # all examples from test_gen split
```
"""

import argparse
import asyncio
import json
import logging
import random
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from datasets import load_dataset
from utils import download_file, id_for_conversation

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass
class SourceDatasetSpec:
    """Defines which dataset to read and how to sample from its splits."""

    name: str
    # Accepts list (implied "all") or dict (specific counts/percentages)
    splits: list[str] | dict[str, int | float | str]

    def __post_init__(self):
        # Normalize list format ["train"] -> {"train": "all"}
        if isinstance(self.splits, list):
            self.splits = dict.fromkeys(self.splits, "all")


@dataclass
class OutputConfig:
    """Defines a single target JSONL file to generate."""

    filename: str
    sources: list[SourceDatasetSpec]
    global_limit: int | None = None

    def __post_init__(self):
        # Convert dictionary dictionaries into strongly typed objects
        self.sources = [SourceDatasetSpec(**s) if isinstance(s, dict) else s for s in self.sources]


@dataclass
class DataMixingConfig:
    """The top-level configuration containing all output jobs."""

    outputs: list[OutputConfig]

    @classmethod
    def load(cls, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Instantiate objects from the parsed YAML dict
        return cls(outputs=[OutputConfig(**o) for o in data.get("outputs", [])])


def check_row_constraint(constraint) -> int | float | None:
    if constraint == "all":
        return None
    if constraint < 0:
        raise ValueError("Number of samples to use for a split cannot be negative.")
    if isinstance(constraint, (float, int)):
        return constraint
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare conversation datasets based on a YAML configuration."
    )

    parser.add_argument(
        "--config-file",
        "-c",
        "-f",
        type=Path,
        required=True,
        help="Path to the YAML configuration file specifying dataset construction.",
    )
    parser.add_argument(
        "--full-conversations",
        "--full",
        action="store_true",
        help="If set, include full conversations including assistant completions. "
        "By default, the last assistant completion is stripped to use the conversation as a prompt.",
    )

    return parser.parse_args()


def max_samples_for_constraint(total_size: int, row_constraint: int | float | None) -> int:
    """Get the maximum number of samples to draw from a dataset split based on the constraint."""
    if row_constraint is None:
        # "all"
        return total_size
    elif isinstance(row_constraint, float):
        # Percentage
        return int(total_size * row_constraint)
    else:
        # Absolute number
        return min(row_constraint, total_size)


MTBENCH_QUESTIONS_URL = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"


async def _load_mtbench_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    if split_name != "all":
        logger.warning("MTBench dataset has no splits; you should provide it as 'all'. Skipping.")
        yield 0
        return

    # Download the MTBench questions file if not provided
    mtbench_questions_file = (
        Path("~/.cache/modelopt/mtbench_questions.jsonl").expanduser().resolve()
    )
    if not mtbench_questions_file.exists():
        logger.info("Downloading MTBench questions dataset...")
        await download_file(MTBENCH_QUESTIONS_URL, mtbench_questions_file)

    # Error if we failed to download the file
    if not mtbench_questions_file.exists():
        err_msg = f"MTBench questions file {mtbench_questions_file} does not exist."
        raise FileNotFoundError(err_msg)

    with mtbench_questions_file.open("r", encoding="utf-8") as f:
        mtbench_raw = [json.loads(line) for line in f]

    random.shuffle(mtbench_raw)
    yield len(mtbench_raw)

    for entry in mtbench_raw:
        if not entry:
            continue
        prompt = entry.get("turns", [""])[0]
        if not prompt:
            continue
        prompt_id = f"mtbench-{entry['question_id']:03}-" + id_for_conversation(prompt)
        yield {"conversation_id": prompt_id, "conversations": [{"role": "user", "content": prompt}]}
    logger.info("Finished loading MTBench conversations.")


SHAREGPT_DATASET_URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"


def _parse_sharegpt_conversation(sharegpt_conv: dict) -> list[dict] | None:
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


async def _load_sharegpt_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    if split_name != "all":
        logger.warning("ShareGPT dataset has no splits; you should provide it as 'all'. Skipping.")
        yield 0
        return

    # Download the ShareGPT dataset if not provided
    sharegpt_file = Path("~/.cache/modelopt/sharegpt.json").expanduser().resolve()
    if not sharegpt_file.exists():
        logger.info("Downloading ShareGPT dataset...")
        await download_file(SHAREGPT_DATASET_URL, sharegpt_file)

    # Error if we failed to download the file
    if not sharegpt_file.exists():
        err_msg = f"ShareGPT file {sharegpt_file} does not exist."
        raise FileNotFoundError(err_msg)

    with sharegpt_file.open("r", encoding="utf-8") as f:
        sharegpt_raw = json.load(f)

    random.shuffle(sharegpt_raw)
    yield len(sharegpt_raw)

    for source_conv in sharegpt_raw:
        msgs = _parse_sharegpt_conversation(source_conv)
        if not msgs:
            continue
        cid = source_conv.get("id")
        conv_id = id_for_conversation(msgs)
        if cid:
            cid = f"{cid}-{conv_id}"
        else:
            cid = conv_id
        cid = f"sharegpt-{cid}"

        yield {"conversation_id": cid, "conversations": msgs}
    logger.info("Finished loading ShareGPT conversations.")


async def _load_ultrachat_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=split_name)
    ds = ds.shuffle(seed=42)
    yield len(ds)
    for i in range(len(ds)):
        prompt = ds[i]["prompt"].strip()
        prompt_id = ds[i]["prompt_id"].strip()
        if prompt:
            msgs = [{"role": "user", "content": prompt}]
            if not prompt_id:
                prompt_id = id_for_conversation(msgs)
            prompt_id = f"ultrachat-{split_name}-{prompt_id}"
            yield {"conversation_id": prompt_id, "conversations": msgs}
    logger.info(f"Finished loading UltraChat {split_name} conversations.")


def _parse_daring_anteater_conversation(daring_anteater_conv: list) -> list[dict] | None:
    """Parse a DaringAnteater conversation into a list of messages."""
    msgs = []
    for turn in daring_anteater_conv:
        if "from" in turn:
            role = turn["from"].lower()
        elif "role" in turn:
            role = turn["role"].lower()
        else:
            continue
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        if "value" in turn:
            content = turn["value"]
        elif "text" in turn:
            content = turn["text"]
        elif "content" in turn:
            content = turn["content"]
        else:
            continue
        content = content.strip()
        if content:
            msgs.append({"role": role, "content": content})

    return msgs


async def _load_daring_anteater_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    ds = load_dataset("nvidia/Daring-Anteater", split=split_name)
    ds = ds.shuffle(seed=42)
    yield len(ds)
    for i in range(len(ds)):
        conversations = ds[i]["conversations"]
        if conversations and isinstance(conversations, list):
            prompt_id = f"daring-anteater-{split_name}-" + id_for_conversation(conversations)
            processed_conversations = _parse_daring_anteater_conversation(conversations)
            if processed_conversations:
                yield {"conversation_id": prompt_id, "conversations": processed_conversations}
    logger.info(f"Finished loading Daring-Anteater {split_name} conversations.")


async def _load_magpie_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    if split_name not in ("300k", "500k", "1M"):
        logger.warning("Only Magpie splits '300k', '500k' and '1M' are available. Skipping.")
        yield 0
        return
    if split_name == "500k":
        ds = load_dataset("Magpie-Align/Magpie-Llama-3.3-Pro-500K-Filtered", split="train")
    elif split_name == "300k":
        ds = load_dataset("Magpie-Align/Magpie-Llama-3.1-Pro-300K-Filtered", split="train")
    else:
        assert split_name == "1M"
        ds = load_dataset("Magpie-Align/Magpie-Llama-3.3-Pro-1M-v0.1", split="train")
    ds = ds.shuffle(seed=42)
    yield len(ds)
    for i in range(len(ds)):
        prompt = ds[i]["instruction"].strip()
        if prompt:
            conversations = [{"role": "user", "content": prompt}]
            prompt_id = f"magpie-{split_name}-" + id_for_conversation(conversations)
            yield {"conversation_id": prompt_id, "conversations": conversations}
    logger.info(f"Finished loading Magpie {split_name} conversations.")


async def load_nemotron_post_training_v2_conversations(
    split_name: str,
) -> AsyncGenerator[int | dict[str, Any], None]:
    nemotron_splits = [
        "math",
        "code",
        "chat",
        "stem",
        "multilingual_ja",
        "multilingual_it",
        "multilingual_de",
        "multilingual_es",
        "multilingual_fr",
    ]
    if split_name not in nemotron_splits:
        logger.warning(
            f"Nemotron Post-Training V2 splits are: {', '.join(nemotron_splits)}. Skipping."
        )
        yield 0
        return

    ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split_name)
    ds = ds.shuffle(seed=42)
    yield len(ds)

    for i in range(len(ds)):
        conversations = ds[i]["messages"]
        if conversations and isinstance(conversations, list):
            # Strip leading empty system messages
            while (
                conversations
                and conversations[0]["role"] == "system"
                and not conversations[0]["content"].strip()
            ):
                conversations.pop(0)
            prompt_id = f"nemotron-post-training-v2-{split_name}-" + id_for_conversation(
                conversations
            )
            yield {"conversation_id": prompt_id, "conversations": conversations}
    logger.info(f"Finished loading Nemotron Post-Training V2 {split_name} conversations.")


async def load_conversations_for_split(
    dataset_name: str,
    split_name: str,
    row_constraint: int | float | None,
    strip_last_completion: bool = True,
) -> list[dict]:
    if dataset_name == "mtbench":
        samples_it = _load_mtbench_conversations(split_name)
    elif dataset_name == "sharegpt":
        samples_it = _load_sharegpt_conversations(split_name)
    elif dataset_name == "ultrachat":
        samples_it = _load_ultrachat_conversations(split_name)
    elif dataset_name == "daring-anteater":
        samples_it = _load_daring_anteater_conversations(split_name)
    elif dataset_name == "magpie":
        samples_it = _load_magpie_conversations(split_name)
    elif dataset_name == "nemotron-post-training-v2":
        samples_it = load_nemotron_post_training_v2_conversations(split_name)
    else:
        logger.warning(f"Dataset {dataset_name} is not yet implemented. Ignoring.")
        return []

    num_samples = await samples_it.__anext__()
    assert isinstance(num_samples, int), "First yielded value must be the total number of samples."
    deduplication_ids = set()
    unique_samples = []
    max_num_samples = max_samples_for_constraint(num_samples, row_constraint)
    async for sample in samples_it:
        assert isinstance(sample, dict) and "conversations" in sample, (
            "Each conversation sample must be a dict with a 'conversations' field."
        )

        # Strip the last turn of the conversation as long as it is an assistant completion,
        # since we want to use these conversations as prompts only.
        if strip_last_completion:
            while sample["conversations"] and sample["conversations"][-1]["role"] != "user":
                sample["conversations"].pop()

        if not sample["conversations"]:
            continue

        sample["source_dataset"] = dataset_name
        sample["source_split"] = split_name

        # Deduplicate based on the first 512 characters from each turn.
        # To avoid too many similar conversations with minor differences.
        truncated_conversations = [
            {"role": msg["role"], "content": msg["content"][0:512]}
            for msg in sample["conversations"]
        ]
        dedup_id = id_for_conversation(truncated_conversations)
        if dedup_id not in deduplication_ids:
            deduplication_ids.add(dedup_id)
            unique_samples.append(sample)
            if len(unique_samples) >= max_num_samples:
                break
    return unique_samples


async def main(args: argparse.Namespace) -> None:
    config = DataMixingConfig.load(args.config_file)

    for output in config.outputs:
        all_conversations_promises = []
        for source in output.sources:
            for split_name, constraint in source.splits.items():
                row_constraint = check_row_constraint(constraint)
                if row_constraint == 0:
                    continue  # Skip this split, no samples requested
                all_conversations_promises.append(
                    load_conversations_for_split(
                        source.name,
                        split_name,
                        row_constraint,
                        strip_last_completion=not args.full_conversations,
                    )
                )

        all_conversations_results = await asyncio.gather(*all_conversations_promises)
        all_conversations = []
        num_conversations_per_split = {}
        for conversations in all_conversations_results:
            all_conversations.extend(conversations)

        total_num_conversations = len(all_conversations)
        if output.global_limit is not None and total_num_conversations > output.global_limit:
            random_indices = random.sample(range(total_num_conversations), output.global_limit)
            all_conversations = [all_conversations[i] for i in random_indices]
            logger.info(
                "Subsampling uniformly from %d to global limit of %d conversations.",
                total_num_conversations,
                output.global_limit,
            )
        else:
            random.shuffle(all_conversations)

        for conversation in all_conversations:
            key = (conversation["source_dataset"], conversation["source_split"])
            num_conversations_per_split[key] = num_conversations_per_split.get(key, 0) + 1

        # Metadata for pretty-printing
        max_ds_len = max((len(ds) for ds, _ in num_conversations_per_split), default=0)
        max_split_len = max((len(sp) for _, sp in num_conversations_per_split), default=0)
        logger.info("Dataset splits used for output '%s':", output.filename)
        num_conversations_per_split = dict(
            sorted(num_conversations_per_split.items(), key=lambda item: (item[0][0], item[0][1]))
        )
        for (dataset_name, split_name), num_convs in num_conversations_per_split.items():
            logger.info(
                f"  - {dataset_name:<{max_ds_len}} / {split_name:<{max_split_len}} : {num_convs:<8} conversations"
            )

        logger.info(f"Writing {len(all_conversations)} conversations to {output.filename}")
        output_path = Path(output.filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            for entry in all_conversations:
                assert "conversations" in entry, (
                    "Each conversation entry must have a 'conversations' field."
                )
                if "conversation_id" not in entry:
                    entry["conversation_id"] = id_for_conversation(entry["conversations"])
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import asyncio

    args = parse_args()
    asyncio.run(main(args))
