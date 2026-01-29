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

"""Extract hidden states from an HF-compatible LLM."""

import argparse
import asyncio
from pathlib import Path

import torch
from datasets import load_dataset
from tqdm import tqdm as tqdm
from transformers import AutoModel, AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations
        by running full conversations through a Hugging Face model."""
    )

    ## Model & Generation Parameters ##
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the served model.",
    )

    ## Client Parameters ##
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="""Maximum number of tokens in a conversation. Longer conversations will be skipped.
        Defaults to 3072 tokens.""",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--input-data",
        type=Path,
        required=True,
        help="""Path to the `jsonl` file or directory containing `jsonl` files.""",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="""Root directory in which to save the hidden states.
        The data will be saved as a torch (`.pt`) dump file for each conversation.""",
    )
    parser.add_argument(
        "--debug-max-num-conversations",
        type=int,
        default=None,
        help="""For debugging purposes, limit the number of conversations processed.
        Default is None, meaning no limit.""",
    )
    parser.add_argument(
        "--dp-rank",
        type=int,
        default=0,
        help="""Data parallel rank. TASK_ID on SLURM.""",
    )
    parser.add_argument(
        "--dp-world-size",
        type=int,
        default=1,
        help="""Data parallel world size. Number of tasks on SLURM.""",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    # Load conversations
    if args.input_data.is_file() and str(args.input_data).endswith(".jsonl"):
        dataset = load_dataset("json", data_files=str(args.input_data), split="train")
    elif args.input_data.is_dir():
        dataset = load_dataset(
            "json", data_files={"train": f"{args.input_data}/*.jsonl"}, split="train"
        )
    else:
        raise ValueError(
            f"input_data must be a .jsonl file or directory containing .jsonl files, got: {args.input_data}"
        )
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    print(
        f"Sharded dataset to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}"
    )

    # Remove already dumped conversations
    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "conversation_id is required"
        output_file = args.output_dir / f"{conversation_id}.pt"
        return not output_file.exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(
        "Removed",
        original_num - len(dataset),
        "conversations due to existing output files",
    )

    # For debugging
    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    model = AutoModel.from_pretrained(
        args.model, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_skipped_too_long = 0
    num_invalid = 0
    num_success = 0
    pbar = tqdm(total=len(dataset), desc=f"DP#{args.dp_rank} Processing conversations")

    async def dump_hidden_states(idx: int, conversation_id: int, input_ids: torch.Tensor):
        nonlocal num_success
        nonlocal num_hidden_layers

        # Get hidden states
        with torch.inference_mode():
            outputs = model(input_ids=input_ids.to(model.device), output_hidden_states=True)
            if num_hidden_layers is None:
                num_hidden_layers = len(outputs.hidden_states) - 1
            else:
                assert num_hidden_layers + 1 == len(outputs.hidden_states), (
                    f"Expected {num_hidden_layers}+1 layers of hidden states, but got {len(outputs.hidden_states)}."
                )
            # Extract hidden states from layers with index (2, N/2, N-3), and the output hidden states
            hidden_states = outputs.hidden_states
            selected_layer_indices = [
                2,
                max(0, num_hidden_layers // 2),
                max(1, num_hidden_layers - 3),
            ]
            selected_layer_indices = sorted(set(selected_layer_indices))
            aux_hidden_states = torch.cat(
                [hidden_states[i].squeeze(0).cpu() for i in selected_layer_indices], dim=-1
            )
            output_hidden_states = hidden_states[-1].squeeze(0).cpu()
        output_file = output_dir / f"{conversation_id}.pt"

        with open(output_file, "wb") as f:
            torch.save(
                {
                    "input_ids": input_ids.squeeze(0).cpu(),
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "conversation_id": conversation_id,
                },
                f,
            )

        num_success += 1
        pbar.update(1)

    async def submit_generates():
        nonlocal num_skipped_too_long
        nonlocal num_invalid
        tasks = []
        idx = 0
        for entry in dataset:
            conversation_id = entry.get("conversation_id", entry.get("uuid"))

            conversations = entry["conversations"]
            if not conversations or not isinstance(conversations, list):
                num_invalid += 1
                continue

            # Tokenize and check length
            input_ids = tokenizer.apply_chat_template(
                conversations, return_tensors="pt", add_generation_template=False
            )["input_ids"]
            num_input_tokens = input_ids.shape[1]
            if num_input_tokens <= 10 or num_input_tokens > args.max_seq_len:
                num_skipped_too_long += 1
                continue

            tasks.append(dump_hidden_states(idx, conversation_id, input_ids))
            # Increment only for valid conversations to match dump file index
            idx += 1
        await asyncio.gather(*tasks)

    asyncio.run(submit_generates())

    if num_skipped_too_long > 0:
        print(f"Skipped {num_skipped_too_long} conversations due to length constraints.")
    if num_invalid > 0:
        print(f"Skipped {num_invalid} invalid conversations without proper fields.")

    if num_success == len(dataset):
        print(f"Successfully processed all {num_success} conversations.")
    else:
        print(f"Successfully processed {num_success} out of {len(dataset)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
