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
import json
from pathlib import Path

import torch
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
        "--input-file",
        type=Path,
        required=True,
        help="""Path to the input `jsonl` file containing conversations.
        Each entry must have a unique `conversation_id` field and a `conversations` field
        containing a list of messages.""",
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

    return parser.parse_args()


async def main(args: argparse.Namespace) -> None:
    all_conversations = []
    with args.input_file.open("r", encoding="utf-8") as f:
        all_conversations.extend([json.loads(line) for line in f if line.strip()])

    print("Loaded", len(all_conversations), "conversations from", args.input_file)

    model = AutoModel.from_pretrained(args.model, torch_dtype="auto", device_map="auto")
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_skipped_too_long = 0
    num_invalid = 0
    num_success = 0
    num_total_conversations = min(
        len(all_conversations), args.debug_max_num_conversations or len(all_conversations)
    )
    for idx, entry in enumerate(
        tqdm(
            all_conversations[: args.debug_max_num_conversations],
            desc="Processing conversations",
            total=num_total_conversations,
        )
    ):
        conversation_id = entry.get("conversation_id", "{:08d}".format(idx))
        conversations = entry["conversations"]
        if not conversations or not isinstance(conversations, list):
            num_invalid += 1
            continue

        # Tokenize and check length
        input_ids = tokenizer.apply_chat_template(
            conversations, return_tensors="pt", add_generation_template=False
        )
        num_input_tokens = input_ids.shape[1]
        if num_input_tokens <= 10 or num_input_tokens > args.max_seq_len:
            num_skipped_too_long += 1
            continue

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
            output_hidden_states = outputs.last_hidden_state.squeeze(0).cpu()
        output_file = output_dir / f"{conversation_id}.pt"
        num_success += 1
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

    if num_skipped_too_long > 0:
        print(f"Skipped {num_skipped_too_long} conversations due to length constraints.")
    if num_invalid > 0:
        print(f"Skipped {num_invalid} invalid conversations without proper fields.")

    if num_success == num_total_conversations:
        print(f"Successfully processed all {num_success} conversations.")
    else:
        print(
            f"Successfully processed {num_success} out of {num_total_conversations} conversations."
        )


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
