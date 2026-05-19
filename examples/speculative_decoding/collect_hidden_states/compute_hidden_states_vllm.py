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

"""Extract hidden states from an LLM using vLLM + speculators."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from speculators.data_generation import VllmHiddenStatesGenerator
from tqdm import tqdm
from transformers import AutoTokenizer

REMOVE_THINK_CHAT_TEMPLATE = (
    "{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations using vLLM + speculators."""
    )

    parser.add_argument("--model", type=str, required=True, help="HF model path.")
    parser.add_argument(
        "--max-seq-len", type=int, default=3072, help="Max tokens per conversation."
    )
    parser.add_argument(
        "--input-data", type=Path, required=True, help="Path to jsonl file or directory."
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory to save hidden states."
    )
    parser.add_argument("--dp-rank", type=int, default=0, help="Data parallel rank.")
    parser.add_argument("--dp-world-size", type=int, default=1, help="Data parallel world size.")
    parser.add_argument(
        "--trust_remote_code", action="store_true", help="Trust remote code for HF models."
    )
    parser.add_argument("--tp", type=int, default=None, help="Tensor parallel size.")
    parser.add_argument(
        "--debug-max-num-conversations", type=int, default=None, help="Limit conversations."
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
        raise ValueError(f"input_data must be a .jsonl file or directory, got: {args.input_data}")
    print(f"Loaded {len(dataset)} conversations from {args.input_data}")

    # Shard data
    if args.dp_world_size > 1:
        dataset = dataset.shard(num_shards=args.dp_world_size, index=args.dp_rank)
    print(f"Sharded to {len(dataset)} conversations for DP#{args.dp_rank}/{args.dp_world_size}")

    # Remove already dumped conversations
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    def keep_conversation(entry):
        conversation_id = entry.get("conversation_id", entry.get("uuid", None))
        assert conversation_id is not None, "conversation_id is required"
        return not (output_dir / f"{conversation_id}.pt").exists()

    original_num = len(dataset)
    dataset = dataset.filter(keep_conversation)
    print(f"Removed {original_num - len(dataset)} conversations due to existing output files")

    if args.debug_max_num_conversations is not None:
        dataset = dataset.select(range(args.debug_max_num_conversations))

    # Tokenize conversations
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.chat_template = tokenizer.chat_template.replace(REMOVE_THINK_CHAT_TEMPLATE, "")

    # Prepare prompts for vLLM
    prompts = []
    conversation_ids = []
    num_skipped_too_long = 0
    num_invalid = 0

    for entry in dataset:
        conversation_id = entry.get("conversation_id", entry.get("uuid"))
        conversations = entry["conversations"]
        if not conversations or not isinstance(conversations, list):
            num_invalid += 1
            continue

        tokenized = tokenizer.apply_chat_template(
            conversations, return_tensors="pt", add_generation_prompt=False
        )
        # transformers 5.x: BatchEncoding may not inherit from dict; use .input_ids
        if hasattr(tokenized, "input_ids"):
            input_ids = tokenized.input_ids
        elif hasattr(tokenized, "__getitem__") and "input_ids" in tokenized:
            input_ids = tokenized["input_ids"]
        else:
            input_ids = tokenized
        if not hasattr(input_ids, "shape"):
            input_ids = torch.tensor(input_ids)
        num_tokens = input_ids.shape[0] if input_ids.dim() == 1 else input_ids.shape[1]
        if num_tokens <= 10 or num_tokens > args.max_seq_len:
            num_skipped_too_long += 1
            continue

        prompts.append(input_ids.squeeze(0))
        conversation_ids.append(conversation_id)

    print(
        f"Prepared {len(prompts)} prompts ({num_skipped_too_long} skipped too long, {num_invalid} invalid)"
    )

    if len(prompts) == 0:
        print("No prompts to process.")
        return

    # Initialize vLLM hidden states generator
    tp = args.tp
    if tp is None:
        import torch as _torch

        tp = _torch.cuda.device_count()

    generator = VllmHiddenStatesGenerator(
        model_path=args.model,
        tensor_parallel_size=tp,
        max_model_len=args.max_seq_len,
    )

    # Generate hidden states
    results = generator.generate(prompts)

    # Save in the same format as compute_hidden_states_hf.py
    num_success = 0
    for conv_id, result in tqdm(zip(conversation_ids, results), total=len(results), desc="Saving"):
        input_ids = result["input_ids"]
        # speculators returns hidden_states as a list of tensors ordered by layer_ids
        hidden_states_list = result["hidden_states"]

        # Last element = output hidden states (last captured layer)
        output_hidden_states = hidden_states_list[-1].cpu()

        # All but the last = aux layers, concatenated along the hidden dim
        if len(hidden_states_list) > 1:
            aux_hidden_states = torch.cat([h.cpu() for h in hidden_states_list[:-1]], dim=-1)
        else:
            aux_hidden_states = torch.empty(0)

        output_file = output_dir / f"{conv_id}.pt"
        with open(output_file, "wb") as f:
            torch.save(
                {
                    "input_ids": input_ids.cpu() if hasattr(input_ids, "cpu") else input_ids,
                    "hidden_states": output_hidden_states,
                    "aux_hidden_states": aux_hidden_states,
                    "conversation_id": conv_id,
                },
                f,
            )
        num_success += 1

    print(f"Successfully processed {num_success} out of {len(prompts)} conversations.")


if __name__ == "__main__":
    cli_args = parse_args()
    main(cli_args)
