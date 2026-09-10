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

"""Send conversations from a dataset to an OpenAI-compatible endpoint."""

import argparse
import asyncio
import json
from pathlib import Path

import httpx
import openai
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="""Collect hidden states from conversations
        by sending full conversations as prompts to an OpenAI-compatible endpoint."""
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
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="""HTTP URL for the OpenAI-compatible endpoint.
        Defaults to `http://localhost:8000/v1`.""",
    )
    parser.add_argument(
        "--openai-api-key",
        default="EMPTY",
        help="""Access key required by the OpenAI Python client
        (not required for local serving engines like vLLM).""",
    )

    ## I/O Parameters ##
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=3072,
        help="""Maximum number of tokens in a conversation. Longer conversations will be skipped.
        Defaults to 3072 tokens.""",
    )
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

    client: AsyncOpenAI = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url=args.base_url,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos_token_id = tokenizer.bos_token_id
    if bos_token_id is None:
        raise ValueError("The tokenizer does not have a BOS token.")

    temp_meta_file = Path("/tmp/meta.json")
    if temp_meta_file.exists():
        print(f"Temporary meta file {temp_meta_file} found, removing it.")
        temp_meta_file.unlink()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_invalid = 0
    num_error = 0
    num_too_long = 0
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

        hidden_states_file = output_dir / f"{conversation_id}.pt"

        # Use /tmp/meta.json to communicate with the local serving engine.
        # See usage guide for more details
        with temp_meta_file.open("w") as f:
            json.dump(
                {
                    "conversation_id": conversation_id,
                    "output_file": str(hidden_states_file),
                },
                f,
            )

        input_ids = tokenizer.apply_chat_template(
            conversations, return_tensors=None, add_generation_template=False, tokenize=True
        )
        num_input_tokens = len(input_ids)
        if num_input_tokens <= 10 or num_input_tokens > args.max_seq_len:
            num_too_long += 1
            continue
        if input_ids[0] == bos_token_id:
            # Remove the leading BOS token. vLLM's completion generation
            # endpoint will prepend the BOS token automatically.
            input_ids = input_ids[1:]
        input_string = tokenizer.decode(input_ids, skip_special_tokens=False)

        try:
            # Send the message to the OpenAI-compatible endpoint
            await client.completions.create(
                model=args.model,
                prompt=input_string,
                temperature=0.0,
                max_tokens=1,
            )
        except httpx.HTTPStatusError as e:
            print(f"HTTP error for conversation {conversation_id}: {e}")
            num_error += 1
            continue
        except openai.BadRequestError:
            # Most likely the conversation is too long, ignore
            num_too_long += 1
            continue
        except Exception as e:
            num_error += 1
            print(f"Error sending conversation {conversation_id}: {e}")
            continue
        finally:
            # Ensure the meta file is cleaned up after each request
            if temp_meta_file.exists():
                temp_meta_file.unlink()
        num_success += 1
        continue

    if num_invalid > 0:
        print(f"Skipped {num_invalid} invalid conversations without proper fields.")
    if num_too_long > 0:
        print(f"Skipped {num_too_long} conversations likely due to length constraints.")
    if num_error > 0:
        print(f"Encountered errors for {num_error} conversations.")

    if num_success == num_total_conversations:
        print(f"Successfully processed all {num_success} conversations.")
    else:
        print(
            f"Successfully processed {num_success} out of {num_total_conversations} conversations."
        )


if __name__ == "__main__":
    cli_args = parse_args()
    asyncio.run(main(cli_args))
