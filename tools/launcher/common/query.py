# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""OpenAI-compatible client for querying LLM inference servers.

Used by TRT-LLM and vLLM query scripts to send prompts to a running server,
collect responses, and optionally save them to disk for downstream pipelines
(e.g., EAGLE3 data synthesis).
"""

# ruff: noqa: D101, D102, D103, D107, F841, PLR1722
import argparse
import os

from datasets import load_dataset
from openai import OpenAI

early_termination = False


class LLM:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(base_url=args.base_url)
        self.generate(messages=[{"role": "user", "content": "Hello! /no_think"}], verbose=True)

    def generate(self, messages, verbose=False, **chat_template_kwargs):
        try:
            completion = self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                temperature=self.args.temperature,
            )
            new_message = completion.choices[0].message.content
            if verbose:
                for msg in messages:
                    print("[OLD] {:10}: {:64}".format(msg["role"], msg["content"]))
                print("[NEW] {:10}: {:64}\n\n".format("assistant", new_message))

            new_message = {"role": "assistant", "content": new_message}
        except Exception as e:
            print(e)

            if "Connection error" in str(e):
                early_termination = True

            new_message = None

        return new_message


parser = argparse.ArgumentParser(prog="query")
parser.add_argument("base_url", type=str, help="url to the OpenAI compatible API.")
parser.add_argument("model", type=str, help="model name")
parser.add_argument(
    "--data", type=str, default=None, help="path to OAI chat data (local or HF hub)"
)
parser.add_argument("--data-split", type=str, default="train", help="HF dataset split")
parser.add_argument("--save", type=str, default=None, help="path to store the generated output.")
parser.add_argument("--num-shards", type=int, default=1000, help="number of shards.")
parser.add_argument("--shard-id-begin", type=int, default=0, help="the shard id to start.")
parser.add_argument(
    "--shard-id-step", type=int, default=1, help="the step that the shard id progress."
)
parser.add_argument("--num-proc", type=int, default=32, help="number of processes (concurrency).")
parser.add_argument("--temperature", type=float, default=0.0, help="temperature.")
args = parser.parse_args()

llm = LLM(args)

if args.data is None:
    exit(0)


def disable_thinking_column(data):
    data.update({"enable_thinking": False})
    return data


def synthesize(data):
    messages = data.get("conversations", None)
    if messages is None:
        messages = data.get("messages", None)
    if messages is None:
        raise ValueError(
            "No conversations of messages in the data. Only OAI chat data is supported."
        )

    # Handle generation specific kwargs.
    enable_thinking = data.get("enable_thinking", True)

    current_messages = []

    for msg in messages:
        if msg["role"] == "system":
            current_messages.append(msg)
        elif msg["role"] == "user":
            if not enable_thinking:
                msg["content"] = msg["content"] + " /no_think"

            current_messages.append(msg)
            new_message = llm.generate(current_messages, verbose=False)
            if new_message is None:
                break
            else:
                current_messages.append(new_message)
        elif msg["role"] == "assistant":
            # Original assistant messages are not used
            pass
        else:
            raise ValueError("unknown role: {}".format(msg["role"]))

    return {"conversations": current_messages}


dataset = load_dataset(args.data, split=args.data_split)

if args.num_shards * 100 > len(dataset):
    args.num_shards = min(16, len(dataset) // 100)

if args.save is not None:
    print("Create save dir: {}".format(args.save))
    os.makedirs(args.save, exist_ok=True)

for shard_id in range(args.shard_id_begin, args.num_shards, args.shard_id_step):
    file_path = args.save + "/train-{:05}-{:05}.jsonl".format(shard_id + 1, args.num_shards)

    if os.path.exists(file_path):
        continue

    shard = dataset.shard(num_shards=args.num_shards, index=shard_id)
    print(len(shard), file_path)

    if shard_id % 2 == 0:
        shard = shard.map(disable_thinking_column, num_proc=args.num_proc)
    updated_shard = shard.map(synthesize, num_proc=args.num_proc)
    updated_shard.to_json(file_path)
    print(updated_shard[0])

    if early_termination:
        print("Terminate earlier due to server connection error!")
        break
