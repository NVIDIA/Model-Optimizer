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

# ruff: noqa: D101, D102, D103, D107, PLR1722
import argparse
import json
import os
import re

from datasets import load_dataset
from openai import BadRequestError, OpenAI

early_termination = False


def _strip_thinking(content: str) -> str:
    """Strip <think>...</think> blocks from assistant message content.

    Used to clean intermediate assistant turns before they are appended to the
    context for the next generation step.  Only the final assistant turn in a
    multi-turn conversation should retain the full reasoning trace.
    """
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()


class LLM:
    def __init__(self, args):
        self.args = args
        self._pid = os.getpid()
        self.client = OpenAI(base_url=args.base_url)
        # Exercise the selected no-thinking control path during server warmup.
        if args.thinking_control == "chat-template-kwargs":
            self.generate(
                messages=[{"role": "user", "content": "Hello!"}],
                verbose=True,
                enable_thinking=False,
            )
        else:
            self.generate(
                messages=[{"role": "user", "content": "Hello! /no_think"}], verbose=True
            )

    def _ensure_client(self):
        """Reinitialize the HTTP client if we've been forked into a new process.

        datasets.map(num_proc>1) forks worker processes that inherit the parent's
        connection pool.  Reusing inherited sockets across processes causes
        "Invalid HTTP request" errors.  Creating a fresh client per-process avoids this.
        """
        if os.getpid() != self._pid:
            self._pid = os.getpid()
            self.client = OpenAI(base_url=self.args.base_url)

    def generate(
        self,
        messages,
        verbose=False,
        sample_id="<unknown>",
        sampling_params=None,
        **chat_template_kwargs,
    ):
        global early_termination
        self._ensure_client()
        try:
            sampling_params = sampling_params or {}
            arg_temperature = self.args.temperature if self.args.temperature is not None else 0.0
            chat_template_kwargs = {
                key: value for key, value in chat_template_kwargs.items() if value is not None
            }
            # vLLM exposes top_k and chat-template controls as OpenAI API extensions.
            request_kwargs = {}
            extra_body = {}
            if chat_template_kwargs:
                extra_body["chat_template_kwargs"] = chat_template_kwargs
            if "top_k" in sampling_params:
                extra_body["top_k"] = sampling_params["top_k"]
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            completion = self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                temperature=sampling_params.get("temperature", arg_temperature),
                top_p=sampling_params.get("top_p", 1.0),
                presence_penalty=sampling_params.get("presence_penalty", 0.0),
                max_tokens=self.args.max_tokens,
                **request_kwargs,
            )
            new_message = completion.choices[0].message.content
            if verbose:
                for msg in messages:
                    print("[OLD] {:10}: {:64}".format(msg["role"], msg["content"]))
                print("[NEW] {:10}: {:64}\n\n".format("assistant", new_message))

            new_message = {"role": "assistant", "content": new_message}
        except BadRequestError as e:
            # Skip overlength rows; all other request errors must fail the shard.
            if e.param == "input_tokens":
                print(f"[SKIP] {sample_id}: {e}")
                return None
            print(e)
            raise RuntimeError(str(e)) from None
        except Exception as e:
            print(e)
            if "Connection error" in str(e):
                early_termination = True
            raise RuntimeError(str(e)) from None

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
parser.add_argument("--shard-id", type=int, default=None, help="single shard id to process.")
parser.add_argument("--shard-id-begin", type=int, default=0, help="the shard id to start.")
parser.add_argument(
    "--shard-id-step", type=int, default=1, help="the step that the shard id progress."
)
parser.add_argument(
    "--num-samples", "--num_samples", type=int, default=None, help="maximum samples to process."
)
parser.add_argument("--num-proc", type=int, default=32, help="number of processes (concurrency).")
parser.add_argument("--temperature", type=float, default=None, help="temperature (default: 0).")
parser.add_argument("--sampling-params", type=json.loads, default=None)
parser.add_argument("--non-thinking-sampling-params", type=json.loads, default=None)
parser.add_argument(
    "--max-tokens", type=int, default=None, help="maximum tokens to generate per response."
)
parser.add_argument(
    "--max-total-length",
    type=int,
    default=8192,
    help="maximum total length (prompt + output). Stops synthesizing remaining turns "
    "when context exceeds this limit.",
)
parser.add_argument(
    "--thinking-control",
    choices=["soft-switch", "chat-template-kwargs"],
    default="soft-switch",
    help="Disable thinking with /no_think or the server's chat_template_kwargs API.",
)
args = parser.parse_args()

if args.temperature is not None and any(
    params is not None and "temperature" in params
    for params in (args.sampling_params, args.non_thinking_sampling_params)
):
    parser.error("--temperature cannot be combined with temperature in a sampling profile")

llm = LLM(args)

if args.data is None:
    exit(0)


def disable_thinking_column(data):
    data.update({"enable_thinking": False})
    return data


def synthesize(data):
    messages = data.get("messages") or data.get("conversations")
    if messages is None:
        raise ValueError(
            "No conversations or messages in the data. Only OAI chat data is supported."
        )
    sample_id = data.get("uuid") or data.get("conversation_id") or "<unknown>"

    # Handle generation specific kwargs.
    enable_thinking = data.get("enable_thinking", True)

    current_messages = []
    last_full_message = None  # tracks the most recent generated response (unstripped)
    max_total = args.max_total_length

    for msg in messages:
        role = msg["role"]
        if role == "system":
            current_messages.append(msg)
        elif role == "user":
            if not enable_thinking and args.thinking_control == "soft-switch":
                # Copy to avoid mutating the original dataset row.
                msg = dict(msg)
                msg["content"] = msg["content"] + " /no_think"

            current_messages.append(msg)

            # Estimate context length; stop if remaining budget is too small.
            if max_total is not None and args.max_tokens is not None:
                ctx_chars = sum(len(m.get("content", "")) for m in current_messages)
                est_tokens = ctx_chars // 3  # rough char-to-token estimate
                if est_tokens + args.max_tokens > max_total:
                    # Drop this user turn — context too long for another generation
                    print(f"[SKIP] {sample_id}: estimated context exceeds {max_total} tokens")
                    current_messages.pop()
                    break

            # Thinking and non-thinking rows use their own sampling profiles.
            new_message = llm.generate(
                current_messages,
                verbose=False,
                sample_id=sample_id,
                sampling_params=(
                    args.non_thinking_sampling_params
                    if not enable_thinking and args.non_thinking_sampling_params is not None
                    else args.sampling_params
                ),
                enable_thinking=(
                    enable_thinking if args.thinking_control == "chat-template-kwargs" else None
                ),
            )
            if new_message is None:
                current_messages.pop()
                break

            # Preserve the mode so the training template can handle unfinished thinking.
            new_message["enable_thinking"] = enable_thinking
            last_full_message = new_message

            if enable_thinking:
                # Append a thinking-stripped copy as context for the next turn.
                # Multi-turn reasoning: only the *last* assistant turn should
                # retain the full <think>...</think> trace; prior turns are
                # already resolved and the trace would distract the model.
                # The full trace is restored to the last turn after the loop.
                stripped = {
                    "role": "assistant",
                    "content": _strip_thinking(new_message["content"]),
                }
                current_messages.append(stripped)
            else:
                current_messages.append(new_message)
        elif role == "developer":
            # Map developer-role messages to system per OpenAI schema conventions.
            current_messages.append({"role": "system", "content": msg["content"]})
        elif role == "assistant":
            # Original assistant messages are not used — the model generates fresh responses.
            pass
        elif role == "tool":
            # Tool turns are not sent to the generation model — skip them.
            pass
        else:
            raise ValueError(f"Unexpected message role {role!r} in conversation.")

    # Restore the full reasoning trace for the last generated assistant turn.
    if enable_thinking and last_full_message is not None:
        for i in range(len(current_messages) - 1, -1, -1):
            if current_messages[i]["role"] == "assistant":
                current_messages[i] = last_full_message
                break

    # Preserve the dataset schema for failed rows, then filter them after mapping.
    synthesis_ok = last_full_message is not None
    output_messages = [dict(msg) for msg in (current_messages if synthesis_ok else messages)]
    for msg in output_messages:
        msg.setdefault("enable_thinking", enable_thinking)
    return {"messages": output_messages, "_synthesis_ok": synthesis_ok}


# Support both HF Hub repo IDs and local file paths (.jsonl, .json, .parquet, etc.)
if os.path.isfile(args.data):
    ext = os.path.splitext(args.data)[1].lower()
    fmt = "parquet" if ext == ".parquet" else "json"
    dataset = load_dataset(fmt, data_files={"train": args.data}, split=args.data_split)
else:
    dataset = load_dataset(args.data, split=args.data_split)

if args.shard_id is None and args.num_shards * 100 > len(dataset):
    args.num_shards = max(1, min(16, len(dataset) // 100))

# Apply --num-samples globally BEFORE sharding so the cap bounds total output,
# not per-shard output (coderabbit:query.py:241).
if args.num_samples is not None:
    dataset = dataset.select(range(min(args.num_samples, len(dataset))))

# Validate --shard-id once at the interface boundary (coderabbit:query.py:225).
# dataset.shard(index=...) raises a confusing ValueError on out-of-range ids;
# fail loud with a clear message instead.
if args.shard_id is not None and not (0 <= args.shard_id < args.num_shards):
    parser.error(f"--shard-id {args.shard_id} out of range [0, {args.num_shards})")

if args.save is not None:
    print(f"Create save dir: {args.save}")
    os.makedirs(args.save, exist_ok=True)

shard_ids = (
    [args.shard_id]
    if args.shard_id is not None
    else range(args.shard_id_begin, args.num_shards, args.shard_id_step)
)

for shard_id in shard_ids:
    if args.shard_id is None:
        file_path = args.save + f"/train-{shard_id + 1:05}-{args.num_shards:05}.jsonl"
        done_path = f"{file_path}.done"
    else:
        file_path = args.save + f"/shard_{shard_id}.jsonl"
        done_path = args.save + f"/shard_{shard_id}.done"

    if os.path.exists(file_path) and os.path.exists(done_path):
        continue

    shard = dataset.shard(num_shards=args.num_shards, index=shard_id)
    print(len(shard), file_path)

    num_proc = min(args.num_proc, len(shard))
    if shard_id % 2 == 0:
        shard = shard.map(disable_thinking_column, num_proc=num_proc)
    # Reuse completed map-worker caches and omit rows without a generated response.
    cache_dir = os.path.join(args.save, ".cache")
    os.makedirs(cache_dir, exist_ok=True)
    updated_shard = shard.map(
        synthesize,
        num_proc=num_proc,
        cache_file_name=os.path.join(cache_dir, f"shard_{shard_id}.arrow"),
    )
    updated_shard = updated_shard.filter(lambda row: row["_synthesis_ok"])
    updated_shard = updated_shard.remove_columns("_synthesis_ok")
    updated_shard.to_json(file_path)
    with open(done_path, "w") as done_file:
        done_file.write("done\n")
    print(updated_shard[0])

    if early_termination:
        print("Terminate earlier due to server connection error!")
        break
