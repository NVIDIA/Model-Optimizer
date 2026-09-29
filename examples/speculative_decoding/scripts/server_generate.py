# Adapted from: https://github.com/FasterDecoding/Medusa/blob/e2a5d20/data_generation/generate.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText:Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import concurrent.futures
import json
import os
import sys

import tqdm
from openai import APIConnectionError, APIStatusError, OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the data file")
parser.add_argument("--output_path", type=str, help="Path to the output file")
parser.add_argument(
    "--num_threads", type=int, default=256, help="Number of threads to use (batch size)"
)
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
parser.add_argument(
    "--max_tokens",
    type=int,
    default=2048,
    help="Maximum generated tokens; 0 lets the server determine the remaining context budget",
)
parser.add_argument("--request_timeout", type=float, default=600, help="API timeout in seconds")
parser.add_argument("--chat", default=True, type=bool, help="Use chat mode")
parser.add_argument("--model", type=str, default="model", help="Model name")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="URL of the API")
parser.add_argument("--api_key", type=str, default="token-abc123", help="API key (if any)")
parser.add_argument(
    "--log_empty_conversations", action="store_true", help="Log empty conversations"
)
parser.add_argument("--system_prompt", nargs="+", type=str, default="", help="System prompt")
parser.add_argument(
    "--extra_body", type=json.loads, help="JSON object of additional chat request parameters"
)
parser.add_argument(
    "--fail_on_error",
    action="store_true",
    help="Exit nonzero after processing a batch with failures",
)
parser.add_argument(
    "--retry_failed", action="store_true", help="Retry previously rejected conversations on resume"
)
args = parser.parse_args()
if args.extra_body is not None and not isinstance(args.extra_body, dict):
    parser.error("--extra_body must be a JSON object")
if args.max_tokens < 0:
    parser.error("--max_tokens must be nonnegative")


if args.data_path.endswith("jsonl"):
    with open(args.data_path) as f:
        data = [json.loads(line) for line in f]
else:
    data = json.load(open(args.data_path))

client = OpenAI(
    base_url=args.url,
    api_key=args.api_key,
    timeout=args.request_timeout,
)


class RejectedConversationError(ValueError):
    """A conversation cannot be generated with the current input and request settings."""


class RetryableConversationError(RuntimeError):
    """A generation failure may succeed when the conversation is retried on resume."""


def generate_data(sample, idx, system_prompt):
    """Generate a complete conversation, retaining reasoning and marking truncated responses."""
    try:
        if not isinstance(sample, dict):
            raise RejectedConversationError("Expected a conversation object.")
        messages = sample.get("conversations", sample.get("messages"))
        if not isinstance(messages, list):
            raise RejectedConversationError("Expected a conversations or messages list.")
        model_name = args.model

        if args.chat:
            output_messages = []
            truncated = False

            if system_prompt and len(messages) > 0:
                system_message = {"role": "system", "content": system_prompt}
                output_messages.append(system_message)

            for message in messages:
                # Detect message format
                if not isinstance(message, dict):
                    raise RejectedConversationError("Expected a message object.")
                if "from" in message and "value" in message:
                    role = message["from"]
                    content = message["value"]
                elif "role" in message and "content" in message:
                    role = message["role"]
                    content = message["content"]
                else:
                    raise RejectedConversationError("Message format not recognized.")
                if not isinstance(role, str):
                    raise RejectedConversationError("Expected a string message role.")
                role = role.lower()

                if message.get("tool_calls") or message.get("function_call"):
                    raise RejectedConversationError("Tool calls require a tool-execution loop.")
                if role == "system":
                    if not system_prompt:
                        output_messages.append({"role": "system", "content": content})
                    else:
                        print(
                            f"Warning: conversation {idx}: --system_prompt overrides the input system message.",
                            file=sys.stderr,
                        )
                    continue
                if role in ["assistant", "gpt"]:
                    continue
                if role not in ["user", "human"]:
                    raise RejectedConversationError(f"Unsupported message role: {role}")
                output_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
                response = client.chat.completions.create(
                    model=model_name,
                    messages=output_messages,
                    max_tokens=args.max_tokens or None,
                    temperature=args.temperature,
                    extra_body=args.extra_body,
                )
                choice = response.choices[0]
                if choice.message.tool_calls or choice.message.function_call:
                    raise RejectedConversationError("Tool calls require a tool-execution loop.")
                generated_message = {
                    "role": "assistant",
                    "content": (choice.message.content or "").strip(),
                }
                reasoning = getattr(choice.message, "reasoning_content", None) or getattr(
                    choice.message, "reasoning", None
                )
                if reasoning:
                    generated_message["reasoning_content"] = reasoning
                truncated = (
                    choice.finish_reason in ("length", "repetition")
                    or getattr(choice, "stop_reason", None) == "repetition_detected"
                )
                if not generated_message["content"] and not truncated:
                    error_type = (
                        RetryableConversationError
                        if args.temperature > 0
                        else RejectedConversationError
                    )
                    raise error_type(
                        f"Model returned an empty final answer (finish_reason={choice.finish_reason}, "
                        f"reasoning_characters={len(reasoning or '')})."
                    )
                output_messages.append(generated_message)
                if truncated:
                    break
            if not any(message["role"] == "assistant" for message in output_messages):
                if not args.log_empty_conversations:
                    return
                to_write = {"conversation_id": idx}
            else:
                to_write = {"conversation_id": idx, "conversations": output_messages}
            if truncated:
                to_write["truncated"] = True
                to_write["finish_reason"] = choice.finish_reason
                to_write["stop_reason"] = getattr(choice, "stop_reason", None)
            with open(args.output_path, "a") as f:
                # write in share gpt format
                f.write(json.dumps(to_write) + "\n")
        else:
            from fastchat.model.model_adapter import get_conversation_template

            conv = get_conversation_template(model_name)
            conv.append_message(conv.roles[0], messages[0]["value"])
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            response = client.chat.completions.create(
                model=model_name,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                ignore_eos=False,
                skip_special_tokens=False,
                spaces_between_special_tokens=False,
            )
            response = response.choices[0].text.strip()
            with open(args.output_path, "a") as f:
                # write in share gpt format
                if args.log_empty_conversations:
                    to_write = {"conversation_id": idx, "text": prompt + response}
                else:
                    to_write = {"text": prompt + response}
                f.write(json.dumps(to_write) + "\n")
    except Exception as e:
        raise RuntimeError(f"Failed to generate conversation {idx}") from e


# if output_path exists identify the conversation_ids that have already been generated
finished_ids = set()
done = False
if os.path.exists(args.output_path):
    with open(args.output_path) as f:
        for line in f:
            outdata = json.loads(line)
            if "conversation_id" in outdata:
                finished_ids.add(outdata["conversation_id"])
            done = outdata.get("finished", False)

# Keep the JSONL failure journal outside the shard combiner's *.jsonl input set.
failures_path = args.output_path + ".failures"
failures = {}
if os.path.exists(failures_path):
    with open(failures_path) as f:
        for line in f:
            failure = json.loads(line)
            if failure["conversation_id"] not in finished_ids:
                failures[failure["conversation_id"]] = failure
rejected_ids = {idx for idx, failure in failures.items() if not failure["retryable"]}
if failures:
    print(f"Found {len(failures)} unresolved failures in {failures_path}.", file=sys.stderr)

# Ensure the output directory exists before writing to the output file
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

if (
    done
    and not (args.retry_failed and rejected_ids)
    and not any(failure["retryable"] for failure in failures.values())
):
    print("All conversations already processed")
    sys.exit(1 if args.fail_on_error and failures else 0)

fatal_error = False
with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    futures = {}
    system_prompt = " ".join(args.system_prompt)

    for idx, sample in enumerate(data):
        if idx in finished_ids or (idx in rejected_ids and not args.retry_failed):
            continue
        future = executor.submit(generate_data, sample, idx, system_prompt)
        futures[future] = idx

    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        idx = futures[future]
        try:
            future.result()
        except Exception as exc:
            cause = exc.__cause__ or exc
            status = cause.status_code if isinstance(cause, APIStatusError) else None
            rejected = isinstance(cause, RejectedConversationError) or status in (400, 422)
            transient = isinstance(cause, (APIConnectionError, RetryableConversationError)) or (
                status is not None and (status in (408, 409, 429) or status >= 500)
            )
            fatal_error |= not (rejected or transient)
            failure = {
                "conversation_id": idx,
                "retryable": not rejected,
                "error_type": type(cause).__name__,
                "error": str(cause),
            }
            failures[idx] = failure
            with open(failures_path, "a") as f:
                f.write(json.dumps(failure) + "\n")
            print(f"Failed conversation {idx}: {cause}", file=sys.stderr)
        else:
            failures.pop(idx, None)

if failures:
    retryable = sum(failure["retryable"] for failure in failures.values())
    print(
        f"{len(failures)} conversations failed ({retryable} retryable); see {failures_path}.",
        file=sys.stderr,
    )

if args.log_empty_conversations and not any(failure["retryable"] for failure in failures.values()):
    with open(args.output_path, "a") as f:
        f.write(json.dumps({"finished": True}) + "\n")

if fatal_error or (args.fail_on_error and failures):
    sys.exit(1)
