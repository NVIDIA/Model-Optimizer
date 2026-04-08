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
from openai import OpenAI

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, help="Path to the data file")
parser.add_argument("--output_path", type=str, help="Path to the output file")
parser.add_argument(
    "--num_threads", type=int, default=256, help="Number of threads to use (batch size)"
)
parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model")
parser.add_argument(
    "--max_tokens", type=int, default=2048, help="Maximum number of tokens to generate"
)
parser.add_argument("--chat", default=True, type=bool, help="Use chat mode")
parser.add_argument("--model", type=str, default="model", help="Model name")
parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="URL of the API")
parser.add_argument("--api_key", type=str, default="token-abc123", help="API key (if any)")
parser.add_argument(
    "--log_empty_conversations", action="store_true", help="Log empty conversations"
)
parser.add_argument("--system_prompt", nargs="+", type=str, default="", help="System prompt")
args = parser.parse_args()


if args.data_path.endswith("jsonl"):
    with open(args.data_path) as f:
        data = [json.loads(line) for line in f]
else:
    data = json.load(open(args.data_path))

client = OpenAI(
    base_url=args.url,
    api_key=args.api_key,
)


def generate_data(messages, idx, system_prompt):
    try:
        model_name = args.model

        if args.chat:
            output_messages = []

            if system_prompt and len(messages) > 0:
                system_message = {"role": "system", "content": system_prompt}
                output_messages.append(system_message)

            for message in messages[::2]:
                # Detect message format
                if "from" in message and "value" in message:
                    role = message["from"].lower()
                    content = message["value"]
                elif "role" in message and "content" in message:
                    role = message["role"].lower()
                    content = message["content"]
                else:
                    raise ValueError(f"Message format not recognized: {message}")

                if role not in ["user", "human"]:
                    return
                output_messages.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=output_messages,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    if response.choices[0].finish_reason == "length":
                        break
                    response = response.choices[0].message.content.strip()
                    output_messages.append(
                        {
                            "role": "assistant",
                            "content": response,
                        }
                    )
                except Exception as e:
                    print(e)
                    break
            if len(output_messages) == 1 or (system_prompt and len(output_messages) == 2):
                if not args.log_empty_conversations:
                    return
                to_write = {"conversation_id": idx}
            else:
                to_write = {"conversation_id": idx, "conversations": output_messages}
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
        print(e)
        print(prompt)
        print("Failed to generate data")


# if output_path exists identify the conversation_ids that have already been generated
finished_ids = []
done = False
if os.path.exists(args.output_path):
    with open(args.output_path) as f:
        for line in f:
            outdata = json.loads(line)
            finished_ids.append(outdata.get("conversation_id", -1))
            if outdata.get("finished", False):
                done = True
                break
finished_ids = set(finished_ids)

# Ensure the output directory exists before writing to the output file
output_dir = os.path.dirname(args.output_path)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

if done:
    print("All conversations already generated")
    sys.exit()

with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
    futures = []
    system_prompt = " ".join(args.system_prompt)

    for idx, sample in enumerate(data):
        if idx in finished_ids:
            continue
        future = executor.submit(generate_data, sample["conversations"], idx, system_prompt)
        futures.append(future)

    for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
        future.result()

if args.log_empty_conversations:
    with open(args.output_path, "a") as f:
        f.write(json.dumps({"finished": True}) + "\n")
