# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json

from transformers import AutoTokenizer


def get_tokenizer(path, trust_remote_code=False):
    return AutoTokenizer.from_pretrained(path, trust_remote_code=trust_remote_code)


def encode_chat(tokenizer, messages, chat_template_args={}, completions=False):
    if completions:
        return tokenizer.encode(messages[-1]["content"], add_special_tokens=False)
    return tokenizer.encode(
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **chat_template_args
        ),
        add_special_tokens=False,
    )


def decode_chat(tokenizer, out_tokens):
    return tokenizer.decode(out_tokens)


def read_json(path):
    if path is not None:
        with open(path) as f:
            data = json.load(f)
        return data
    return {}


def postprocess_base(text):
    return text


def postprocess_gptoss(text):
    final_message = text.split("<|channel|>final<|message|>")[-1]
    if "<|end|>" in final_message:
        final_message = final_message.split("<|end|>")[0]
    if "<|return|>" in final_message:
        final_message = final_message.split("<|return|>")[0]
    if "<|channel|>" in final_message:
        final_message = final_message.split("<|channel|>")[0]
    return final_message
