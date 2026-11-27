# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare correct and looping responses, preserving each response's own prompt."""

from typing import Any

import torch


def preference_processor(datum_dict, task_data_spec, tokenizer, max_seq_length, idx):
    """Prepare a correct response and a looping response for NeMo RL.

    The row stores the correct response in chosen_serialized and the looping
    response in rejected_serialized. Optional chosen_prompt_serialized and
    rejected_prompt_serialized fields preserve each response's own prompt;
    prompt_serialized is the required shared fallback.

    Tokenize the prompt and response separately without adding special tokens
    automatically. Append EOS (end of response) to completed correct answers so
    KTO learns to produce the answer and stop. Looping examples are unfinished
    prefixes; adding EOS would make stopping part of the undesirable response.

    The model reads the prompt as context, but the KTO loss scores only response
    tokens. Store the prompt as a "user" message and the response as an
    "assistant" message so NeMo RL's collator can identify the tokens to score.

    Args:
        datum_dict: Prepared JSONL row containing the prompts and responses above.
            An optional task_name field supplies a task label; defaults to "kto".
        task_data_spec: Unused; accepted for NeMo RL's processor interface.
        tokenizer: Tokenizer providing encode() and eos_token_id.
        max_seq_length: Maximum prompt-plus-response length, including added EOS.
        idx: Row index passed through to the returned data.

    Returns:
        Chosen/rejected tokenized message logs, their lengths, and loss/task metadata.

    Raises:
        AssertionError: A prompt and response exceed max_seq_length.
        ValueError: A response is blank or produces no tokens.
    """
    output: dict[str, Any] = {}
    # NeMo RL's chosen/rejected fields carry independent labels; prompts need not match.
    for branch in ["chosen", "rejected"]:
        prompt = datum_dict.get(branch + "_prompt_serialized", datum_dict["prompt_serialized"])
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        response_field = branch + "_serialized"
        response = datum_dict[response_field]
        ids = tokenizer.encode(response, add_special_tokens=False)
        # Reject blank responses and responses with no tokens before adding EOS.
        # KTO needs response content to learn from, even when a correct answer gets EOS.
        if not response.strip() or not ids:
            raise ValueError(f"Row {idx} has an empty {response_field}")
        if branch == "chosen":
            ids.append(tokenizer.eos_token_id)
        assert len(prefix) + len(ids) <= max_seq_length, "Response exceeds training sequence limit"
        output["message_log_" + branch] = [
            {
                "role": "user",
                "content": prompt,
                "token_ids": torch.tensor(prefix, dtype=torch.long),
            },
            {
                "role": "assistant",
                "content": response,
                "token_ids": torch.tensor(ids, dtype=torch.long),
            },
        ]
        output["length_" + branch] = len(prefix) + len(ids)
    # Keep an optional task label from the input row; it does not change the KTO loss.
    output.update(loss_multiplier=1.0, idx=idx, task_name=datum_dict.get("task_name", "kto"))
    return output
