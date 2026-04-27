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

"""Unit test for ChatML heuristic assistant masking using real Qwen3 tokenizer."""

import sys
from pathlib import Path

import pytest
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "llm_qat"))

from dataset_utils import _chatml_assistant_mask, _supports_chatml_heuristic

CONVERSATION = [
    {"role": "user", "content": "Hello assistant"},
    {"role": "assistant", "content": "Hello user"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm good"},
]


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def test_chatml_assistant_mask(tokenizer):
    assert _supports_chatml_heuristic(tokenizer)

    result = tokenizer.apply_chat_template(
        CONVERSATION,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    input_ids = result["input_ids"]
    heuristic = _chatml_assistant_mask(input_ids, tokenizer)

    assert len(heuristic) == len(input_ids)
    assert sum(heuristic) > 0, "heuristic should mask some assistant tokens"

    tokens = [tokenizer.decode([tid]) for tid in input_ids]
    for tok, m in zip(tokens, heuristic):
        if m == 1:
            assert tok not in ("<|im_start|>", "user", "system"), (
                f"non-content token {tok!r} should not be masked"
            )
