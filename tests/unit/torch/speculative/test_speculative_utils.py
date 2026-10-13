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

"""Tests for get_conversation_input_ids, the shared offline-dump tokenizer helper.

apply_chat_template returns a BatchEncoding on transformers>=5, so the old len(input_ids)
was 2 (field count) and every conversation got dropped by the num_input_tokens <= 10 filter.
"""

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from transformers import BatchEncoding

from modelopt.torch.speculative.utils import get_conversation_input_ids

CONVERSATIONS = [
    {"role": "user", "content": "Explain why the sky is blue in a few sentences."},
    {
        "role": "assistant",
        "content": "Rayleigh scattering makes shorter blue wavelengths scatter more. " * 8,
    },
]


def _expected_ids(tokenizer):
    rendered = tokenizer.apply_chat_template(
        CONVERSATIONS, add_generation_prompt=False, tokenize=False
    )
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def test_matches_rendered_prompt():
    tokenizer = get_tiny_tokenizer()
    input_ids = get_conversation_input_ids(tokenizer, CONVERSATIONS)
    assert input_ids == _expected_ids(tokenizer)
    assert len(input_ids) > 10


@pytest.mark.parametrize(
    "wrap", ["batch_encoding", "plain_dict", "tensor_2d", "batched_list", "plain_list"]
)
def test_normalizes_to_flat_list(wrap):
    """Pin every apply_chat_template return shape to a flat list[int], regardless of version."""
    expected = _expected_ids(get_tiny_tokenizer())
    returns = {
        "batch_encoding": BatchEncoding(
            {"input_ids": expected, "attention_mask": [1] * len(expected)}
        ),
        "plain_dict": {"input_ids": expected, "attention_mask": [1] * len(expected)},
        "tensor_2d": torch.tensor([expected]),
        "batched_list": [expected],
        "plain_list": expected,
    }

    class _Stub:
        def apply_chat_template(self, conversations, **kwargs):
            return returns[wrap]

    input_ids = get_conversation_input_ids(_Stub(), CONVERSATIONS)
    assert input_ids == expected
    # The pre-fix code saw len() in {1, 2} here and silently dropped the conversation.
    assert len(input_ids) > 10
