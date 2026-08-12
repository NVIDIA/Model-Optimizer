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

"""Regression test for get_conversation_input_ids, the shared offline-dump tokenizer helper.

On transformers>=5 apply_chat_template returns a BatchEncoding, so the old len(input_ids)
was 2 (field count) and every conversation got dropped by the num_input_tokens <= 10 filter.
"""

from collections.abc import Mapping

from _test_utils.torch.transformers_models import get_tiny_tokenizer

from modelopt.torch.speculative.utils import get_conversation_input_ids


def test_get_conversation_input_ids_returns_token_sequence():
    tokenizer = get_tiny_tokenizer()
    conversations = [
        {"role": "user", "content": "Explain why the sky is blue in a few sentences."},
        {
            "role": "assistant",
            "content": "Rayleigh scattering makes shorter blue wavelengths scatter more. " * 8,
        },
    ]

    input_ids = get_conversation_input_ids(tokenizer, conversations)

    # A token-id list, not a BatchEncoding (whose len() is 2).
    assert not isinstance(input_ids, Mapping)

    # Values must match the rendered prompt's ids, not just the length.
    actual_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
    rendered = tokenizer.apply_chat_template(
        conversations, add_generation_prompt=False, tokenize=False
    )
    expected_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    assert actual_ids == expected_ids
    assert len(actual_ids) > 10
