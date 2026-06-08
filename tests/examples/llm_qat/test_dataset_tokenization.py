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

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples" / "llm_qat"))

from dataset_utils import IGNORE_TOKEN_ID, DatasetSourceConfig, make_chat_tokenize_fn

CONVERSATION = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
]


class FakeChatTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    unk_token_id = 999
    vocab_size = 1024

    def __init__(
        self,
        *,
        chat_template: str = "plain template",
        chatml: bool = False,
        input_ids: list[int] | None = None,
        attention_mask: list[int] | None = None,
        assistant_masks: list[int] | None = None,
        name_or_path: str = "custom-model",
    ):
        self.chat_template = chat_template
        self.chatml = chatml
        self.input_ids = input_ids or [11, 22, 33, self.pad_token_id]
        self.attention_mask = attention_mask or [1, 1, 1, 0]
        self.assistant_masks = assistant_masks or [0, 1, 0, 0]
        self.name_or_path = name_or_path
        self.return_assistant_tokens_mask_calls: list[bool] = []

    def convert_tokens_to_ids(self, token: str) -> int:
        if not self.chatml:
            return self.unk_token_id
        return {"<|im_start|>": 100, "<|im_end|>": 101}.get(token, self.unk_token_id)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return {"assistant": [10], "\n": [11]}.get(text, [88])

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        return_dict: bool,
        return_assistant_tokens_mask: bool,
        padding: str,
        truncation: bool,
        max_length: int,
    ) -> dict[str, list[int]]:
        del messages, tokenize, return_dict, padding, truncation, max_length
        self.return_assistant_tokens_mask_calls.append(return_assistant_tokens_mask)
        result = {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
        }
        if return_assistant_tokens_mask:
            result["assistant_masks"] = list(self.assistant_masks)
        return result


def test_train_only_assistant_tokens_auto_uses_native_generation_tags():
    tokenizer = FakeChatTokenizer(chat_template="{% generation %}{{ content }}{% endgeneration %}")

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=4)({"messages": CONVERSATION})

    assert tokenizer.return_assistant_tokens_mask_calls == [True]
    assert tokenized["labels"] == [IGNORE_TOKEN_ID, 22, IGNORE_TOKEN_ID, IGNORE_TOKEN_ID]


def test_train_only_assistant_tokens_auto_uses_chatml_heuristic():
    tokenizer = FakeChatTokenizer(
        chatml=True,
        input_ids=[100, 10, 11, 42, 101, 0],
        attention_mask=[1, 1, 1, 1, 1, 0],
        name_or_path="Qwen/test-tokenizer",
    )

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=6)({"messages": CONVERSATION})

    assert tokenizer.return_assistant_tokens_mask_calls == [False]
    assert tokenized["labels"] == [
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
        42,
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
    ]


def test_train_only_assistant_tokens_auto_falls_back_for_untested_chatml():
    tokenizer = FakeChatTokenizer(
        chatml=True,
        input_ids=[100, 10, 11, 42, 101, 0],
        attention_mask=[1, 1, 1, 1, 1, 0],
    )

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=6)({"messages": CONVERSATION})

    assert tokenizer.return_assistant_tokens_mask_calls == [False]
    assert tokenized["labels"] == [100, 10, 11, 42, 101, IGNORE_TOKEN_ID]


def test_train_only_assistant_tokens_auto_falls_back_to_full_chat_labels():
    tokenizer = FakeChatTokenizer()

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=4)({"messages": CONVERSATION})

    assert tokenizer.return_assistant_tokens_mask_calls == [False]
    assert tokenized["labels"] == [11, 22, 33, IGNORE_TOKEN_ID]


def test_train_only_assistant_tokens_true_requires_supported_masking():
    tokenizer = FakeChatTokenizer()

    with pytest.raises(ValueError, match="train_only_assistant_tokens: false"):
        make_chat_tokenize_fn(tokenizer, max_length=4, train_only_assistant_tokens=True)

    assert tokenizer.return_assistant_tokens_mask_calls == []


def test_train_only_assistant_tokens_true_allows_explicit_chatml_heuristic():
    tokenizer = FakeChatTokenizer(
        chatml=True,
        input_ids=[100, 10, 11, 42, 101, 0],
        attention_mask=[1, 1, 1, 1, 1, 0],
    )

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=6, train_only_assistant_tokens=True)(
        {"messages": CONVERSATION}
    )

    assert tokenizer.return_assistant_tokens_mask_calls == [False]
    assert tokenized["labels"] == [
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
        42,
        IGNORE_TOKEN_ID,
        IGNORE_TOKEN_ID,
    ]


def test_train_only_assistant_tokens_false_uses_full_chat_labels():
    tokenizer = FakeChatTokenizer(chat_template="{% generation %}{{ content }}{% endgeneration %}")

    tokenized = make_chat_tokenize_fn(tokenizer, max_length=4, train_only_assistant_tokens=False)(
        {"messages": CONVERSATION}
    )

    assert tokenizer.return_assistant_tokens_mask_calls == [False]
    assert tokenized["labels"] == [11, 22, 33, IGNORE_TOKEN_ID]


def test_dataset_source_config_normalizes_train_only_assistant_tokens():
    source = DatasetSourceConfig(
        hf_path="dataset",
        ratio=1,
        split="train",
        train_only_assistant_tokens="false",
    )

    assert source.train_only_assistant_tokens is False
    with pytest.raises(ValueError, match="train_only_assistant_tokens"):
        DatasetSourceConfig(
            hf_path="dataset",
            ratio=1,
            split="train",
            train_only_assistant_tokens="unsupported",
        )
