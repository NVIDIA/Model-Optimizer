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
# mypy: ignore-errors

"""
Only needed for DeciLM models that use Megatron tokenizers.
DeciLM models that use Llama tokenizers do not need external code.
"""

import json
import os
from pathlib import Path
from typing import Literal

from transformers import PreTrainedTokenizer
from transformers.dynamic_module_utils import custom_object_save
from transformers.tokenization_utils import TOKENIZER_CONFIG_FILE, AddedToken

from .megatron_lm__megatron_tokenizer import (
    MegatronTokenizer,  # fake import to make AutoTokenizer infer the dependency
)
from .megatron_lm__tokenizer import PATTERN_TIKTOKEN, PATTERN_TIKTOKEN_V2, CustomTikTokenizer

MegatronTokenizer  # make sure that auto-formatting doesn't remove the import


class MegatronTikTokenizer(PreTrainedTokenizer):
    vocab_files_names: dict[str, str] = {"vocab_file": "tiktoken_vocab.json"}
    model_input_names: list[str] = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        tiktoken_pattern: Literal["v1", "v2"],
        vocab_size: int,
        tiktoken_num_special_tokens: int,
        tiktoken_special_tokens: list[str] | None,
        add_bos_token: bool = False,  # nm5 does not use bos token
        add_eos_token: bool = False,  # nm5 does not use eos token
        **unused_kwargs,
    ):
        assert "chat_template" not in unused_kwargs, (
            "We enforce the Nemotron5 chat template from the code, "
            "please do not provide a chat_template in the tokenizer_config.json file"
        )

        pattern = PATTERN_TIKTOKEN if tiktoken_pattern == "v1" else PATTERN_TIKTOKEN_V2
        self._tokenizer = CustomTikTokenizer(
            path=vocab_file,
            pattern=pattern,
            vocab_size=vocab_size,
            num_special_tokens=tiktoken_num_special_tokens,
            special_tokens=tiktoken_special_tokens,
        )

        eos_token = self._tokenizer.detokenize([self._tokenizer.eos])
        bos_token = self._tokenizer.detokenize([self._tokenizer.bos])
        self.vocab = self._tokenizer.vocab
        super().__init__(
            eos_token=AddedToken(eos_token, normalized=False, special=True),
            bos_token=AddedToken(bos_token, normalized=False, special=True),
            pad_token=AddedToken(eos_token, normalized=False, special=True),
        )

        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.chat_template = NEMOTRON5_CHAT_TEMPLATE

        self._vocab_file_contents = Path(vocab_file).read_text()
        self._tokenizer_config = {
            "tiktoken_pattern": tiktoken_pattern,
            "vocab_size": vocab_size,
            "tiktoken_num_special_tokens": tiktoken_num_special_tokens,
            "tiktoken_special_tokens": tiktoken_special_tokens,
            "add_bos_token": add_bos_token,
            "add_eos_token": add_eos_token,
            "tokenizer_class": "MegatronTikTokenizer",
            "auto_map": {
                "AutoTokenizer": ["tokenization_decilm.MegatronTikTokenizer", None],
            },
        }

    def get_vocab(self) -> dict[str, int]:
        """to satisfy PreTrainedTokenizer.__init__()"""
        return self.vocab

    def tokenize(self, text: str, **kwargs) -> list[str]:
        return [text]

    def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:
        is_single_token = isinstance(tokens, str)
        if is_single_token:
            text = tokens
        else:
            assert len(tokens) == 1
            text = tokens[0]

        ids = self._tokenizer._model.encode(text, allowed_special="all")

        if is_single_token:
            assert len(ids) == 1, (
                f"Asked to convert a single token to its id, but it's not a single token: encode('{tokens}') = {ids}"
            )
            return ids[0]
        else:
            return ids

    def convert_ids_to_tokens(
        self, ids: int | list[int], skip_special_tokens: bool = False
    ) -> str | list[str]:
        is_single_id = isinstance(ids, int)
        if is_single_id:
            ids = [ids]

        if skip_special_tokens:
            ids = [idd for idd in ids if idd not in (self.eos_token_id, self.bos_token_id)]

        text = self._tokenizer.detokenize(ids)

        if is_single_id:
            return text
        else:
            return [text]

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Taken from LlamaTokenizer"""
        bos_token_id = [self.bos_token_id] if self.add_bos_token else []
        eos_token_id = [self.eos_token_id] if self.add_eos_token else []

        output = bos_token_id + token_ids_0 + eos_token_id

        if token_ids_1 is not None:
            output = output + bos_token_id + token_ids_1 + eos_token_id

        return output

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        legacy_format: bool | None = None,
        filename_prefix: str | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ) -> tuple[str, ...]:
        assert legacy_format is None, "Unsupported"
        assert filename_prefix is None, "Unsupported"
        assert not push_to_hub, "Unsupported"

        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        tokenizer_config_path = save_directory / TOKENIZER_CONFIG_FILE
        tokenizer_config_path.write_text(json.dumps(self._tokenizer_config, indent=2))

        vocab_files_name = self.vocab_files_names["vocab_file"]
        vocab_file_path = save_directory / vocab_files_name
        vocab_file_path.write_text(self._vocab_file_contents)

        custom_object_save(self, save_directory)

        return str(tokenizer_config_path), str(vocab_file_path)


NEMOTRON5_CHAT_TEMPLATE = """{% if messages[0].role != "system" %}
    {% set messages = [{"role": "system", "content": ""}] + messages %}
{% endif %}
{% for message in messages %}
  {% if message.role == "system" %}
<SPECIAL_10>System
{{ message.content }}
  {% elif message.role == "user" %}
<SPECIAL_11>User
{{ message.content }}
  {% elif message.role == "assistant" %}
<SPECIAL_11>Assistant
{{ message.content }}
  {% endif %}
{% endfor %}
{% if add_generation_prompt %}
<SPECIAL_11>Assistant
{% else %}
<SPECIAL_11>
{% endif %}"""
