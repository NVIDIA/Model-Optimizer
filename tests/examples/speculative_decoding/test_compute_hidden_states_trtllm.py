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

"""Regression test for the TRT-LLM offline hidden-state dump tokenization.

``tokenizer.apply_chat_template`` returns a ``BatchEncoding`` (a dict carrying
``input_ids``) on newer ``transformers`` versions rather than a plain list of token
ids. The dump counted tokens with ``len(input_ids)``, which then evaluated to the
number of dict fields (2) instead of the real token count, so every conversation
tripped the ``num_input_tokens <= 10`` length filter and was silently dropped --
producing zero ``.pt`` files and failing offline EAGLE3 training with
``No .pt files found``.

Lives in the trtllm example lane because ``compute_hidden_states_trtllm`` imports
``tensorrt_llm`` at module load.
"""

import importlib
import sys
from collections.abc import Mapping

import pytest
from _test_utils.examples.run_command import MODELOPT_ROOT
from _test_utils.torch.transformers_models import create_tiny_qwen3_dir
from transformers import AutoTokenizer

pytest.importorskip("tensorrt_llm")


@pytest.fixture(scope="module")
def compute_hidden_states_trtllm():
    """Import the TRT-LLM hidden-state dump example module."""
    example_dir = MODELOPT_ROOT / "examples" / "speculative_decoding" / "collect_hidden_states"
    sys.path.insert(0, str(example_dir))
    try:
        yield importlib.import_module("compute_hidden_states_trtllm")
    finally:
        sys.path.remove(str(example_dir))
        sys.modules.pop("compute_hidden_states_trtllm", None)


def test_get_conversation_input_ids_returns_token_sequence(tmp_path, compute_hidden_states_trtllm):
    model_dir = create_tiny_qwen3_dir(tmp_path, with_tokenizer=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    conversations = [
        {"role": "user", "content": "Explain why the sky is blue in a few sentences."},
        {
            "role": "assistant",
            "content": "Rayleigh scattering makes shorter blue wavelengths scatter more. " * 8,
        },
    ]

    input_ids = compute_hidden_states_trtllm.get_conversation_input_ids(tokenizer, conversations)

    # Must be a flat token-id sequence, never a BatchEncoding/dict -- otherwise len() is
    # the field count (2), which trips the dump's `num_input_tokens <= 10` filter and
    # silently drops every conversation.
    assert not isinstance(input_ids, Mapping)
    num_tokens = input_ids.shape[1] if hasattr(input_ids, "shape") else len(input_ids)
    assert num_tokens > 10

    # And it reflects the real token count of the rendered chat prompt.
    rendered = tokenizer.apply_chat_template(
        conversations, add_generation_prompt=False, tokenize=False
    )
    expected = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
    assert num_tokens == expected
