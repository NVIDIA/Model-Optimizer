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

import hashlib
import sys
from unittest.mock import patch

import pytest
import torch
from _test_utils.torch.transformers_models import get_tiny_tokenizer
from transformers import BatchEncoding

import modelopt.torch.speculative.utils as spec_utils
from modelopt.torch.speculative.utils import (
    KIMI_K2_FILE_SHA256,
    KIMI_K2_PACKAGE_NAME,
    _setup_kimi_k2_decoder,
    get_conversation_input_ids,
)

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


def _serve_kimi_k2_files(tmp_path, monkeypatch, sources):
    """Serve ``sources`` as the downloaded Kimi-K2 files, with their hashes pinned."""
    for filename, source in sources.items():
        (tmp_path / filename).write_text(source)
        digest = hashlib.sha256(source.encode()).hexdigest()
        monkeypatch.setitem(KIMI_K2_FILE_SHA256, filename, digest)
    monkeypatch.setattr(
        spec_utils, "hf_hub_download", lambda repo_id, filename, **kwargs: str(tmp_path / filename)
    )
    # Keep the global DynamicCache patch out of the rest of the test session.
    monkeypatch.setattr(spec_utils, "_patch_dynamic_cache_compatibility", lambda: None)


def test_kimi_k2_decoder_refuses_unverified_code(tmp_path, monkeypatch):
    """No downloaded Kimi-K2 code runs unless every file matches its pinned hash."""
    marker = tmp_path / "executed"
    run_marker = f"open({str(marker)!r}, 'w').close()\n"
    _serve_kimi_k2_files(tmp_path, monkeypatch, dict.fromkeys(KIMI_K2_FILE_SHA256, run_marker))
    # Only the modeling file fails its check, so the valid config file must not run either.
    (tmp_path / "modeling_deepseek.py").write_text(run_marker + "# tampered\n")

    with pytest.raises(RuntimeError):
        _setup_kimi_k2_decoder()
    assert not marker.exists()


def test_kimi_k2_decoder_imports_verified_code(tmp_path, monkeypatch):
    """Verified files load as one package, so the modeling file's relative import resolves."""
    modeling = (
        "from .configuration_deepseek import DeepseekV3Config\n"
        "class DeepseekV3Attention:\n"
        "    _init_rope = forward = None\n"
        "class DeepseekV3DecoderLayer:\n"
        "    config_class = DeepseekV3Config\n"
        "    forward = None\n"
    )
    config = "class DeepseekV3Config:\n    pass\n"
    _serve_kimi_k2_files(
        tmp_path,
        monkeypatch,
        {"configuration_deepseek.py": config, "modeling_deepseek.py": modeling},
    )

    with patch.dict(sys.modules):
        decoder_cls = _setup_kimi_k2_decoder()
    assert decoder_cls.config_class.__module__ == f"{KIMI_K2_PACKAGE_NAME}.configuration_deepseek"


def test_kimi_k2_decoder_drops_failed_module(tmp_path, monkeypatch):
    """A module that raises while importing is not left half-initialized in sys.modules."""
    _serve_kimi_k2_files(
        tmp_path,
        monkeypatch,
        {"configuration_deepseek.py": "", "modeling_deepseek.py": "raise ImportError\n"},
    )

    with patch.dict(sys.modules):
        with pytest.raises(ImportError):
            _setup_kimi_k2_decoder()
        assert f"{KIMI_K2_PACKAGE_NAME}.modeling_deepseek" not in sys.modules
