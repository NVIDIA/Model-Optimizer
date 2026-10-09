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
"""Unit tests for :mod:`modelopt.torch.export.trtllm.decoder_type` and its deprecated aliases."""

from types import SimpleNamespace

import pytest
import torch.nn as nn

import modelopt.torch.export as mte
from modelopt.torch.export import model_utils
from modelopt.torch.export.trtllm import model_config_export
from modelopt.torch.export.trtllm.decoder_type import MODEL_NAME_TO_DECODER_TYPE, get_decoder_type


def _named_module(name: str) -> nn.Module:
    return type(name, (nn.Module,), {})()


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("LlamaForCausalLM", "llama"),
        ("Llama4ForConditionalGeneration", "llama4"),
        ("DiffusionGemmaForCausalLM", "diffusion_gemma"),
        ("Gemma3ForCausalLM", "gemma3"),
        ("WhisperForConditionalGeneration", "whisper"),
        ("DeepseekV3ForCausalLM", "deepseek"),
        ("UnknownForCausalLM", None),
    ],
)
def test_get_decoder_type(class_name: str, expected: str | None):
    assert get_decoder_type(_named_module(class_name)) == expected


def test_get_model_type_is_deprecated():
    with pytest.warns(DeprecationWarning, match="get_model_type"):
        assert mte.get_model_type(_named_module("LlamaForCausalLM")) == "llama"


def test_model_name_to_type_is_deprecated():
    with pytest.warns(DeprecationWarning, match="MODEL_NAME_TO_TYPE"):
        assert model_utils.MODEL_NAME_TO_TYPE is MODEL_NAME_TO_DECODER_TYPE


def test_exporter_detects_decoder_type(monkeypatch: pytest.MonkeyPatch):
    seen = {}

    def fake_dtype(model: nn.Module):
        raise RuntimeError("stop after decoder_type resolution")

    monkeypatch.setattr(model_config_export, "get_dtype", fake_dtype)
    monkeypatch.setattr(
        model_config_export,
        "get_decoder_type",
        lambda model: seen.setdefault("decoder_type", get_decoder_type(model)),
    )
    with pytest.raises(RuntimeError, match="stop after"):
        next(
            model_config_export._torch_to_tensorrt_llm_checkpoint(_named_module("GPT2LMHeadModel"))
        )
    assert seen["decoder_type"] == "gpt"


def test_exporter_falls_back_for_undetectable_hf_decoder_type(monkeypatch: pytest.MonkeyPatch):
    """An unrecognized model with config.architectures exports through the generic path."""

    def fake_dtype(model: nn.Module):
        raise RuntimeError("stop after decoder_type resolution")

    monkeypatch.setattr(model_config_export, "get_dtype", fake_dtype)
    model = _named_module("Unknown")
    model.config = SimpleNamespace(architectures=["UnknownForCausalLM"])
    with (
        pytest.warns(UserWarning, match="Unknown decoder_type for Unknown"),
        pytest.raises(RuntimeError, match="stop after"),
    ):
        next(model_config_export._torch_to_tensorrt_llm_checkpoint(model))


def test_exporter_rejects_undetectable_decoder_type_without_architectures():
    """Without config.architectures (e.g. Megatron-Core) the generic path cannot work."""
    with pytest.raises(ValueError, match="Pass decoder_type explicitly"):
        next(model_config_export._torch_to_tensorrt_llm_checkpoint(_named_module("GPTModel")))
