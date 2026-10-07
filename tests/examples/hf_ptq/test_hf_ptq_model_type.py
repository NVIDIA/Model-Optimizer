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
"""hf_ptq keys its model-specific behavior on the root model's Hugging Face ``model_type``."""

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


@pytest.fixture
def hf_ptq(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


@pytest.mark.parametrize(
    ("model_type", "skips_generation"),
    [
        ("deepseek_v3", True),
        ("deepseek_v32", True),
        ("kimi_k2", True),  # DeepSeek-V3 architecture under its own model_type
        ("kimi_k25", True),  # VLM wrapping a kimi_k2 language model
        ("nemotron_h", True),
        ("llama", False),
    ],
)
def test_pre_quantize_generation_skip(hf_ptq, model_type: str, skips_generation: bool):
    generate_calls = []
    full_model = SimpleNamespace(generate=lambda *args, **kwargs: generate_calls.append(1) or "out")
    args = SimpleNamespace(specdec_offline_dataset=None, skip_generate=False)
    batch = {"input_ids": torch.ones(2, 4, dtype=torch.long)}

    _, _, generated = hf_ptq.pre_quantize(args, full_model, model_type, None, [batch], False)

    assert (generated is None) == skips_generation
    assert len(generate_calls) == (0 if skips_generation else 1)


@pytest.mark.parametrize(
    ("model_type", "expected"),
    [
        ("t5", True),
        ("mt5", True),
        ("mbart", True),
        ("whisper", True),
        # Encoder-decoder, but not supported by the TensorRT-LLM exporter: unified HF export.
        ("longt5", False),
        ("plbart", False),
        ("t5gemma", False),
        ("florence2", False),  # VLM whose language model is BART; the root type decides
        ("qwen3_vl", False),
    ],
)
def test_is_trtllm_enc_dec_export(hf_ptq, model_type: str, expected: bool):
    assert hf_ptq.is_trtllm_enc_dec_export(model_type) is expected


def _config(model_type: str, is_encoder_decoder: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_type=model_type, is_encoder_decoder=is_encoder_decoder, architectures=None
    )


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config("longt5", True), True),
        (_config("plbart", True), True),
        (_config("t5gemma", True), True),
        (_config("whisper", True), True),
        (_config("llama", False), False),
        # Structurally encoder-decoder, but generate() returns prompt+canvas concatenated.
        (_config("diffusion_gemma", True), False),
    ],
)
def test_generate_excludes_prompt(hf_ptq, config: SimpleNamespace, expected: bool):
    assert hf_ptq.generate_excludes_prompt(SimpleNamespace(config=config)) is expected


@pytest.mark.parametrize(
    ("config", "generated"),
    [
        # Encoder-decoder generate() returns only the decoder sequence, shorter than the prompt.
        (_config("longt5", True), [[7, 8]]),
        # Decoder-only generate() returns the prompt followed by the answer.
        (_config("llama", False), [[1, 2, 3, 4, 5, 7, 8]]),
    ],
)
def test_post_quantize_preview_keeps_whole_answer(
    hf_ptq, monkeypatch: pytest.MonkeyPatch, capsys, config: SimpleNamespace, generated
):
    """The preview must decode the answer, not slice a prompt-length prefix off it."""
    monkeypatch.setattr(hf_ptq, "export_quantized", lambda *args, **kwargs: None)
    full_model = SimpleNamespace(
        config=config, generate=lambda *args, **kwargs: torch.tensor(generated)
    )
    tokenizer = SimpleNamespace(
        batch_decode=lambda ids, **kwargs: [" ".join(map(str, row.tolist())) for row in ids]
    )
    args = SimpleNamespace(specdec_offline_dataset=None, verbose=False)

    hf_ptq.post_quantize(
        args,
        full_model,
        full_model,
        config.model_type,
        tokenizer,
        None,
        torch.tensor([[1, 2, 3, 4, 5]]),
        None,
        torch.tensor(generated),
        False,
        None,
        None,
        None,
        None,
    )

    assert "example outputs after ptq: ['7 8']" in capsys.readouterr().out
