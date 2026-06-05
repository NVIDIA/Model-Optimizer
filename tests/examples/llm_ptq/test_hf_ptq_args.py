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

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "llm_ptq"


def _import_hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


def _parse_hf_ptq_args(monkeypatch, *args):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(sys, "argv", ["hf_ptq.py", *args])
    parsed_args = hf_ptq.parse_args()
    parsed_args.dataset = (
        parsed_args.dataset.split(",")
        if isinstance(parsed_args.dataset, str)
        else parsed_args.dataset
    )
    parsed_args.calib_size = [int(num_sample) for num_sample in parsed_args.calib_size.split(",")]
    return hf_ptq, parsed_args


def test_parse_args_rejects_autoquant_image_calibration(monkeypatch):
    hf_ptq = _import_hf_ptq(monkeypatch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hf_ptq.py",
            "--pyt_ckpt_path",
            "nemotron-vl",
            "--auto_quantize_bits",
            "5.0",
            "--calib_with_images",
        ],
    )

    with pytest.raises(SystemExit) as error:
        hf_ptq.parse_args()

    assert error.value.code == 2


def test_load_model_keeps_nemotron_vl_text_calibration_for_autoquant(monkeypatch):
    hf_ptq, args = _parse_hf_ptq_args(
        monkeypatch,
        "--pyt_ckpt_path",
        "nemotron-vl",
        "--auto_quantize_bits",
        "5.0",
    )
    fake_model = SimpleNamespace(device="cpu")
    fake_tokenizer = SimpleNamespace(padding_side="right", pad_token="<pad>")

    monkeypatch.setattr(hf_ptq, "get_model", lambda *args, **kwargs: fake_model)
    monkeypatch.setattr(hf_ptq, "get_model_type", lambda model: "qwen2")
    monkeypatch.setattr(hf_ptq, "get_tokenizer", lambda *args, **kwargs: fake_tokenizer)
    monkeypatch.setattr(hf_ptq, "is_nemotron_vl", lambda model: True)

    full_model, language_model, _, _, _, tokenizer, _, _, _ = hf_ptq.load_model(args)

    assert args.calib_with_images is False
    assert full_model is fake_model
    assert language_model is fake_model
    assert tokenizer is fake_tokenizer


def test_qwen_autoquant_disabled_layers_are_scoped_to_qwen_models(monkeypatch):
    hf_ptq = _import_hf_ptq(monkeypatch)
    qwen_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_moe"))
    llama_model = SimpleNamespace(config=SimpleNamespace(model_type="llama"))
    qwen_only_patterns = {
        "*shared_expert_gate*",
        "*linear_attn.in_proj_a*",
        "*linear_attn.in_proj_b*",
    }

    monkeypatch.setattr(hf_ptq, "is_multimodal_model", lambda model: False)

    qwen_disabled_layers = set(hf_ptq._get_auto_quantize_disabled_layers(qwen_model))
    llama_disabled_layers = set(hf_ptq._get_auto_quantize_disabled_layers(llama_model))

    assert qwen_only_patterns <= qwen_disabled_layers
    assert qwen_only_patterns.isdisjoint(llama_disabled_layers)
