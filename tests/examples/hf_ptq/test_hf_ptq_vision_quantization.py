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
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "hf_ptq"


@pytest.fixture
def hf_ptq(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("hf_ptq")


@pytest.mark.parametrize(
    ("recipe", "extracts_language_model"),
    [
        (None, True),
        ("huggingface/qwen3_vl/ptq/fp8_vision-kv_none", False),
    ],
)
def test_image_calibration_model_target_follows_recipe(
    hf_ptq, monkeypatch, recipe, extracts_language_model
):
    """Plain PTQ extracts the LM; a recipe keeps the complete VLM as its target."""
    full_model = SimpleNamespace(device=torch.device("cpu"))
    extracted_language_model = object()
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="</s>", padding_side="right")
    extraction_calls = []

    args = SimpleNamespace(
        use_fsdp2=False,
        specdec_offline_dataset=None,
        low_memory_mode=False,
        pyt_ckpt_path="dummy",
        dist_state=SimpleNamespace(device=torch.device("cpu")),
        gpu_max_mem_percentage=0.8,
        trust_remote_code=False,
        use_seq_device_map=False,
        attn_implementation=None,
        calib_with_images=True,
        recipe=recipe,
        auto_quantize_bits=None,
    )

    monkeypatch.setattr(hf_ptq, "get_model", lambda *args, **kwargs: full_model)
    monkeypatch.setattr(hf_ptq, "get_model_type", lambda model: "qwen3_vl")
    monkeypatch.setattr(hf_ptq, "is_nemotron_vl", lambda model: False)
    monkeypatch.setattr(
        hf_ptq.AutoProcessor,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(tokenizer=tokenizer),
    )

    def extract_language_model(model):
        extraction_calls.append(model)
        return extracted_language_model, "qwen3"

    monkeypatch.setattr(
        hf_ptq, "extract_and_prepare_language_model_from_vl", extract_language_model
    )

    loaded = hf_ptq.load_model(args)
    quantization_target = loaded[1]

    assert extraction_calls == ([full_model] if extracts_language_model else [])
    assert quantization_target is (
        extracted_language_model if extracts_language_model else full_model
    )


def test_image_calibration_uses_full_vlm_forward(hf_ptq, monkeypatch):
    args = SimpleNamespace(
        qformat="fp8",
        calib_with_images=True,
        specdec_offline_dataset=None,
    )
    full_model = torch.nn.Module()
    calib_dataloader = object()
    calibration_loop = object()
    calls = {}

    monkeypatch.setattr(hf_ptq, "is_quantized", lambda model: False)
    monkeypatch.setattr(hf_ptq, "need_calibration", lambda quant_cfg: True)

    def create_vlm_loop(model, dataloader):
        calls["loop_model"] = model
        calls["dataloader"] = dataloader
        return calibration_loop

    def quantize(model, quant_cfg, forward_loop):
        calls["quantization_target"] = model
        calls["forward_loop"] = forward_loop
        return model

    monkeypatch.setattr(hf_ptq, "create_vlm_calibration_loop", create_vlm_loop)
    monkeypatch.setattr(
        hf_ptq,
        "get_language_model_from_vl",
        lambda model: pytest.fail("The complete VLM must not be reinserted as its language model"),
    )
    monkeypatch.setattr(
        hf_ptq,
        "create_forward_loop",
        lambda **kwargs: pytest.fail("Text-only calibration loop must not be selected"),
    )
    monkeypatch.setattr(hf_ptq.mtq, "quantize", quantize)

    # Force the Nemotron reinsertion branch: a full-model recipe target must not be assigned as
    # its own nested language model.
    hf_ptq.mono_quantize(
        args,
        {"algorithm": "max"},
        full_model,
        full_model,
        model_type="qwen3_vl",
        calibration_only=False,
        calib_dataloader=calib_dataloader,
        is_nemotron_vl_model=True,
    )

    assert calls == {
        "loop_model": full_model,
        "dataloader": calib_dataloader,
        "quantization_target": full_model,
        "forward_loop": calibration_loop,
    }
