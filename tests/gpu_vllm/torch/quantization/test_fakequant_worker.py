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

"""CPU tests for the ``fakequant_worker`` config-driven calibration gate.

The removed ``QUANT_SKIP_CALIB`` env flag is replaced by ``need_calibration(quant_cfg)`` (config,
not env): the calibration dataset is built only when the recipe has an enabled quantizer that is
not marked ``type: "dynamic"``. A static recipe can therefore never silently skip its required
calibration, and a fully dynamic recipe skips the dataset (and the HF ``datasets`` stack).
"""

import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch.nn as nn

# The worker module lives under examples/vllm_serve and is imported as a top-level module there.
_VLLM_SERVE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "examples", "vllm_serve"
)
sys.path.insert(0, os.path.abspath(_VLLM_SERVE_DIR))

import fakequant_worker as fqw

_STATIC_CFG = {
    "quant_cfg": {
        "*input_quantizer": {"num_bits": 8, "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}
_DYNAMIC_CFG = {
    "quant_cfg": {
        "*input_quantizer": {"num_bits": 8, "type": "dynamic", "enable": True},
        "default": {"enable": False},
    },
    "algorithm": "max",
}


def _fake_self():
    model_runner = SimpleNamespace(
        model=nn.Module(),  # no quantizers -> the post-fold weight-quantizer check is a no-op
        model_config=SimpleNamespace(tokenizer="dummy-tokenizer"),
    )
    return SimpleNamespace(model_runner=model_runner, device="cpu")


@pytest.fixture
def calibration_calls(monkeypatch):
    """Stub every heavy dependency and record whether the dataset/forward_loop was built."""
    calls = {"datasets_built": 0, "forward_loops": []}

    monkeypatch.setattr(
        fqw,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(pad_token="<unk>")),
    )

    def _fake_loader(**kwargs):
        calls["datasets_built"] += 1
        return object()

    monkeypatch.setattr(fqw, "get_dataset_dataloader", _fake_loader)
    monkeypatch.setattr(fqw, "calibrate_fun", lambda dataloader, worker: lambda model: None)

    def _fake_quantize(model, cfg, forward_loop=None):
        calls["forward_loops"].append(forward_loop)
        return model

    monkeypatch.setattr(fqw.mtq, "quantize", _fake_quantize)
    monkeypatch.setattr(fqw.mtq, "print_quant_summary", lambda model: None)
    monkeypatch.setattr(fqw.mtq, "fold_weight", lambda model: None)
    monkeypatch.setattr(fqw, "post_restore_vllm_attentions", lambda model: None)
    monkeypatch.setattr(fqw, "disable_compilation", lambda model: nullcontext())
    monkeypatch.setitem(fqw.quant_config, "modelopt_state_path", None)
    monkeypatch.setitem(fqw.quant_config, "quant_file_path", None)
    return calls


def test_static_recipe_builds_calibration_dataset(monkeypatch, calibration_calls):
    monkeypatch.setattr(fqw, "get_quant_config", lambda quant_config, model: _STATIC_CFG)

    fqw._fakequant_run_prolog_worker(_fake_self())

    # A static recipe must calibrate: dataset built, forward_loop is a real calibration callable.
    assert calibration_calls["datasets_built"] == 1
    assert len(calibration_calls["forward_loops"]) == 1
    assert callable(calibration_calls["forward_loops"][0])


def test_dynamic_recipe_skips_calibration_dataset(monkeypatch, calibration_calls):
    monkeypatch.setattr(fqw, "get_quant_config", lambda quant_config, model: _DYNAMIC_CFG)

    fqw._fakequant_run_prolog_worker(_fake_self())

    # A fully dynamic recipe needs no calibration: no dataset built, forward_loop is None.
    assert calibration_calls["datasets_built"] == 0
    assert calibration_calls["forward_loops"] == [None]
