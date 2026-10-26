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
"""Unit tests for the dummy forward that ``requantize_resmooth_fused_llm_layers`` traces."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers")

import transformers

from modelopt.torch.export.unified_export_hf import _llm_dummy_forward


class _Recorder(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *args, **kwargs):
        self.calls.append((args, kwargs))


def _vlm(class_name: str, architectures: list[str] | None) -> nn.Module:
    """A VLM whose root forward rejects text-only input, as a real one does without pixels."""

    def forward(self, *args, **kwargs):
        raise AssertionError("root forward needs pixel_values")

    model = type(class_name, (nn.Module,), {"forward": forward, "device": torch.device("cpu")})()
    model.config = SimpleNamespace(
        model_type="nemotron_vl_test", architectures=architectures, is_encoder_decoder=False
    )
    model.language_model = _Recorder()
    return model


@pytest.mark.parametrize(
    ("class_name", "architectures"),
    [
        ("RemoteVLM", ["NemotronH_Nano_VL_V2"]),  # detected from config.architectures
        ("NemotronH_Nano_VL_V2", None),  # built from a config without architectures
    ],
)
def test_nemotron_vl_runs_language_model_only(class_name: str, architectures: list[str] | None):
    model = _vlm(class_name, architectures)

    _llm_dummy_forward(model)

    ((args, kwargs),) = model.language_model.calls
    assert args[0].shape == (1, 2)
    assert kwargs == {"use_cache": False}


def test_non_nemotron_vl_runs_root_forward():
    model = _vlm("OtherVLM", ["OtherVLMForConditionalGeneration"])

    with pytest.raises(AssertionError, match="needs pixel_values"):
        _llm_dummy_forward(model)
    assert model.language_model.calls == []


def test_whisper_uses_mel_spectrogram_input(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        transformers.AutoFeatureExtractor,
        "from_pretrained",
        lambda name: SimpleNamespace(nb_max_frames=3000),
    )
    model = _Recorder()
    model.device = torch.device("cpu")
    model.dtype = torch.float32
    model.name_or_path = "openai/whisper-tiny"
    model.config = SimpleNamespace(
        model_type="whisper", architectures=None, is_encoder_decoder=True, num_mel_bins=80
    )

    _llm_dummy_forward(model)

    ((args, kwargs),) = model.calls
    assert args[0].shape == (1, 80, 3000)
    assert args[0].dtype == torch.float32
    assert kwargs["decoder_input_ids"].shape == (1, 2)
