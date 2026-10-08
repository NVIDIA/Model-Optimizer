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

"""Exercise Nano-VL calibration with independently dispatched vision and token embeddings."""

from types import SimpleNamespace

import pytest
import torch
from accelerate import dispatch_model
from torch import nn

import modelopt.torch.quantization as mtq
from examples.hf_ptq.nemotron_vl_calib import safe_nemotron_vl_forward


class _LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8)
        self.received = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, **kwargs):
        self.received = inputs_embeds.detach().clone()
        return self.head(inputs_embeds)


class _NanoVL(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _LanguageModel()
        self.vision_model = nn.Linear(4, 4, dtype=torch.float64)
        self.vision_model.config = SimpleNamespace(torch_dtype=torch.float64)
        self.mlp1 = nn.Linear(4, 4, dtype=torch.float64)
        self.img_context_token_id = 7

    def extract_feature(self, pixel_values):
        return self.mlp1(self.vision_model(pixel_values))


@pytest.mark.parametrize(
    "devices",
    [
        ("cpu", "cpu"),
        pytest.param(
            ("cuda:0", "cuda:1"),
            marks=pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs"),
        ),
    ],
)
@pytest.mark.parametrize("extra_features", [False, True])
@torch.no_grad()
def test_calibration_aligns_vision_features(devices, extra_features):
    token_device, vision_device = devices
    model = dispatch_model(
        _NanoVL().eval(),
        device_map={
            "language_model": token_device,
            "vision_model": vision_device,
            "mlp1": vision_device,
        },
        force_hooks=True,
    )
    batch = {
        "input_ids": torch.tensor([[1, 7, 7, 2]], device=token_device),
        "pixel_values": torch.randn(2, 3 if extra_features else 2, 4, device=token_device),
        "image_flags": torch.tensor([[1], [0]], device=token_device),
    }
    expected = model.language_model.get_input_embeddings()(batch["input_ids"]).clone()
    features = model.extract_feature(batch["pixel_values"].double())
    assert str(features.device) == vision_device
    expected[:, 1:3] = features[0, :2].to(expected)

    config = {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "language_model.head.*_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
                "enable": True,
            },
        ],
        "algorithm": "max",
    }
    mtq.quantize(model, config, lambda model: safe_nemotron_vl_forward(model, batch))
    torch.testing.assert_close(model.language_model.received, expected)
    assert model.language_model.head.input_quantizer.amax.item() > 0
