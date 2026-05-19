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

from types import SimpleNamespace

import torch

import modelopt.torch.export.unified_export_hf as export_hf


class _DummyModel(torch.nn.Module):
    config = SimpleNamespace(is_encoder_decoder=False)
    device = torch.device("cpu")


def test_nvfp4_awq_export_enables_grouped_head_prequant_fusion(monkeypatch):
    """NVFP4 AWQ export should fuse GQA/MQA o_proj scales when possible."""
    fuse_calls = []

    monkeypatch.setattr(export_hf, "get_quantization_format", lambda model: "nvfp4_awq")
    monkeypatch.setattr(
        export_hf,
        "fuse_prequant_to_linear",
        lambda model, **kwargs: fuse_calls.append(kwargs),
    )
    monkeypatch.setattr(export_hf, "is_moe", lambda module: False)
    monkeypatch.setattr(
        export_hf,
        "collect_shared_input_modules",
        lambda model, forward, collect_layernorms=True: ({}, {}),
    )
    monkeypatch.setattr(export_hf, "_fuse_shared_input_modules", lambda *args, **kwargs: {})

    export_hf.requantize_resmooth_fused_llm_layers(_DummyModel())

    assert fuse_calls == [{"fuse_grouped_heads": True}]
