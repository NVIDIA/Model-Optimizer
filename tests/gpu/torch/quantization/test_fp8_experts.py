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

"""Native FP8 expert PTQ and packed NVFP4 export."""

import pytest
import torch
from transformers import PretrainedConfig
from transformers.integrations.finegrained_fp8 import FP8Experts

import modelopt.torch.quantization as mtq
from modelopt.torch.export.moe_utils import _export_fused_experts


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_gate", [True, False])
def test_fp8_experts_nvfp4_export(has_gate, dtype):
    torch.manual_seed(42)
    config = PretrainedConfig(
        hidden_size=256, intermediate_size=256, num_experts=2, hidden_act="silu"
    )
    experts = FP8Experts(config, block_size=(128, 128), has_gate=has_gate).cuda()
    first = "gate_up_proj" if has_gate else "up_proj"
    expected = {}
    for name in (first, "down_proj"):
        weight = getattr(experts, name)
        scales = getattr(experts, f"{name}_scale_inv")
        with torch.no_grad():
            weight.copy_(torch.randn(weight.shape, device="cuda").to(weight.dtype))
            scales.copy_(
                torch.linspace(0.01, 0.08, scales.numel(), device="cuda").reshape(scales.shape)
            )
        expected[name] = weight.float() * scales.repeat_interleave(128, 1).repeat_interleave(128, 2)
    inputs = torch.randn(2, 256, device="cuda", dtype=dtype)
    indices = torch.tensor([[0], [1]], device="cuda")
    routing = torch.ones(2, 1, device="cuda")
    mtq.quantize(experts, mtq.NVFP4_DEFAULT_CFG, lambda m: m(inputs, indices, routing))
    assert torch.isfinite(experts(inputs, indices, routing)).all()
    _export_fused_experts(experts, torch.float32)
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda"
    )
    for idx in range(2):
        exported = getattr(experts, str(idx))
        projections = [("down_proj", expected["down_proj"][idx])]
        if has_gate:
            projections += [
                ("gate_proj", expected[first][idx, :256]),
                ("up_proj", expected[first][idx, 256:]),
            ]
            torch.testing.assert_close(
                exported.gate_proj.weight_scale_2, exported.up_proj.weight_scale_2
            )
        else:
            projections += [("up_proj", expected[first][idx])]
        for name, reference in projections:
            proj = getattr(exported, name)
            packed = proj.weight
            codes = torch.stack([packed & 15, packed >> 4], -1).reshape(reference.shape).long()
            decoded = (
                levels[codes]
                * proj.weight_scale.float().repeat_interleave(16, -1)
                * proj.weight_scale_2
            )
            relative_l2 = torch.linalg.vector_norm(decoded - reference) / torch.linalg.vector_norm(
                reference
            )
            assert relative_l2 < 0.15
    assert not any("scale_inv" in key for key in experts.state_dict())
