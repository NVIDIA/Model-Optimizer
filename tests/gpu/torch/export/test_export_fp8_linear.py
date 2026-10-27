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

"""Export-path tests for Hugging Face FP8Linear sources (CUDA-only).

``_QuantFP8Linear.unpack_weight`` dequantizes under ``torch.cuda.device``, so the suite
lives under ``tests/gpu/`` rather than ``tests/unit/``.
"""

import pytest
import torch
import torch.nn as nn

pytest.importorskip("transformers.integrations.finegrained_fp8")
from transformers.integrations.finegrained_fp8 import FP8Linear

import modelopt.torch.quantization as mtq
from modelopt.torch.export.unified_export_hf import _process_quantized_modules

BLOCK = 128
DIM = 2 * BLOCK


def _fp8_linear() -> FP8Linear:
    """Block-scaled FP8Linear holding a random weight, as loaded from an FP8 checkpoint."""
    module = FP8Linear(DIM, DIM, block_size=(BLOCK, BLOCK))
    weight = torch.randn(DIM, DIM) * 0.05
    blocks = weight.reshape(DIM // BLOCK, BLOCK, DIM // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)) / 448.0
    with torch.no_grad():
        module.weight.copy_(
            (blocks / scale[:, None, :, None]).reshape(DIM, DIM).to(torch.float8_e4m3fn)
        )
        module.weight_scale_inv.copy_(scale)
    return module


class _FP8Block(nn.Module):
    """Two FP8 linears: ``mlp`` gets quantized, ``attn`` is left out by the recipe."""

    def __init__(self):
        super().__init__()
        self.mlp = _fp8_linear()
        self.attn = _fp8_linear()

    def forward(self, x):
        return self.attn(self.mlp(x))


def _mlp_only_nvfp4_cfg() -> dict:
    """NVFP4 on ``mlp`` only, mirroring partial-model presets such as nvfp4_mlp_only."""
    nvfp4 = {
        "num_bits": (2, 1),
        "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
    }
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {"quantizer_name": "*mlp.weight_quantizer", "cfg": dict(nvfp4)},
        ],
        "algorithm": "max",
    }


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unquantized_fp8_linear_exported_in_export_dtype(dtype):
    """An FP8 source layer the recipe leaves unquantized is written in the export dtype,
    not in torch's default dtype (fp32)."""
    torch.manual_seed(0)
    model = _FP8Block().cuda()
    blocks = model.attn.weight.float().reshape(DIM // BLOCK, BLOCK, DIM // BLOCK, BLOCK)
    scale = model.attn.weight_scale_inv[:, None, :, None]
    expected = (blocks * scale).reshape(DIM, DIM).to(dtype)

    x = torch.randn(4, DIM, dtype=dtype, device="cuda")
    model = mtq.quantize(model, _mlp_only_nvfp4_cfg(), lambda m: m(x))
    _process_quantized_modules(model, dtype=dtype)

    assert model.mlp.weight.dtype == torch.uint8  # packed NVFP4
    assert model.attn.weight.dtype == dtype
    assert torch.equal(model.attn.weight, expected)
