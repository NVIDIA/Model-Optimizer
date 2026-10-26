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

"""The NVFP4 transcode of decoded GGML weights, for W1A4 and W2A4 fake quant."""

import pytest
import torch

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.ggml.common as ggml_common
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY
from modelopt.torch.quantization.ggml.nvfp4 import (
    _round_e2m1,
    _tables,
    nvfp4_transcode,
    nvfp4_transcode_scales,
)
from modelopt.torch.quantization.qtensor import NVFP4QTensor

IQ_FORMATS = ["iq1_s", "iq1_m", "iq2_xxs", "iq2_xs", "iq2_s"]


def _decoded(num_bits, rows=32, cols=1024):
    weight = torch.randn(rows, cols, generator=torch.Generator().manual_seed(0)) * 0.02
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    packed, shape = ggml_format.quantize(weight)
    return ggml_format.dequantize(packed, shape, dtype=torch.float32)


@pytest.mark.parametrize("num_bits", IQ_FORMATS)
def test_transcode_is_what_nvfp4_packing_stores_and_beats_the_amax_scale(num_bits):
    decoded = _decoded(num_bits)
    block_scales, global_scale = nvfp4_transcode_scales(decoded)
    transcoded = nvfp4_transcode(decoded, block_scales, global_scale)

    # The NVFP4 conversion packs with these scales and must store exactly the fake-quant weight.
    packed, _, _ = NVFP4QTensor.quantize(decoded, 16, block_scales, global_scale)
    stored = packed.dequantize(
        torch.float32, scale=block_scales, double_scale=global_scale, block_sizes={-1: 16}
    )
    assert torch.equal(stored, transcoded)
    assert torch.equal(transcoded.bfloat16().float(), transcoded)
    # GPTQ transcodes one 256-column group at a time.
    groups = torch.cat([nvfp4_transcode(g) for g in decoded.split(256, dim=-1)], dim=-1)
    assert torch.equal(groups, transcoded)

    # No block does worse than at the E4M3 scale nearest amax / 6, which the search includes.
    e4m3, *_ = _tables(decoded.device)
    blocks = decoded.reshape(-1, 16)
    amax_code = torch.searchsorted(e4m3, blocks.abs().amax(-1) / (6 * global_scale))
    amax_scale = e4m3[amax_code.clamp(max=len(e4m3) - 1)].unsqueeze(-1) * global_scale
    amax_error = (blocks - _round_e2m1(blocks / amax_scale) * amax_scale).square().sum(-1)
    error = (blocks - transcoded.reshape(-1, 16)).square().sum(-1)
    assert (error <= amax_error).all()
    assert error.sum() < 0.9 * amax_error.sum()


@pytest.mark.parametrize("num_bits", IQ_FORMATS)
def test_fake_quant_transcodes_the_decoded_weight_and_caches_its_scales(monkeypatch, num_bits):
    model = torch.nn.Linear(1024, 32, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.randn(32, 1024, generator=torch.Generator().manual_seed(0)) * 0.02)
    weight_cfg = {
        "num_bits": num_bits,
        "block_sizes": {-1: 256},
        "backend": "ggml",
        "backend_extra_args": {"transcode": "nvfp4"},
    }
    mtq.quantize(
        model,
        {
            "quant_cfg": [
                {"quantizer_name": "*", "enable": False},
                {"quantizer_name": "*weight_quantizer", "cfg": weight_cfg},
            ],
            "algorithm": None,
        },
    )
    searches = []
    search = ggml_common.nvfp4_transcode_scales
    monkeypatch.setattr(
        ggml_common, "nvfp4_transcode_scales", lambda w: searches.append(w) or search(w)
    )

    with torch.no_grad():
        first = model.weight_quantizer(model.weight)
        second = model.weight_quantizer(model.weight)

    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    packed, shape = ggml_format.quantize(model.weight.detach())
    expected = nvfp4_transcode(ggml_format.dequantize(packed, shape, dtype=torch.float32))
    # Close rather than equal: the straight-through form w + (q - w) rounds. The decoding without
    # the transcode is about 1e-3 away.
    torch.testing.assert_close(first, expected)
    assert torch.equal(second, first)
    assert len(searches) == 1
