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
"""GGML GPTQ with the CUDA encoders, through ``mtq.quantize`` and ``mtq.auto_quantize``."""

import copy

import pytest
import torch

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.extensions import get_cuda_ext_ggml
from modelopt.torch.quantization.ggml import GGML_FORMAT_REGISTRY, choice_search
from modelopt.torch.quantization.ggml.common import pinned_packed_weight
from modelopt.torch.quantization.ggml.gptq import compensated_block_hessian
from modelopt.torch.quantization.utils.calib_utils import compute_hessian_inverse


def _iq1_s_config(algorithm):
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "iq1_s", "block_sizes": {-1: 256}, "backend": "ggml"},
                "enable": True,
            },
        ],
        "algorithm": algorithm,
    }


def _correlated_inputs(cols=512, seed=0):
    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(256, cols, generator=generator) @ torch.randn(
        cols, cols, generator=generator
    )
    return inputs.cuda()


def _weighted_error(quantized, weight, inputs):
    return ((quantized - weight) @ inputs.T).square().sum()


def test_gptq_on_a_ggml_format_pins_the_payload_it_chose():
    torch.manual_seed(0)
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    original = model.weight.detach().clone()
    inputs = _correlated_inputs()
    plain = copy.deepcopy(model)
    mtq.quantize(plain, _iq1_s_config("max"), forward_loop=lambda m: m(inputs))

    mtq.quantize(
        model,
        _iq1_s_config({"method": "gptq", "block_size": 256}),
        forward_loop=lambda m: m(inputs),
    )

    iq1_s = GGML_FORMAT_REGISTRY["iq1_s"]
    packed = pinned_packed_weight(model.weight_quantizer, "iq1_s")
    decoded = iq1_s.dequantize(packed, torch.tensor(model.weight.shape), dtype=torch.float32)
    # IQ1_S keeps its pinned scales, so GPTQ leaves the weight unrounded; fake quant decodes the pin.
    torch.testing.assert_close(model(inputs), inputs @ decoded.T)
    assert torch.equal(iq1_s.pack(model.weight, model.weight_quantizer).flatten(), packed.flatten())
    plain_weight = plain.weight_quantizer(plain.weight).detach()
    assert _weighted_error(decoded, original, inputs) < _weighted_error(
        plain_weight, original, inputs
    )


def test_gptq_on_a_ggml_format_rejects_blocks_smaller_than_a_ggml_block():
    model = torch.nn.Linear(512, 8, bias=False).cuda()
    inputs = _correlated_inputs()

    with pytest.raises(ValueError, match="multiples of the quantization group size"):
        mtq.quantize(
            model,
            _iq1_s_config({"method": "gptq", "block_size": 128}),
            forward_loop=lambda m: m(inputs),
        )


def test_auto_quantize_refuses_gptq_on_native_fp8_weights_and_hands_back_the_model():
    # An FP8 checkpoint's forward dequantizes the stored weight with its block scales, but GPTQ
    # would encode the stored FP8 values, and the selected codes cannot decode into FP8.
    fp8_linear = pytest.importorskip("transformers.integrations.finegrained_fp8").FP8Linear
    torch.manual_seed(0)
    layers = []
    for _ in range(2):
        layer = fp8_linear(256, 256, block_size=(128, 128))
        blocks = (torch.randn(256, 256) * 0.05).reshape(2, 128, 2, 128)
        scale = blocks.abs().amax(dim=(1, 3)) / 448.0
        with torch.no_grad():
            layer.weight.copy_(
                (blocks / scale[:, None, :, None]).reshape(256, 256).to(torch.float8_e4m3fn)
            )
            layer.weight_scale_inv.copy_(scale)
        layers.append(layer)
    source = torch.nn.Sequential(*layers).cuda()
    model = copy.deepcopy(source)
    inputs = torch.randn(64, 256, dtype=torch.bfloat16, device="cuda")
    bits = GGML_FORMAT_REGISTRY["iq1_s"].effective_bits

    def search(algorithm):
        config = _iq1_s_config(algorithm)
        config["quant_cfg"][1]["cfg"]["effective_bits"] = bits
        mtq.auto_quantize(
            model,
            constraints={"effective_bits": 2.0},
            quantization_formats=[(config, "IQ1_S")],
            data_loader=[inputs],
            forward_step=lambda m, x: m(x),
            num_calib_steps=1,
            num_score_steps=1,
            method="kl_div",
        )

    with pytest.raises(NotImplementedError):
        search({"method": "gptq", "block_size": 256})

    assert not mto.ModeloptStateManager.is_converted(model)
    for param, original in zip(model.parameters(), source.parameters()):
        assert torch.equal(param, original)
    # The same model searches again with a calibration that leaves its weights alone.
    search("max")


@pytest.mark.parametrize("num_bits", ["iq2_xxs", "iq2_xs", "iq2_s", "iq1_s", "iq1_m"])
def test_fused_choice_search_matches_the_pytorch_search(monkeypatch, num_bits):
    get_cuda_ext_ggml(raise_if_failed=True)  # without it, both searches would be the PyTorch one
    ggml_format = GGML_FORMAT_REGISTRY[num_bits]
    inputs = _correlated_inputs()
    weight = torch.randn(64, 512, generator=torch.Generator().manual_seed(1)).cuda()
    h_inv = compute_hessian_inverse(inputs.T @ inputs / len(inputs), weight, 0.3)
    hessian = compensated_block_hessian(h_inv, slice(256, 512))[None]
    block = weight[:, 256:].contiguous()

    fused = choice_search.search_choices(ggml_format, block, hessian)
    monkeypatch.setattr(choice_search, "get_cuda_ext_ggml", lambda: None)
    assert torch.equal(fused, choice_search.search_choices(ggml_format, block, hessian))
