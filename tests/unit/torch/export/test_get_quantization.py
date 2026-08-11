# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import fnmatch

import pytest
import torch
from _test_utils.torch.export.utils import (
    ToyModel,
    partial_fp8_config,
    partial_nvfp4_config,
    partial_w4a8_config,
)

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import get_quantization_format
from modelopt.torch.export.model_config import (
    QUANTIZATION_FP8,
    QUANTIZATION_FP8_PB_W8A8,
    QUANTIZATION_NVFP4,
    QUANTIZATION_W4A8_AWQ,
)
from modelopt.torch.export.quant_utils import get_quant_config
from modelopt.torch.quantization.nn import NVFP4StaticQuantizer

# Block-wise FP8 W8A8 on the ".1" linear: weight block + enabled dynamic input quantizer.
_fp8_pb_w8a8_config = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*.1.weight_quantizer",
            "cfg": {"num_bits": (4, 3), "block_sizes": {-1: 128, -2: 128}, "axis": None},
            "enable": True,
        },
        {
            "quantizer_name": "*.1.input_quantizer",
            "cfg": {"num_bits": (4, 3), "block_sizes": {-1: 128, "type": "dynamic"}, "axis": None},
            "enable": True,
        },
    ],
    "algorithm": "max",
}


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (partial_fp8_config, QUANTIZATION_FP8),
        (partial_w4a8_config, QUANTIZATION_W4A8_AWQ),
        (_fp8_pb_w8a8_config, QUANTIZATION_FP8_PB_W8A8),
    ],
)
def test_get_quantization_format(config, expected):
    model = ToyModel()
    mtq.quantize(model, config, lambda x: x(torch.randn(1, 4, 10)))
    assert get_quantization_format(model) == expected


def test_nvfp4_static_quantizer_export():
    """NVFP4StaticQuantizer: get_quantization_format returns NVFP4 and get_quant_config returns export config."""
    model = ToyModel()
    mtq.quantize(model, partial_nvfp4_config, lambda x: x(torch.randn(1, 4, 10)))

    # Convert all weight quantizers to NVFP4StaticQuantizer
    for module in model.modules():
        tq = getattr(module, "weight_quantizer", None)
        if tq is not None and hasattr(tq, "_amax") and not isinstance(tq, NVFP4StaticQuantizer):
            global_amax = tq._amax.max() if tq._amax.dim() > 0 else tq._amax
            NVFP4StaticQuantizer.from_tensor_quantizer(tq, global_amax=global_amax)

    assert get_quantization_format(model) == QUANTIZATION_NVFP4

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"
    assert quant_config["quantization"]["group_size"] == 16


class _FakeTopKRouter(torch.nn.Module):
    """Mimics a transformers>=5.0 MoE router: owns a ``weight`` but is NOT an ``nn.Linear``.

    ``mtq.quantize`` only attaches quantizers to registered modules (e.g. ``nn.Linear``), so a
    router like this never receives one -- reproducing the condition behind NVBug 5718750.
    """

    def __init__(self, hidden: int, num_experts: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(num_experts, hidden))
        self.top_k = 2
        self.num_experts = num_experts

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight)


class _FakeMoEBlock(torch.nn.Module):
    def __init__(self, hidden: int = 16, num_experts: int = 4):
        super().__init__()
        self.gate = _FakeTopKRouter(hidden, num_experts)
        self.experts = torch.nn.ModuleList(
            torch.nn.Linear(hidden, hidden, bias=False) for _ in range(num_experts)
        )

    def forward(self, x):
        self.gate(x)  # exercise the router so it is reachable
        out = x
        for expert in self.experts:
            out = expert(out)
        return out


class _FakeMoEModel(torch.nn.Module):
    def __init__(self, hidden: int = 16, num_experts: int = 4):
        super().__init__()
        self.block = _FakeMoEBlock(hidden, num_experts)

    def forward(self, x):
        return self.block(x)


_nvfp4_all_linears_config = {
    "quant_cfg": [
        {"quantizer_name": "*", "enable": False},
        {
            "quantizer_name": "*weight_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
        {
            "quantizer_name": "*input_quantizer",
            "cfg": {
                "num_bits": (2, 1),
                "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
                "axis": None,
            },
            "enable": True,
        },
    ],
    "algorithm": "max",
}


def test_moe_router_excluded_when_not_quantized():
    """NVBug 5718750: a non-Linear MoE router (transformers>=5.0 TopKRouter) gets no quantizer.

    Its BF16 weight is still exported, so it must be listed in ``exclude_modules``; otherwise
    deployment frameworks treat it as a quantized weight and fail to load the checkpoint.
    """
    hidden = 16
    model = _FakeMoEModel(hidden=hidden)
    mtq.quantize(model, _nvfp4_all_linears_config, lambda m: m(torch.randn(2, hidden)))

    # The router is not an nn.Linear, so quantize attached no quantizer to it.
    assert not hasattr(model.block.gate, "weight_quantizer")
    # The experts are quantized to NVFP4.
    assert get_quantization_format(model.block.experts[0]) == QUANTIZATION_NVFP4

    quant_config = get_quant_config(model)
    assert quant_config["quantization"]["quant_algo"] == "NVFP4"

    exclude_modules = quant_config["quantization"]["exclude_modules"]
    assert any(fnmatch.fnmatch("block.gate", pattern) for pattern in exclude_modules), (
        f"MoE router 'block.gate' missing from exclude_modules: {exclude_modules}"
    )
    # The quantized experts must NOT be excluded.
    assert not any(fnmatch.fnmatch("block.experts.0", pattern) for pattern in exclude_modules), (
        f"Quantized expert wrongly excluded: {exclude_modules}"
    )


def test_moe_router_names_handle_root_module():
    """When the MoE block itself is the root module, router names have no leading dot."""
    from modelopt.torch.export.quant_utils import _get_unquantized_moe_router_names

    block = _FakeMoEBlock(hidden=16)
    # name == "" for the root module; the router must be "gate", not ".gate".
    assert _get_unquantized_moe_router_names(block) == ["gate"]
