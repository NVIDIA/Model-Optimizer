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


import pytest
import torch
from _test_utils.torch.export.utils import ToyModel, partial_fp8_config, partial_w4a8_config

import modelopt.torch.quantization as mtq
from modelopt.torch.export.layer_utils import sync_linear_attn_fused_projection_amax
from modelopt.torch.export.unified_export_hf import _export_quantized_weight
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn.modules.tensor_quantizer import (
    NVFP4StaticQuantizer,
    TensorQuantizer,
)
from modelopt.torch.quantization.utils import quantizer_attr_names


@pytest.mark.parametrize(
    "weight_name",
    ["weight", "weight_2", "some_other_w"],
)
def test_quantizer_attr_names(weight_name):
    quantizer_attrs = quantizer_attr_names(weight_name)
    if weight_name == "weight":
        assert quantizer_attrs.weight_scale == "weight_scale"
        assert quantizer_attrs.input_scale == "input_scale"
        assert quantizer_attrs.weight_scale_2 == "weight_scale_2"
        assert quantizer_attrs.weight_quantizer == "weight_quantizer"
        assert quantizer_attrs.input_quantizer == "input_quantizer"
        assert quantizer_attrs.output_quantizer == "output_quantizer"
        assert quantizer_attrs.output_scale == "output_scale"
    else:
        assert quantizer_attrs.weight_scale == f"{weight_name}_weight_scale"
        assert quantizer_attrs.input_scale == f"{weight_name}_input_scale"
        assert quantizer_attrs.weight_scale_2 == f"{weight_name}_weight_scale_2"
        assert quantizer_attrs.weight_quantizer == f"{weight_name}_weight_quantizer"
        assert quantizer_attrs.input_quantizer == f"{weight_name}_input_quantizer"
        assert quantizer_attrs.output_quantizer == f"{weight_name}_output_quantizer"
        assert quantizer_attrs.output_scale == f"{weight_name}_output_scale"


def test_export_per_tensor_quantized_weight():
    model = ToyModel(dims=[32, 256, 32, 128])

    mtq.quantize(model, partial_fp8_config, lambda x: x(torch.randn(1, 4, 32)))

    orig_dtype = model.linears[0].weight.dtype
    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[0], torch.float32, "weight")
    assert model.linears[0].weight.dtype == orig_dtype
    assert hasattr(model.linears[0], quantizer_attrs.weight_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.weight_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[0], quantizer_attrs.weight_scale_2)
    assert not hasattr(model.linears[0], quantizer_attrs.input_scale)
    assert hasattr(model.linears[0], quantizer_attrs.input_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.input_quantizer).is_enabled
    assert hasattr(model.linears[0], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[0], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[0], quantizer_attrs.output_scale)

    _export_quantized_weight(model.linears[1], torch.float32, "weight")
    assert model.linears[1].weight.dtype == torch.float8_e4m3fn
    assert hasattr(model.linears[1], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.weight_scale)
    assert not hasattr(model.linears[1], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[1], quantizer_attrs.input_quantizer)
    assert hasattr(model.linears[1], quantizer_attrs.input_scale)
    assert hasattr(model.linears[1], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[1], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[1], quantizer_attrs.output_scale)


def test_export_per_block_quantized_weight():
    model = ToyModel(dims=[32, 256, 256, 32])

    mtq.quantize(model, partial_w4a8_config, lambda x: x(torch.randn(1, 4, 32)))

    quantizer_attrs = quantizer_attr_names("weight")
    _export_quantized_weight(model.linears[2], torch.float32, "weight")
    assert model.linears[2].weight.dtype == torch.uint8
    assert hasattr(model.linears[2], quantizer_attrs.weight_quantizer)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale)
    assert hasattr(model.linears[2], quantizer_attrs.weight_scale_2)
    assert hasattr(model.linears[2], quantizer_attrs.input_scale)
    assert hasattr(model.linears[2], quantizer_attrs.input_quantizer)

    assert hasattr(model.linears[2], quantizer_attrs.output_quantizer)
    assert not getattr(model.linears[2], quantizer_attrs.output_quantizer).is_enabled
    assert not hasattr(model.linears[2], quantizer_attrs.output_scale)


class _GatedDeltaNetProjectionToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = torch.nn.Module()
        self.linear_attn.in_proj_qkv = torch.nn.Linear(16, 48, bias=False)
        self.linear_attn.in_proj_z = torch.nn.Linear(16, 16, bias=False)


def _attach_quantizers(module, weight_cfg, input_cfg):
    module.weight_quantizer = TensorQuantizer(weight_cfg)
    module.input_quantizer = TensorQuantizer(input_cfg)


def test_sync_linear_attn_fused_projection_fp8_amax():
    model = _GatedDeltaNetProjectionToy()
    quant_cfg = QuantizerAttributeConfig(num_bits=(4, 3), axis=None)
    _attach_quantizers(model.linear_attn.in_proj_qkv, quant_cfg, quant_cfg)
    _attach_quantizers(model.linear_attn.in_proj_z, quant_cfg, quant_cfg)

    model.linear_attn.in_proj_qkv.weight_quantizer.amax = torch.tensor(3.0)
    model.linear_attn.in_proj_z.weight_quantizer.amax = torch.tensor(5.0)
    model.linear_attn.in_proj_qkv.input_quantizer.amax = torch.tensor(7.0)
    model.linear_attn.in_proj_z.input_quantizer.amax = torch.tensor(11.0)

    synced = sync_linear_attn_fused_projection_amax(model)

    assert synced == 1
    assert torch.equal(model.linear_attn.in_proj_qkv.weight_quantizer.amax, torch.tensor(5.0))
    assert torch.equal(model.linear_attn.in_proj_z.weight_quantizer.amax, torch.tensor(5.0))
    assert torch.equal(model.linear_attn.in_proj_qkv.input_quantizer.amax, torch.tensor(11.0))
    assert torch.equal(model.linear_attn.in_proj_z.input_quantizer.amax, torch.tensor(11.0))
    assert hasattr(model.linear_attn, "in_proj_qkv")
    assert hasattr(model.linear_attn, "in_proj_z")


def test_sync_linear_attn_fused_projection_nvfp4_global_amax():
    model = _GatedDeltaNetProjectionToy()
    weight_cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        axis=None,
    )
    input_cfg = QuantizerAttributeConfig(
        num_bits=(2, 1),
        block_sizes={-1: 16, "type": "dynamic", "scale_bits": (4, 3)},
        axis=None,
    )
    _attach_quantizers(model.linear_attn.in_proj_qkv, weight_cfg, input_cfg)
    _attach_quantizers(model.linear_attn.in_proj_z, weight_cfg, input_cfg)
    NVFP4StaticQuantizer.from_tensor_quantizer(
        model.linear_attn.in_proj_qkv.weight_quantizer, global_amax=torch.tensor(13.0)
    )
    NVFP4StaticQuantizer.from_tensor_quantizer(
        model.linear_attn.in_proj_z.weight_quantizer, global_amax=torch.tensor(17.0)
    )
    model.linear_attn.in_proj_qkv.input_quantizer.amax = torch.tensor(19.0)
    model.linear_attn.in_proj_z.input_quantizer.amax = torch.tensor(23.0)

    synced = sync_linear_attn_fused_projection_amax(model)

    assert synced == 1
    assert torch.equal(
        model.linear_attn.in_proj_qkv.weight_quantizer.global_amax, torch.tensor(17.0)
    )
    assert torch.equal(model.linear_attn.in_proj_z.weight_quantizer.global_amax, torch.tensor(17.0))
    assert torch.equal(model.linear_attn.in_proj_qkv.input_quantizer.amax, torch.tensor(23.0))
    assert torch.equal(model.linear_attn.in_proj_z.input_quantizer.amax, torch.tensor(23.0))
    assert hasattr(model.linear_attn, "in_proj_qkv")
    assert hasattr(model.linear_attn, "in_proj_z")
