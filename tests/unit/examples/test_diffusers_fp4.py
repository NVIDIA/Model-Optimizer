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

import logging
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")
pytest.importorskip("diffusers")
from diffusers.models.attention_processor import Attention
from onnx import TensorProto, helper, numpy_helper

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.config import QuantizerAttributeConfig
from modelopt.torch.quantization.nn import TensorQuantizer

_QUANTIZATION_EXAMPLE = (
    Path(__file__).resolve().parents[3] / "examples" / "diffusers" / "quantization"
)
sys.path.insert(0, str(_QUANTIZATION_EXAMPLE))

from models_utils import ModelType
from quantize import Quantizer, _restore_sdxl_fp4_policy
from quantize_config import ModelConfig, QuantFormat, QuantizationConfig

from examples.diffusers.quantization.onnx_utils import export as diffusion_export


class _RecipeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)
        self.attn = nn.Module()
        self.attn.to_q = nn.Linear(16, 16, bias=False)
        self.attn.to_k = nn.Linear(16, 16, bias=False)
        self.attn.to_v = nn.Linear(16, 16, bias=False)
        self.conv = nn.Conv2d(4, 4, kernel_size=1, bias=False)


@pytest.mark.parametrize("model_type", [ModelType.SDXL_BASE, ModelType.SDXL_TURBO])
def test_sdxl_fp4_recipe(model_type):
    model = _RecipeBackbone()
    config = Quantizer(
        QuantizationConfig(format=QuantFormat.FP4),
        ModelConfig(model_type=model_type),
        logging.getLogger(__name__),
    ).get_quant_config(n_steps=1, backbone=model)

    mtq.replace_quant_module(model)
    mtq.set_quantizer_by_cfg(model, config["quant_cfg"])

    for quantizer in (model.linear.input_quantizer, model.linear.weight_quantizer):
        assert quantizer.is_enabled
        assert quantizer.is_nvfp4_dynamic
        assert quantizer.block_sizes[-1] == 16
    for projection in (model.attn.to_q, model.attn.to_k, model.attn.to_v):
        assert not projection.input_quantizer.is_enabled
        assert not projection.weight_quantizer.is_enabled
    for quantizer in (model.conv.input_quantizer, model.conv.weight_quantizer):
        assert quantizer.is_enabled
        assert quantizer.is_fp8


def test_restore_reapplies_sdxl_fp4_mha_policy():
    attention = Attention(query_dim=16, heads=1, dim_head=16)
    skipped_attention = Attention(query_dim=16, heads=1, dim_head=16)
    attention._disable_fp8_mha = True
    skipped_attention._disable_fp8_mha = True

    class _PipelineManager:
        def iter_backbones(self):
            return (("unet", attention), ("vae", skipped_attention))

    _restore_sdxl_fp4_policy(
        _PipelineManager(),
        QuantizationConfig(format=QuantFormat.FP4, quantize_mha=True),
        ModelType.SDXL_BASE,
    )

    assert not attention._disable_fp8_mha
    assert skipped_attention._disable_fp8_mha


def _fp8_quantizer(*, enabled=True):
    quantizer = TensorQuantizer(QuantizerAttributeConfig(num_bits=(4, 3), axis=None))
    quantizer.amax = torch.tensor(448.0)
    if not enabled:
        quantizer.disable()
    return quantizer


def _add_fp8_quantizers(module, *, enabled=True):
    module.input_quantizer = _fp8_quantizer(enabled=enabled)
    module.weight_quantizer = _fp8_quantizer(enabled=enabled)


@pytest.mark.parametrize("raises", [False, True])
def test_temporary_fp8_conv_export_scales_restore_state(raises):
    model = nn.Module()
    model.conv = nn.Conv2d(1, 1, 1)
    model.disabled_conv = nn.Conv2d(1, 1, 1)
    model.linear = nn.Linear(1, 1)
    _add_fp8_quantizers(model.conv)
    _add_fp8_quantizers(model.disabled_conv, enabled=False)
    _add_fp8_quantizers(model.linear)
    linear_only = nn.Sequential(nn.Linear(1, 1))
    _add_fp8_quantizers(linear_only[0])
    assert not diffusion_export._has_enabled_conv(linear_only)
    assert diffusion_export._has_enabled_conv(model)
    changed = (model.conv.input_quantizer, model.conv.weight_quantizer)
    unchanged = (
        model.disabled_conv.input_quantizer,
        model.disabled_conv.weight_quantizer,
        model.linear.input_quantizer,
        model.linear.weight_quantizer,
    )
    original_state = {
        quantizer: (quantizer._num_bits, quantizer._amax) for quantizer in changed + unchanged
    }

    def run_export():
        with diffusion_export._temporary_fp8_export_scales(model, conv_only=True):
            for quantizer in changed:
                assert quantizer.num_bits == 8
                assert quantizer.amax == 127.0
            for quantizer in unchanged:
                assert (quantizer._num_bits, quantizer._amax) == original_state[quantizer]
            if raises:
                raise RuntimeError("export failed")

    if raises:
        with pytest.raises(RuntimeError, match="export failed"):
            run_export()
    else:
        run_export()

    for quantizer, (num_bits, amax) in original_state.items():
        assert quantizer._num_bits == num_bits
        assert quantizer._amax is amax


def _make_mixed_fp4_fp8_model():
    fp4_weight = numpy_helper.from_array(
        np.linspace(-1.0, 1.0, 16 * 16, dtype=np.float16).reshape(16, 16), "fp4_weight"
    )
    fp8_weight = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float16), "fp8_weight")
    scale = numpy_helper.from_array(np.array(0.25, dtype=np.float16), "fp8_scale_value")
    zero = numpy_helper.from_array(np.array(0, dtype=np.int8), "fp8_zero_value")
    nodes = [
        helper.make_node(
            "TRT_FP4QDQ",
            ["fp4_weight"],
            ["fp4_weight_dq"],
            name="fp4_weight_qdq",
            domain="trt",
            block_size=16,
        ),
        helper.make_node(
            "MatMul", ["linear_input", "fp4_weight_dq"], ["linear_output"], name="fp4_matmul"
        ),
        helper.make_node("Constant", [], ["fp8_scale"], name="fp8_scale", value=scale),
        helper.make_node("Constant", [], ["fp8_zero"], name="fp8_zero", value=zero),
        helper.make_node(
            "QuantizeLinear",
            ["fp8_weight", "fp8_scale", "fp8_zero"],
            ["fp8_weight_q"],
            name="fp8_weight_quantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["fp8_weight_q", "fp8_scale", "fp8_zero"],
            ["fp8_weight_dq"],
            name="fp8_weight_dequantize",
        ),
        helper.make_node(
            "QuantizeLinear",
            ["conv_input", "fp8_scale", "fp8_zero"],
            ["conv_input_q"],
            name="fp8_activation_quantize",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_input_q", "fp8_scale", "fp8_zero"],
            ["conv_input_dq"],
            name="fp8_activation_dequantize",
        ),
        helper.make_node(
            "Conv", ["conv_input_dq", "fp8_weight_dq"], ["conv_output"], name="fp8_conv"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "mixed_fp4_fp8",
        [
            helper.make_tensor_value_info("linear_input", TensorProto.FLOAT16, [1, 16]),
            helper.make_tensor_value_info("conv_input", TensorProto.FLOAT16, [1, 1, 2, 2]),
        ],
        [
            helper.make_tensor_value_info("linear_output", TensorProto.FLOAT16, [1, 16]),
            helper.make_tensor_value_info("conv_output", TensorProto.FLOAT16, [1, 1, 2, 2]),
        ],
        [fp4_weight, fp8_weight],
        value_info=[helper.make_tensor_value_info("fp4_weight_dq", TensorProto.FLOAT16, [16, 16])],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("trt", 1)],
    )


def _constant_dtype(model, name):
    node = next(node for node in model.graph.node if node.name == name)
    return next(attribute for attribute in node.attribute if attribute.name == "value").t.data_type


def test_mixed_sdxl_fp4_graph_postprocessing():
    model = diffusion_export._process_fp4_onnx_graph(_make_mixed_fp4_fp8_model(), "sdxl-1.0")

    assert not any(node.op_type == "TRT_FP4QDQ" for node in model.graph.node)
    assert any(tensor.data_type == TensorProto.FLOAT4E2M1 for tensor in model.graph.initializer)
    assert _constant_dtype(model, "fp8_zero") == TensorProto.FLOAT8E4M3FN
    assert any(node.op_type == "Conv" and node.name == "fp8_conv" for node in model.graph.node)
    assert sum(node.op_type == "QuantizeLinear" for node in model.graph.node) == 2
    assert next(opset.version for opset in model.opset_import if not opset.domain) >= 23
    onnx.checker.check_model(model)
