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

"""Check scale preservation and FP4 double-DQ typing with real ONNX validation."""

import copy

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto as T
from onnx import helper, numpy_helper

from modelopt.onnx.export.nvfp4_dtype import canonicalize_nvfp4_fp16
from modelopt.torch._deploy.utils.torch_onnx import _prepare_nvfp4_fp16_export
from modelopt.torch.quantization.nn import TensorQuantizer


@pytest.mark.parametrize("wrapped", [False, True])
def test_double_dq_preserves_fp32_scale(wrapped):
    """Exercise the exported dtype contract without bypassing validation."""
    scale = numpy_helper.from_array(np.array(1e-8, dtype=np.float32), "global")
    fp8 = helper.make_tensor("fp8", T.FLOAT8E4M3FN, [1], bytes([56]), raw=True)
    fp4 = helper.make_tensor("fp4", T.FLOAT4E2M1, [16], bytes([0x22] * 8), raw=True)
    nodes = []
    source_scale = "global"
    if wrapped:
        nodes.append(
            helper.make_node("Cast", ["global"], ["half_scale"], name="scale_cast", to=T.FLOAT16)
        )
        source_scale = "half_scale"
    nodes.extend(
        [
            helper.make_node(
                "DequantizeLinear", ["fp8", source_scale], ["blocks"], name="scale_dq"
            ),
            helper.make_node(
                "Cast",
                ["blocks"],
                ["cast_blocks"],
                name="unused_scale_cast",
                to=T.FLOAT,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["fp4", "cast_blocks"],
                ["out"],
                name="data_dq",
                block_size=16,
                axis=0,
            ),
        ]
    )
    graph = helper.make_graph(
        nodes,
        "double_dq",
        [],
        [helper.make_tensor_value_info("out", T.FLOAT, [16])],
        [scale, fp8, fp4],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    result = canonicalize_nvfp4_fp16(model)
    onnx.checker.check_model(result, full_check=True)
    onnx.shape_inference.infer_shapes(result, strict_mode=True, check_type=True)
    assert result.graph.output[0].type.tensor_type.elem_type == T.FLOAT16
    assert [node.op_type for node in result.graph.node] == [
        "DequantizeLinear",
        "DequantizeLinear",
    ]
    literal = next(
        t for t in result.graph.initializer if t.name == "scale_dq_global_scale_fp32_literal"
    )
    assert literal.data_type == T.FLOAT
    assert numpy_helper.to_array(literal).tobytes() == numpy_helper.to_array(scale).tobytes()


def test_prepare_retains_tiny_quantizer_buffers():
    """Exercise the exported dtype contract without bypassing validation."""
    model = torch.nn.Sequential(torch.nn.Linear(16, 16), TensorQuantizer())
    model[1].register_buffer("_test_scale", torch.tensor(1e-10))
    before = model[1]._test_scale.clone()
    _prepare_nvfp4_fp16_export(model)
    assert model[0].weight.dtype == torch.float16
    assert model[0].bias.dtype == torch.float16
    assert model[1]._test_scale.dtype == torch.float32
    assert torch.equal(model[1]._test_scale, before)


def test_preconverted_matmul_drops_stale_fp32_annotation():
    """Exercise the exported dtype contract without bypassing validation."""
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "weight"], ["product"], name="matmul"),
            helper.make_node("Add", ["product", "bias"], ["out"], name="add"),
        ],
        "stale_annotation",
        [helper.make_tensor_value_info("x", T.FLOAT16, [1, 2])],
        [helper.make_tensor_value_info("out", T.FLOAT16, [1, 2])],
        [
            numpy_helper.from_array(np.ones((2, 2), dtype=np.float16), "weight"),
            numpy_helper.from_array(np.ones(2, dtype=np.float16), "bias"),
        ],
        value_info=[helper.make_tensor_value_info("product", T.FLOAT, [1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])
    with pytest.raises(onnx.shape_inference.InferenceError):
        onnx.checker.check_model(model, full_check=True)
    canonicalize_nvfp4_fp16(model)
    onnx.checker.check_model(model, full_check=True)
    onnx.shape_inference.infer_shapes(model, strict_mode=True, check_type=True)


def test_prepare_rejects_unvalidated_bf16_without_mutation():
    """Exercise the exported dtype contract without bypassing validation."""
    model = torch.nn.Linear(16, 16).bfloat16()
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="FP32/FP16"):
        _prepare_nvfp4_fp16_export(model)
    assert all(torch.equal(value, before[name]) for name, value in model.state_dict().items())
