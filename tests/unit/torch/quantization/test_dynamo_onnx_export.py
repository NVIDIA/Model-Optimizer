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

"""CPU contract tests for the supported Dynamo ONNX export path."""

import inspect

import pytest
import torch
from torch import nn

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

from modelopt.torch.quantization._dynamo_onnx import _get_dynamo_onnx_translation_table

_NVFP4_ARGS = (4, 2, 8, 4, "Float")
_MXFP8_ARGS = (8, 4, 9, 8, "Float")


class _StrictQuantOp(nn.Module):
    def __init__(self, quant_format):
        super().__init__()
        self.quant_format = quant_format

    def forward(self, x, amax):
        if self.quant_format.startswith("fp8"):
            return torch.ops.tensorrt.quantize_op.default(
                x,
                None if self.quant_format == "fp8_none" else amax,
                8,
                4,
                False,
                False,
                "Float",
            )
        if self.quant_format in {"int8", "uint8"}:
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 8, 0, self.quant_format == "uint8", False, "Float", None, 0
            )
        if self.quant_format.startswith("nvfp4"):
            return torch.ops.tensorrt.dynamic_block_quantize_op.default(
                x, 16, amax, *_NVFP4_ARGS, self.quant_format.removeprefix("nvfp4_")
            )
        return torch.ops.tensorrt.dynamic_block_quantize_op.overload(
            x, 32, None, *_MXFP8_ARGS, self.quant_format.removeprefix("mxfp8_")
        )


def _assert_valid_without_custom_ops(model):
    onnx.checker.check_model(model, full_check=True)
    assert not model.functions
    assert not any(
        node.domain == "tensorrt" or node.op_type in {"quantize_op", "dynamic_block_quantize_op"}
        for node in model.graph.node
    )
    return model


def _raw_export(model, inputs, path, opset=23):
    compatibility_options = (
        {"fallback": False} if "fallback" in inspect.signature(torch.onnx.export).parameters else {}
    )
    torch.onnx.export(
        model,
        inputs,
        path,
        dynamo=True,
        opset_version=opset,
        custom_translation_table=_get_dynamo_onnx_translation_table(),
        **compatibility_options,
    )
    exported = onnx.load(path)
    return _assert_valid_without_custom_ops(exported)


def _attribute(node, name):
    return onnx.helper.get_attribute_value(
        next(attr for attr in node.attribute if attr.name == name)
    )


def _constant(model, name):
    initializer = next((item for item in model.graph.initializer if item.name == name), None)
    if initializer is not None:
        return onnx.numpy_helper.to_array(initializer)
    node = next(
        item for item in model.graph.node if item.op_type == "Constant" and name in item.output
    )
    return onnx.numpy_helper.to_array(_attribute(node, "value"))


_STRICT_CASES = [
    ("fp8_none", ("trt", "TRT_FP8QuantizeLinear")),
    ("fp8", ("trt", "TRT_FP8QuantizeLinear")),
    ("uint8", ("", "QuantizeLinear")),
    ("mxfp8_static", ("trt", "TRT_MXFP8DequantizeLinear")),
]


@pytest.mark.parametrize(("quant_format", "expected_node"), _STRICT_CASES)
def test_private_table_strict_formats(tmp_path, quant_format, expected_node):
    sample_input = torch.randn(4, 32)
    amax = torch.ones(4, 1) if quant_format == "uint8" else torch.tensor(1.0)
    exported_program = torch.export.export(
        _StrictQuantOp(quant_format), (sample_input, amax), strict=True
    )
    exported = _raw_export(exported_program, (), tmp_path / f"{quant_format}.onnx")
    assert expected_node in {(node.domain, node.op_type) for node in exported.graph.node}
    if quant_format == "uint8":
        quantizer = next(node for node in exported.graph.node if node.op_type == "QuantizeLinear")
        tensor_types = {
            value.name: value.type.tensor_type.elem_type
            for value in (*exported.graph.input, *exported.graph.value_info, *exported.graph.output)
        }
        assert _attribute(quantizer, "axis") == 0
        assert tensor_types[quantizer.input[2]] == onnx.TensorProto.UINT8
        divisor = next(node for node in exported.graph.node if node.op_type == "Div").input[1]
        assert _constant(exported, divisor).item() == 255.0
    elif quant_format == "fp8":
        divisor = next(node for node in exported.graph.node if node.op_type == "Div").input[1]
        assert _constant(exported, divisor).item() == 448.0


def test_private_table_preserves_nvfp4_carrier_dtype(tmp_path):
    inputs = torch.randn(4, 32, dtype=torch.float16)
    exported_program = torch.export.export(
        _StrictQuantOp("nvfp4_dynamic"), (inputs, torch.tensor(1.0)), strict=True
    )

    exported = _raw_export(exported_program, (), tmp_path / "nvfp4_dtype.onnx", 23)

    assert exported.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert {node.domain for node in exported.graph.node if node.op_type == "DequantizeLinear"} == {
        ""
    }


def test_private_table_rejects_int8_axis_disagreement(tmp_path):
    inputs = torch.randn(2, 3, 4)
    amax = torch.ones(1, 3, 1)
    exported_program = torch.export.export(_StrictQuantOp("int8"), (inputs, amax), strict=True)

    with pytest.raises(torch.onnx.OnnxExporterError, match="axis 0 does not match amax axis 1"):
        _raw_export(exported_program, (), tmp_path / "int8_axis.onnx", 23)


@pytest.mark.parametrize(
    ("fmt", "inputs", "amax", "message"),
    [
        ("fp8", torch.randn(2, 4), torch.ones(2, 1), "scalar amax"),
        ("int8", torch.randn(2, 3, 4), torch.ones(2, 1, 4), "multi-axis"),
        ("nvfp4_dynamic", torch.randn(1, 2, 3, 32), torch.tensor(1.0), "rank 2 or 3"),
    ],
)
def test_private_table_rejects_shapes(tmp_path, fmt, inputs, amax, message):
    exported_program = torch.export.export(_StrictQuantOp(fmt), (inputs, amax), strict=True)
    with pytest.raises(torch.onnx.OnnxExporterError, match=message):
        _raw_export(exported_program, (), tmp_path / f"{fmt}.onnx")
