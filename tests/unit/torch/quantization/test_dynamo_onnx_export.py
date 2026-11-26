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
    def __init__(self, quant_format, axis=0):
        super().__init__()
        self.quant_format = quant_format
        self.axis = axis

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
                x, amax, 8, 0, self.quant_format == "uint8", False, "Float", None, self.axis
            )
        if self.quant_format == "int4":
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 4, 0, False, True, "Float", 16, -1
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
    ("int8", ("", "QuantizeLinear")),
    ("uint8", ("", "QuantizeLinear")),
    ("int4", ("trt", "DequantizeLinear")),
    ("nvfp4_static", ("trt", "TRT_FP4QDQ")),
    ("mxfp8_static", ("trt", "TRT_MXFP8DequantizeLinear")),
    ("mxfp8_dynamic", ("trt", "TRT_MXFP8DynamicQuantize")),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(("quant_format", "expected_node"), _STRICT_CASES)
def test_private_table_strict_formats(tmp_path, quant_format, expected_node, dtype):
    sample_input = torch.randn(4, 32, dtype=dtype)
    if quant_format in {"int8", "uint8"}:
        amax = torch.ones(4, 1)
    elif quant_format == "int4":
        amax = torch.ones(4, 2)
    else:
        amax = torch.tensor(1.0)
    exported_program = torch.export.export(
        _StrictQuantOp(quant_format), (sample_input, amax), strict=True
    )
    exported = _raw_export(exported_program, (), tmp_path / f"{quant_format}.onnx")
    assert expected_node in {(node.domain, node.op_type) for node in exported.graph.node}
    expected_dtype = onnx.TensorProto.FLOAT if dtype == torch.float32 else onnx.TensorProto.FLOAT16
    assert exported.graph.output[0].type.tensor_type.elem_type == expected_dtype
    tensor_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*exported.graph.input, *exported.graph.value_info, *exported.graph.output)
    }
    if quant_format in {"int8", "uint8"}:
        quantizer = next(node for node in exported.graph.node if node.op_type == "QuantizeLinear")
        assert _attribute(quantizer, "axis") == 0
        unsigned = quant_format == "uint8"
        assert tensor_types[quantizer.input[2]] == (
            onnx.TensorProto.UINT8 if unsigned else onnx.TensorProto.INT8
        )
        divisor = next(node for node in exported.graph.node if node.op_type == "Div").input[1]
        assert _constant(exported, divisor).item() == (255.0 if unsigned else 127.0)
    elif quant_format == "int4":
        dequantizer = next(
            node for node in exported.graph.node if node.op_type == "DequantizeLinear"
        )
        assert _attribute(dequantizer, "axis") == -1
        assert _attribute(dequantizer, "block_size") == 16
        assert tensor_types[dequantizer.output[0]] == onnx.TensorProto.FLOAT
        divisor = next(node for node in exported.graph.node if node.op_type == "Div").input[1]
        assert _constant(exported, divisor).item() == 7.0
    elif quant_format == "fp8":
        divisor = next(node for node in exported.graph.node if node.op_type == "Div").input[1]
        assert _constant(exported, divisor).item() == 448.0


@pytest.mark.parametrize("amax_shape", [(), (1,), (1, 1)])
@pytest.mark.parametrize("axis", [0, -1])
def test_private_table_int8_singleton_amax_is_per_tensor(tmp_path, amax_shape, axis):
    inputs = torch.randn(4, 32)
    exported_program = torch.export.export(
        _StrictQuantOp("int8", axis), (inputs, torch.ones(amax_shape)), strict=True
    )
    exported = _raw_export(exported_program, (), tmp_path / "int8_per_tensor.onnx")
    tensor_shapes = {
        value.name: value.type.tensor_type.shape
        for value in (*exported.graph.input, *exported.graph.value_info, *exported.graph.output)
    }
    for node in exported.graph.node:
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}:
            assert len(tensor_shapes[node.input[1]].dim) == 0
            assert len(tensor_shapes[node.input[2]].dim) == 0


def test_private_table_rejects_int4_without_amax(tmp_path):
    exported_program = torch.export.export(
        _StrictQuantOp("int4"), (torch.randn(4, 32), None), strict=True
    )
    with pytest.raises(torch.onnx.OnnxExporterError, match="INT4 ONNX export requires amax"):
        _raw_export(exported_program, (), tmp_path / "int4_without_amax.onnx")


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
