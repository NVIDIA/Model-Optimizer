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

"""CPU tests for Dynamo ONNX export of ModelOpt quantization operators."""

import copy

import onnx
import pytest
import torch
from torch import nn

pytest.importorskip("onnxscript")

import modelopt.torch.quantization as mtq
from modelopt.onnx.export import MXFP8QuantExporter
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization.export_onnx import get_dynamo_onnx_translation_table
from modelopt.torch.quantization.tensor_quant import (
    dynamic_block_quant,
    fake_tensor_quant,
    scaled_e4m3,
)


class _TwoLinear(nn.Module):
    def __init__(self, features=16):
        super().__init__()
        self.linear0 = nn.Linear(features, features, bias=False)
        self.linear1 = nn.Linear(features, features, bias=False)

    def forward(self, x):
        return self.linear1(torch.nn.functional.silu(self.linear0(x)))


class _Linear(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 128, bias=False)

    def forward(self, x):
        return self.linear(x)


class _BiasedQuant(nn.Module):
    def __init__(self, quant_format):
        super().__init__()
        self.quant_format = quant_format

    def forward(self, x, amax, bias):
        if self.quant_format == "fp8":
            return scaled_e4m3(x, amax, bias, 4, 3, "Float", False)
        return fake_tensor_quant(x, amax, bias, 8, False, False, "Float", False, None, None)


def _all_nodes(model):
    yield from model.graph.node
    for function in model.functions:
        yield from function.node


def _node_attributes(node):
    return {
        attribute.name: onnx.helper.get_attribute_value(attribute) for attribute in node.attribute
    }


def _tensor_metadata(model, name):
    for value in (*model.graph.input, *model.graph.value_info, *model.graph.output):
        if value.name == name:
            tensor_type = value.type.tensor_type
            shape = [dimension.dim_value for dimension in tensor_type.shape.dim]
            return tensor_type.elem_type, shape
    for initializer in model.graph.initializer:
        if initializer.name == name:
            return initializer.data_type, list(initializer.dims)
    raise AssertionError(f"Missing type information for {name}")


def _scalar_initializer_values(model):
    values = set()
    for initializer in model.graph.initializer:
        value = onnx.numpy_helper.to_array(initializer)
        if value.size == 1:
            values.add(float(value.item()))
    return values


def _auto_quantize_fp8(model, sample_input):
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": 8.0},
        quantization_formats=[copy.deepcopy(mtq.FP8_DEFAULT_CFG)],
        data_loader=[sample_input],
        forward_step=lambda candidate, batch: candidate(batch),
        loss_func=lambda output, _batch: output.float().square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
    )[0]


@pytest.mark.timeout(120)
def test_autoquant_fp8_dynamo_export_without_translation_table(tmp_path):
    sample_input = torch.randn(2, 16)
    model = _auto_quantize_fp8(_TwoLinear().eval(), sample_input)
    onnx_path = tmp_path / "two_linear_fp8.onnx"

    torch.onnx.export(
        model,
        (sample_input,),
        onnx_path,
        dynamo=True,
        opset_version=24,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    assert not exported.functions
    opset_imports = {opset.domain: opset.version for opset in exported.opset_import}
    assert opset_imports[""] == 24
    assert opset_imports["trt"] == 1
    nodes = list(_all_nodes(exported))
    assert sum(node.op_type == "TRT_FP8QuantizeLinear" for node in nodes) == 4
    assert sum(node.op_type == "TRT_FP8DequantizeLinear" for node in nodes) == 4
    for node in nodes:
        if node.op_type == "TRT_FP8QuantizeLinear":
            assert _tensor_metadata(exported, node.output[0])[0] == onnx.TensorProto.UINT8
    assert not any(node.op_type == "quantize_op" for node in nodes)

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        model_name="two_linear_fp8",
        dynamo_export=True,
        onnx_opset=24,
    )
    processed = onnx.load_model_from_string(
        OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    )
    onnx.checker.check_model(processed)
    processed_nodes = list(_all_nodes(processed))
    assert {"QuantizeLinear", "DequantizeLinear"} <= {node.op_type for node in processed_nodes}
    assert not any(
        node.op_type.startswith("TRT_FP8") or node.op_type == "quantize_op"
        for node in processed_nodes
    )


@pytest.mark.timeout(120)
def test_mixed_autoquant_dynamo_export_without_translation_table(tmp_path):
    sample_input = torch.randn(2, 128)
    model = mtq.auto_quantize(
        _TwoLinear(128).eval(),
        constraints={"effective_bits": 6.0},
        quantization_formats=[
            copy.deepcopy(mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG),
            copy.deepcopy(mtq.FP8_DEFAULT_CFG),
        ],
        data_loader=[sample_input],
        forward_step=lambda candidate, batch: candidate(batch),
        loss_func=lambda output, _batch: output.float().square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
    )[0]
    onnx_path = tmp_path / "two_linear_mixed.onnx"

    torch.onnx.export(
        model,
        (sample_input,),
        onnx_path,
        dynamo=True,
        opset_version=24,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    assert not exported.functions
    nodes = list(_all_nodes(exported))
    assert sum(node.op_type == "TRT_FP8QuantizeLinear" for node in nodes) == 2
    assert sum(node.op_type == "TRT_FP8DequantizeLinear" for node in nodes) == 2
    assert sum(node.domain == "trt" and node.op_type == "DequantizeLinear" for node in nodes) == 1
    assert not any(node.domain == "tensorrt" for node in nodes)

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        model_name="two_linear_mixed",
        dynamo_export=True,
        onnx_opset=24,
        weights_dtype="fp32",
    )
    onnx_package = OnnxBytes.from_bytes(onnx_bytes)
    export_dir = tmp_path / "mixed_helper"
    onnx_package.write_to_disk(str(export_dir))
    processed = onnx.load(
        export_dir / f"{onnx_package.model_name}.onnx",
        load_external_data=True,
    )

    onnx.checker.check_model(processed, full_check=True)
    assert not processed.functions
    processed_nodes = list(_all_nodes(processed))
    assert (
        sum(node.domain == "" and node.op_type == "QuantizeLinear" for node in processed_nodes) == 1
    )
    assert (
        sum(node.domain == "" and node.op_type == "DequantizeLinear" for node in processed_nodes)
        == 2
    )
    int4_dq_nodes = [
        node
        for node in processed_nodes
        if node.domain == "trt" and node.op_type == "DequantizeLinear"
    ]
    assert len(int4_dq_nodes) == 1
    initializers = {initializer.name: initializer for initializer in processed.graph.initializer}
    assert initializers[int4_dq_nodes[0].input[0]].data_type == onnx.TensorProto.INT4
    assert any(
        node.domain == ""
        and node.op_type == "DequantizeLinear"
        and node.input[0] in initializers
        and initializers[node.input[0]].data_type == onnx.TensorProto.FLOAT8E4M3FN
        for node in processed_nodes
    )
    assert not any(node.domain == "tensorrt" for node in processed_nodes)
    assert not any(node.op_type.startswith("TRT_") for node in processed_nodes)
    assert not any(node.op_type == "quantize_op" for node in processed_nodes)


@pytest.mark.parametrize(
    ("quant_format", "quantize_op", "dequantize_op"),
    [
        ("fp8", "TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear"),
        ("int8", "QuantizeLinear", "DequantizeLinear"),
    ],
)
def test_affine_bias_surrounds_dynamo_qdq(tmp_path, quant_format, quantize_op, dequantize_op):
    args = (torch.randn(4, 32), torch.tensor(2.0), torch.full((32,), 0.25))

    for capture in ("direct", "strict"):
        model = _BiasedQuant(quant_format).eval()
        export_args = args
        export_kwargs = {}
        if capture == "strict":
            model = torch.export.export(model, args, strict=True)
            export_args = ()
            export_kwargs["custom_translation_table"] = get_dynamo_onnx_translation_table()
        onnx_path = tmp_path / f"{quant_format}_{capture}_bias.onnx"

        torch.onnx.export(
            model,
            export_args,
            onnx_path,
            dynamo=True,
            opset_version=24,
            **export_kwargs,
        )

        exported = onnx.load(onnx_path)
        onnx.checker.check_model(exported, full_check=True)
        nodes = list(exported.graph.node)
        subtract = next(node for node in nodes if node.op_type == "Sub")
        quantize = next(node for node in nodes if node.op_type == quantize_op)
        dequantize = next(node for node in nodes if node.op_type == dequantize_op)
        add = next(node for node in nodes if node.op_type == "Add")
        assert quantize.input[0] == subtract.output[0]
        assert dequantize.input[0] == quantize.output[0]
        assert dequantize.output[0] in add.input
        if quant_format == "fp8":
            assert 448.0 in _scalar_initializer_values(exported)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("in_features", [128, 256])
def test_int4_awq_helper_dynamo_export(tmp_path, in_features):
    sample_input = torch.randn(2, in_features)
    model = _Linear(in_features).eval()
    model = mtq.quantize(
        model,
        copy.deepcopy(mtq.INT4_AWQ_CFG),
        forward_loop=lambda candidate: candidate(sample_input),
    )

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        model_name=f"linear_int4_{in_features}",
        dynamo_export=True,
        onnx_opset=24,
        weights_dtype="fp32",
    )
    onnx_package = OnnxBytes.from_bytes(onnx_bytes)
    export_dir = tmp_path / str(in_features)
    onnx_package.write_to_disk(str(export_dir))
    exported = onnx.load(
        export_dir / f"{onnx_package.model_name}.onnx",
        load_external_data=True,
    )

    onnx.checker.check_model(exported)
    int4_dq_nodes = [
        node
        for node in exported.graph.node
        if node.domain == "trt" and node.op_type == "DequantizeLinear"
    ]
    assert len(int4_dq_nodes) == 1
    int4_dq = int4_dq_nodes[0]
    initializers = {initializer.name: initializer for initializer in exported.graph.initializer}
    weight = initializers[int4_dq.input[0]]
    scale = initializers[int4_dq.input[1]]
    assert weight.data_type == onnx.TensorProto.INT4
    assert list(weight.dims) == [128, in_features]
    assert list(scale.dims) == [128, in_features // 128]

    attributes = {
        attribute.name: onnx.helper.get_attribute_value(attribute)
        for attribute in int4_dq.attribute
    }
    assert attributes["axis"] == 1
    assert attributes["block_size"] == 128
    dq_output = next(
        value_info
        for value_info in exported.graph.value_info
        if value_info.name == int4_dq.output[0]
    )
    assert [dimension.dim_value for dimension in dq_output.type.tensor_type.shape.dim] == [
        128,
        in_features,
    ]
    assert not any(node.op_type == "Reshape" for node in exported.graph.node)
    gemm = next(node for node in exported.graph.node if node.op_type == "Gemm")
    assert gemm.input[1] == int4_dq.output[0]
    assert not any(node.op_type == "quantize_op" for node in _all_nodes(exported))


class _DirectNVFP4(nn.Module):
    def __init__(self, trt_high_precision_dtype="Float"):
        super().__init__()
        self.trt_high_precision_dtype = trt_high_precision_dtype

    def forward(self, x, amax):
        return dynamic_block_quant(
            x,
            16,
            amax,
            None,
            (2, 1),
            (4, 3),
            self.trt_high_precision_dtype,
            "dynamic",
            True,
        )


class _DirectMXFP8(nn.Module):
    def forward(self, x):
        return dynamic_block_quant(
            x,
            32,
            None,
            None,
            (4, 3),
            (8, 0),
            None,
            "dynamic",
            True,
        )


def test_nvfp4_direct_dynamo_export_has_logical_float4_shape(tmp_path):
    sample_input = torch.randn(4, 32)
    onnx_path = tmp_path / "nvfp4_dynamic.onnx"

    torch.onnx.export(
        _DirectNVFP4().eval(),
        (sample_input, torch.tensor(1.0)),
        onnx_path,
        dynamo=True,
        opset_version=24,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    fp4_node = next(
        node for node in exported.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    )
    assert _node_attributes(fp4_node) == {
        "axis": -1,
        "block_size": 16,
        "scale_type": onnx.TensorProto.FLOAT8E4M3FN,
    }
    assert _tensor_metadata(exported, fp4_node.output[0]) == (
        onnx.TensorProto.FLOAT4E2M1,
        list(sample_input.shape),
    )
    assert _tensor_metadata(exported, fp4_node.output[1]) == (
        onnx.TensorProto.FLOAT8E4M3FN,
        [sample_input.shape[0], sample_input.shape[1] // 16],
    )
    fp4_dq = next(
        node
        for node in exported.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == fp4_node.output[0]
    )
    assert fp4_dq.domain == "trt"
    assert _node_attributes(fp4_dq)["axis"] == -1
    assert _node_attributes(fp4_dq)["block_size"] == 16
    assert _tensor_metadata(exported, fp4_dq.output[0]) == (
        onnx.TensorProto.FLOAT,
        list(sample_input.shape),
    )
    assert 2688.0 in _scalar_initializer_values(exported)
    assert any(node.op_type == "Where" for node in exported.graph.node)


@pytest.mark.parametrize("capture", ["direct", "strict"])
def test_nvfp4_dynamic_opset21_full_check(tmp_path, capture):
    sample_input = torch.randn(4, 32)
    model = _DirectNVFP4().eval()
    args = (sample_input, torch.tensor(1.0))
    export_kwargs = {}
    if capture == "strict":
        model = torch.export.export(model, args, strict=True)
        args = ()
        export_kwargs["custom_translation_table"] = get_dynamo_onnx_translation_table()
    onnx_path = tmp_path / f"nvfp4_opset21_{capture}.onnx"

    torch.onnx.export(
        model,
        args,
        onnx_path,
        dynamo=True,
        opset_version=21,
        **export_kwargs,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    quantize = next(
        node for node in exported.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    )
    scale_dq = next(
        node
        for node in exported.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == quantize.output[1]
    )
    fp4_dq = next(
        node
        for node in exported.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == quantize.output[0]
    )
    assert scale_dq.domain == ""
    assert fp4_dq.domain == "trt"
    assert _tensor_metadata(exported, quantize.output[0])[0] == onnx.TensorProto.FLOAT4E2M1


class _StrictNVFP4(nn.Module):
    def __init__(self, trt_high_precision_dtype, onnx_quantizer_type):
        super().__init__()
        self.trt_high_precision_dtype = trt_high_precision_dtype
        self.onnx_quantizer_type = onnx_quantizer_type

    def forward(self, x, amax):
        return torch.ops.tensorrt.dynamic_block_quantize_op.default(
            x,
            16,
            amax,
            4,
            2,
            8,
            4,
            self.trt_high_precision_dtype,
            self.onnx_quantizer_type,
        )


_NVFP4_DTYPES = [
    pytest.param("Float", torch.float32, onnx.TensorProto.FLOAT, id="float"),
    pytest.param("Half", torch.float16, onnx.TensorProto.FLOAT16, id="half"),
    pytest.param("BFloat16", torch.bfloat16, onnx.TensorProto.BFLOAT16, id="bfloat16"),
]


@pytest.mark.parametrize("capture", ["direct", "strict"])
@pytest.mark.parametrize(("trt_high_precision_dtype", "torch_dtype", "onnx_dtype"), _NVFP4_DTYPES)
def test_nvfp4_dynamic_dynamo_output_dtype(
    tmp_path, capture, trt_high_precision_dtype, torch_dtype, onnx_dtype
):
    sample_input = torch.randn(4, 32, dtype=torch_dtype)
    model = _DirectNVFP4(trt_high_precision_dtype).eval()
    args = (sample_input, torch.tensor(1.0))
    if capture == "strict":
        model = torch.export.export(
            _StrictNVFP4(trt_high_precision_dtype, "dynamic"), args, strict=True
        )
        args = ()
    onnx_path = tmp_path / f"nvfp4_{capture}_{trt_high_precision_dtype}.onnx"

    torch.onnx.export(
        model,
        args,
        onnx_path,
        dynamo=True,
        opset_version=24,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    fp4_node = next(
        node for node in exported.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    )
    assert _node_attributes(fp4_node) == {
        "axis": -1,
        "block_size": 16,
        "scale_type": onnx.TensorProto.FLOAT8E4M3FN,
    }
    assert _tensor_metadata(exported, fp4_node.output[0])[0] == onnx.TensorProto.FLOAT4E2M1
    assert _tensor_metadata(exported, fp4_node.output[1])[0] == onnx.TensorProto.FLOAT8E4M3FN
    fp4_dq = next(
        node
        for node in exported.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == fp4_node.output[0]
    )
    assert fp4_dq.domain == "trt"
    assert _node_attributes(fp4_dq)["axis"] == -1
    assert _node_attributes(fp4_dq)["block_size"] == 16
    assert _tensor_metadata(exported, exported.graph.output[0].name) == (
        onnx_dtype,
        list(sample_input.shape),
    )
    assert 2688.0 in _scalar_initializer_values(exported)
    assert any(node.op_type == "Where" for node in exported.graph.node)


@pytest.mark.parametrize("capture", ["direct", "strict"])
def test_mxfp8_dynamic_dynamo_contract(tmp_path, capture):
    sample_input = torch.randn(4, 64)
    model = _DirectMXFP8().eval()
    args = (sample_input,)
    export_kwargs = {}
    if capture == "strict":
        model = torch.export.export(model, args, strict=True)
        args = ()
        export_kwargs["custom_translation_table"] = get_dynamo_onnx_translation_table()
    onnx_path = tmp_path / f"mxfp8_{capture}.onnx"

    torch.onnx.export(
        model,
        args,
        onnx_path,
        dynamo=True,
        opset_version=24,
        **export_kwargs,
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    opset_imports = {opset.domain: opset.version for opset in exported.opset_import}
    assert opset_imports == {"": 24, "trt": 1}
    quantize = next(
        node for node in exported.graph.node if node.op_type == "TRT_MXFP8DynamicQuantize"
    )
    dequantize = next(
        node for node in exported.graph.node if node.op_type == "TRT_MXFP8DequantizeLinear"
    )
    assert _node_attributes(quantize) == {
        "axis": -1,
        "block_size": 32,
        "output_dtype": onnx.TensorProto.FLOAT8E4M3FN,
    }
    assert _node_attributes(dequantize) == {
        "axis": -1,
        "block_size": 32,
        "output_dtype": onnx.TensorProto.FLOAT,
    }
    assert list(dequantize.input[:2]) == list(quantize.output)
    assert _tensor_metadata(exported, exported.graph.output[0].name) == (
        onnx.TensorProto.FLOAT,
        list(sample_input.shape),
    )
    if capture == "direct":
        assert _tensor_metadata(exported, quantize.output[0]) == (
            onnx.TensorProto.FLOAT8E4M3FN,
            list(sample_input.shape),
        )
        assert _tensor_metadata(exported, quantize.output[1]) == (
            onnx.TensorProto.UINT8,
            [sample_input.shape[0], sample_input.shape[1] // 32],
        )


def test_nvfp4_strict_none_quantizer_type_uses_static_marker(tmp_path):
    sample_input = torch.randn(4, 32)
    exported_program = torch.export.export(
        _StrictNVFP4("Float", None),
        (sample_input, torch.tensor(1.0)),
        strict=True,
    )
    onnx_path = tmp_path / "nvfp4_static_default.onnx"

    torch.onnx.export(
        exported_program,
        (),
        onnx_path,
        dynamo=True,
        opset_version=24,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    opset_imports = {opset.domain: opset.version for opset in exported.opset_import}
    assert opset_imports == {"": 24, "trt": 1}
    node_types = {node.op_type for node in exported.graph.node}
    assert "TRT_FP4QDQ" in node_types
    assert "TRT_FP4DynamicQuantize" not in node_types


class _StrictQuantOp(nn.Module):
    def __init__(self, quant_format):
        super().__init__()
        self.quant_format = quant_format

    def forward(self, x, amax):
        if self.quant_format == "fp8":
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 8, 4, False, False, "Float", None, None
            )
        if self.quant_format == "int8":
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 8, 0, False, False, "Float", None, 0
            )
        if self.quant_format == "int4":
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 4, 0, False, True, "Float", 32, 0
            )
        if self.quant_format.startswith("nvfp4"):
            quantizer_type = self.quant_format.removeprefix("nvfp4_")
            return torch.ops.tensorrt.dynamic_block_quantize_op.default(
                x, 16, amax, 4, 2, 8, 4, "Float", quantizer_type
            )
        if self.quant_format.startswith("mxfp8"):
            quantizer_type = self.quant_format.removeprefix("mxfp8_")
            return torch.ops.tensorrt.dynamic_block_quantize_op.overload(
                x, 32, None, 8, 4, 9, 8, "Float", quantizer_type
            )
        raise AssertionError(f"Unknown format: {self.quant_format}")


class _StrictINT8(nn.Module):
    def __init__(self, unsigned):
        super().__init__()
        self.unsigned = unsigned

    def forward(self, x, amax):
        return torch.ops.tensorrt.quantize_op.default(
            x, amax, 8, 0, self.unsigned, False, "Float", None, 0
        )


class _StrictINT4Weight(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(4, 32, dtype=torch.float16), requires_grad=False)
        self.register_buffer("amax", torch.ones(4, 1))

    def forward(self):
        return torch.ops.tensorrt.quantize_op.default(
            self.weight,
            self.amax,
            4,
            0,
            False,
            True,
            "Float",
            32,
            0,
        )


class _StrictMXFP8Weight(nn.Module):
    def __init__(self):
        super().__init__()
        weight = torch.linspace(-1.0, 1.0, steps=32 * 64).reshape(32, 64)
        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        weight = torch.ops.tensorrt.dynamic_block_quantize_op.overload(
            self.weight,
            32,
            None,
            8,
            4,
            9,
            8,
            "Float",
            "static",
        )
        return torch.matmul(x, weight)


_STRICT_CASES = [
    pytest.param(
        "fp8",
        21,
        {("trt", "TRT_FP8QuantizeLinear"), ("trt", "TRT_FP8DequantizeLinear")},
        id="fp8-opset21",
    ),
    pytest.param(
        "fp8",
        24,
        {("trt", "TRT_FP8QuantizeLinear"), ("trt", "TRT_FP8DequantizeLinear")},
        id="fp8-opset24",
    ),
    pytest.param(
        "int8",
        21,
        {("", "QuantizeLinear"), ("", "DequantizeLinear")},
        id="int8",
    ),
    pytest.param("int4", 21, {("trt", "DequantizeLinear")}, id="int4-awq"),
    pytest.param(
        "nvfp4_dynamic",
        21,
        {("trt", "TRT_FP4DynamicQuantize"), ("", "DequantizeLinear")},
        id="nvfp4-dynamic",
    ),
    pytest.param(
        "nvfp4_static",
        21,
        {("trt", "TRT_FP4QDQ")},
        id="nvfp4-static",
    ),
    pytest.param(
        "mxfp8_dynamic",
        21,
        {
            ("trt", "TRT_MXFP8DynamicQuantize"),
            ("trt", "TRT_MXFP8DequantizeLinear"),
        },
        id="mxfp8-dynamic",
    ),
    pytest.param(
        "mxfp8_static",
        21,
        {("trt", "TRT_MXFP8DequantizeLinear")},
        id="mxfp8-static",
    ),
]


@pytest.mark.parametrize(("quant_format", "opset", "expected_nodes"), _STRICT_CASES)
def test_strict_exported_program_uses_translation_table(
    tmp_path, quant_format, opset, expected_nodes
):
    sample_input = torch.randn(4, 32)
    amax = torch.ones(4, 1) if quant_format in {"int8", "int4"} else torch.tensor(1.0)
    exported_program = torch.export.export(
        _StrictQuantOp(quant_format),
        (sample_input, amax),
        strict=True,
    )
    if quant_format.startswith("mxfp8"):
        assert any(
            node.target == torch.ops.tensorrt.dynamic_block_quantize_op.overload
            for node in exported_program.graph.nodes
        )
    onnx_path = tmp_path / f"{quant_format}.onnx"

    torch.onnx.export(
        exported_program,
        (),
        onnx_path,
        dynamo=True,
        opset_version=opset,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported)
    assert not exported.functions
    actual_nodes = {(node.domain, node.op_type) for node in _all_nodes(exported)}
    assert expected_nodes <= actual_nodes
    assert ("tensorrt", "quantize_op") not in actual_nodes
    assert ("tensorrt", "dynamic_block_quantize_op") not in actual_nodes


@pytest.mark.parametrize(
    ("unsigned", "zero_point_dtype", "scale_denominator"),
    [
        pytest.param(False, onnx.TensorProto.INT8, 127.0, id="signed"),
        pytest.param(True, onnx.TensorProto.UINT8, 255.0, id="unsigned"),
    ],
)
def test_strict_int8_translation_contract(tmp_path, unsigned, zero_point_dtype, scale_denominator):
    sample_input = torch.randn(4, 32)
    amax = torch.ones(4, 1)
    exported_program = torch.export.export(
        _StrictINT8(unsigned),
        (sample_input, amax),
        strict=True,
    )
    onnx_path = tmp_path / f"int8_{'unsigned' if unsigned else 'signed'}.onnx"

    torch.onnx.export(
        exported_program,
        (),
        onnx_path,
        dynamo=True,
        opset_version=24,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    quantize = next(node for node in exported.graph.node if node.op_type == "QuantizeLinear")
    dequantize = next(node for node in exported.graph.node if node.op_type == "DequantizeLinear")
    assert _node_attributes(quantize)["axis"] == 0
    assert _node_attributes(dequantize)["axis"] == 0
    assert quantize.input[2] == dequantize.input[2]
    assert _tensor_metadata(exported, quantize.input[2]) == (zero_point_dtype, [4])
    assert _tensor_metadata(exported, quantize.output[0]) == (
        zero_point_dtype,
        list(sample_input.shape),
    )
    assert scale_denominator in _scalar_initializer_values(exported)
    assert any(node.op_type == "Where" for node in exported.graph.node)


def test_strict_int4_keeps_initializer_as_marker_input(tmp_path):
    exported_program = torch.export.export(_StrictINT4Weight(), (), strict=True)
    onnx_path = tmp_path / "int4_weight.onnx"

    torch.onnx.export(
        exported_program,
        (),
        onnx_path,
        dynamo=True,
        opset_version=24,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    onnx.checker.check_model(exported, full_check=True)
    assert not exported.functions
    int4_dq = next(
        node
        for node in exported.graph.node
        if node.domain == "trt" and node.op_type == "DequantizeLinear"
    )
    assert int4_dq.input[0] in {initializer.name for initializer in exported.graph.initializer}
    assert exported.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT


def test_strict_mxfp8_static_weight_postprocess_syncs_initializer_metadata(tmp_path):
    sample_input = torch.randn(4, 32)
    exported_program = torch.export.export(_StrictMXFP8Weight(), (sample_input,), strict=True)
    onnx_path = tmp_path / "mxfp8_weight.onnx"

    torch.onnx.export(
        exported_program,
        (),
        onnx_path,
        dynamo=True,
        opset_version=21,
        custom_translation_table=get_dynamo_onnx_translation_table(),
    )

    exported = onnx.load(onnx_path)
    assert not exported.functions
    mxfp8_dq = next(
        node for node in exported.graph.node if node.op_type == "TRT_MXFP8DequantizeLinear"
    )
    weight_name, scale_name = mxfp8_dq.input
    assert _tensor_metadata(exported, weight_name) == (onnx.TensorProto.FLOAT, [32, 64])
    assert _tensor_metadata(exported, scale_name) == (onnx.TensorProto.FLOAT, [])

    exported = MXFP8QuantExporter.process_model(exported)
    assert _tensor_metadata(exported, weight_name) == (
        onnx.TensorProto.FLOAT8E4M3FN,
        [32, 64],
    )
    assert _tensor_metadata(exported, scale_name) == (onnx.TensorProto.UINT8, [32, 2])
    assert exported.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    onnx.checker.check_model(exported, full_check=True)


def test_dynamo_translation_table_covers_custom_op_overloads():
    table = get_dynamo_onnx_translation_table()

    assert {
        torch.ops.tensorrt.quantize_op.default,
        torch.ops.tensorrt.dynamic_block_quantize_op.default,
        torch.ops.tensorrt.dynamic_block_quantize_op.overload,
    } <= table.keys()


def test_custom_op_schemas_accept_legacy_positional_calls():
    sample_input = torch.randn(4, 32)
    amax = torch.tensor(1.0)

    fp8_output = torch.ops.tensorrt.quantize_op(sample_input, amax, 8, 4, False, False)
    mxfp8_output = torch.ops.tensorrt.dynamic_block_quantize_op.overload(
        sample_input, 32, None, 8, 4, 9, 8
    )

    assert fp8_output.shape == sample_input.shape
    assert fp8_output.dtype == sample_input.dtype
    assert mxfp8_output.shape == sample_input.shape
    assert mxfp8_output.dtype == sample_input.dtype
