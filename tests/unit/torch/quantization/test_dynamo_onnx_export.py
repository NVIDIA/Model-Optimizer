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

import copy
import inspect

import pytest
import torch
from _test_utils.torch.quantization.models import SimpleLinear
from torch import nn

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")

import modelopt.torch.quantization as mtq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization import tensor_quant
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


class _SharedLinears(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(16, 16, bias=False)
        self.second = nn.Linear(16, 16, bias=False)
        self.second.weight = self.first.weight

    def forward(self, inputs):
        return self.second(self.first(inputs))


class _ConvLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, 16, stride=16)
        self.linear = nn.Linear(512, 128)

    def forward(self, inputs):
        return self.linear(self.conv(inputs).flatten(1))


def _assert_valid_without_custom_ops(model):
    onnx.checker.check_model(model, full_check=True)
    assert not model.functions
    assert not any(
        node.domain == "tensorrt" or node.op_type in {"quantize_op", "dynamic_block_quantize_op"}
        for node in model.graph.node
    )
    return model


def _helper_export(model, inputs, opset, tmp_path):
    payload, _ = get_onnx_bytes_and_metadata(
        model,
        inputs,
        dynamo_export=True,
        onnx_opset=opset,
    )
    package = OnnxBytes.from_bytes(payload)
    package.write_to_disk(str(tmp_path))
    exported = _assert_valid_without_custom_ops(
        onnx.load(tmp_path / f"{package.model_name}.onnx", load_external_data=True)
    )
    imports = {item.domain: item.version for item in exported.opset_import}
    assert imports[""] == opset and imports.get("trt", 1) == 1
    return exported


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


def _auto_quantize(model, sample_input, formats, effective_bits):
    return mtq.auto_quantize(
        model,
        constraints={"effective_bits": effective_bits},
        quantization_formats=[copy.deepcopy(config) for config in formats],
        data_loader=[sample_input],
        forward_step=lambda candidate, batch: candidate(batch),
        loss_func=lambda output, _batch: output.float().square().mean(),
        num_calib_steps=1,
        num_score_steps=1,
    )[0]


def _linears(features):
    return nn.Sequential(*(nn.Linear(features, features, bias=False) for _ in range(2)))


def _quantize(model, inputs, config):
    return mtq.quantize(
        model, copy.deepcopy(config), forward_loop=lambda candidate: candidate(*inputs)
    )


@pytest.fixture
def block_quant_export_only(monkeypatch):
    real_op = tensor_quant.dynamic_block_quantize_op

    def cpu_stub(inputs, *args):
        return real_op(inputs, *args) if torch.compiler.is_exporting() else inputs

    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", cpu_stub)


@pytest.fixture
def fail_dynamo_capture(monkeypatch):
    monkeypatch.setattr(
        torch.onnx,
        "export",
        lambda *args, **kwargs: pytest.fail("capture must not run before preflight succeeds"),
    )


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


def _assert_fp16_gemm_with_float_io(model):
    graph = model.graph
    tensor_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*graph.input, *graph.value_info, *graph.output)
    }
    tensor_types.update(
        {initializer.name: initializer.data_type for initializer in graph.initializer}
    )
    gemms = [node for node in graph.node if node.op_type == "Gemm"]
    assert gemms and all(
        tensor_types[input_name] == onnx.TensorProto.FLOAT16
        for node in gemms
        for input_name in node.input
    )
    producers = {output: node for node in graph.node for output in node.output}
    output_cast = producers[graph.output[0].name]
    assert output_cast.op_type == "Cast" and output_cast.attribute[0].i == onnx.TensorProto.FLOAT
    assert producers[output_cast.input[0]].op_type == "Gemm"


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


def test_helper_uses_same_lowering_after_strict_capture_retry(monkeypatch, tmp_path):
    original_export = torch.export.export
    attempts = []

    def fail_non_strict(*args, **kwargs):
        strict = kwargs.get("strict")
        attempts.append(strict)
        if strict is False:
            raise RuntimeError("force strict capture")
        return original_export(*args, **kwargs)

    monkeypatch.setattr(torch.export, "export", fail_non_strict)
    exported = _helper_export(
        _StrictQuantOp("int8"),
        (torch.randn(4, 32), torch.ones(4, 1)),
        23,
        tmp_path,
    )

    assert attempts == [False, True]
    assert {node.op_type for node in exported.graph.node} >= {
        "QuantizeLinear",
        "DequantizeLinear",
    }


def test_private_table_rejects_int8_axis_disagreement(tmp_path):
    inputs = torch.randn(2, 3, 4)
    amax = torch.ones(1, 3, 1)
    exported_program = torch.export.export(_StrictQuantOp("int8"), (inputs, amax), strict=True)

    with pytest.raises(torch.onnx.OnnxExporterError, match="axis 0 does not match amax axis 1"):
        _raw_export(exported_program, (), tmp_path / "int8_axis.onnx", 23)


def test_int8_fp16_bias_preserves_source_dtype(tmp_path):
    inputs = (torch.randn(2, 32, dtype=torch.float16),)
    model = _quantize(nn.Linear(32, 32, dtype=torch.float16).eval(), inputs, mtq.INT8_DEFAULT_CFG)
    exported = _helper_export(model, inputs, 23, tmp_path)
    assert exported.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    initializers = {item.name: item for item in exported.graph.initializer}
    weight_q = next(
        node
        for node in exported.graph.node
        if node.domain == "" and node.op_type == "QuantizeLinear" and node.input[0] in initializers
    )
    assert _attribute(weight_q, "axis") == 0
    assert initializers[weight_q.input[2]].data_type == onnx.TensorProto.INT8


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


def test_helper_exports_biased_nvfp4_with_default_fp32(tmp_path, block_quant_export_only):
    inputs = (SimpleLinear.get_input(),)
    model = _quantize(SimpleLinear().eval(), inputs, mtq.NVFP4_DEFAULT_CFG)

    exported = _helper_export(model, inputs, 23, tmp_path)

    _assert_fp16_gemm_with_float_io(exported)
    dynamic_q = next(
        node for node in exported.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    )
    assert (
        dynamic_q.domain,
        _attribute(dynamic_q, "axis"),
        _attribute(dynamic_q, "block_size"),
    ) == (
        "trt",
        -1,
        16,
    )
    assert {node.domain for node in exported.graph.node if node.op_type == "DequantizeLinear"} == {
        ""
    }


def test_helper_exports_mxfp8_at_opset23(tmp_path, block_quant_export_only):
    inputs = (torch.randn(2, 64),)
    model = _quantize(nn.Linear(64, 64).eval(), inputs, mtq.MXFP8_DEFAULT_CFG)
    exported = _helper_export(model, inputs, 23, tmp_path)
    assert any(node.op_type == "TRT_MXFP8DequantizeLinear" for node in exported.graph.node)
    assert {onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.UINT8} <= {
        item.data_type for item in exported.graph.initializer
    }
    dynamic_q = next(
        node for node in exported.graph.node if node.op_type == "TRT_MXFP8DynamicQuantize"
    )
    assert (_attribute(dynamic_q, "axis"), _attribute(dynamic_q, "block_size")) == (-1, 32)
    assert _attribute(dynamic_q, "output_dtype") == onnx.TensorProto.FLOAT8E4M3FN
    _assert_fp16_gemm_with_float_io(exported)


def test_mxfp8_fp8_conv_uses_matching_qdq_types(tmp_path, block_quant_export_only):
    config = copy.deepcopy(mtq.MXFP8_DEFAULT_CFG)
    config["quant_cfg"].extend(
        [
            {
                "parent_class": "nn.Conv2d",
                "quantizer_name": f"*{name}_quantizer",
                "cfg": {"num_bits": (4, 3), "axis": None},
            }
            for name in ("weight", "input")
        ]
    )
    config["algorithm"] = "max"
    inputs = (torch.randn(2, 3, 32, 32),)
    model = _quantize(_ConvLinear().eval(), inputs, config)
    exported = _helper_export(model, inputs, 23, tmp_path)
    tensor_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (*exported.graph.input, *exported.graph.value_info, *exported.graph.output)
    }
    tensor_types.update(
        {initializer.name: initializer.data_type for initializer in exported.graph.initializer}
    )
    producers = {output: node for node in exported.graph.node for output in node.output}
    conv = next(node for node in exported.graph.node if node.op_type == "Conv")
    activation_dq = producers[conv.input[0]]
    activation_q = producers[activation_dq.input[0]]
    assert (activation_q.op_type, activation_dq.op_type) == (
        "QuantizeLinear",
        "DequantizeLinear",
    )
    assert tensor_types[activation_q.input[0]] == tensor_types[activation_q.input[1]]
    assert tensor_types[activation_q.input[0]] == onnx.TensorProto.FLOAT16
    assert {value.type.tensor_type.elem_type for value in exported.graph.input} == {
        onnx.TensorProto.FLOAT
    }
    assert {value.type.tensor_type.elem_type for value in exported.graph.output} == {
        onnx.TensorProto.FLOAT
    }


@pytest.mark.parametrize("opset", [23, 24])
def test_helper_exports_fp8_autoquant(tmp_path, opset):
    sample_input = torch.randn(2, 16)
    model = _auto_quantize(
        _linears(16).eval(),
        sample_input,
        [mtq.FP8_DEFAULT_CFG],
        effective_bits=8.0,
    )
    exported = _helper_export(model, (sample_input,), opset, tmp_path)
    quantizers = [node for node in exported.graph.node if node.op_type == "QuantizeLinear"]
    dequantizers = [node for node in exported.graph.node if node.op_type == "DequantizeLinear"]
    assert (len(quantizers), len(dequantizers)) == (2, 4)
    assert {node.domain for node in [*quantizers, *dequantizers]} == {""}
    fp8_weights = {
        item.name
        for item in exported.graph.initializer
        if item.data_type == onnx.TensorProto.FLOAT8E4M3FN and list(item.dims) == [16, 16]
    }
    assert len(fp8_weights) == 2


def test_helper_exports_fp16_int4_awq_at_opset23(tmp_path):
    inputs = (torch.randn(2, 256, dtype=torch.float16),)
    model = _quantize(
        nn.Linear(256, 64, bias=False, dtype=torch.float16).eval(),
        inputs,
        mtq.INT4_AWQ_CFG,
    )

    exported = _helper_export(model, inputs, 23, tmp_path)

    dq = next(
        node
        for node in exported.graph.node
        if node.domain == "" and node.op_type == "DequantizeLinear"
    )
    initializers = {initializer.name: initializer for initializer in exported.graph.initializer}
    weight, scale = initializers[dq.input[0]], initializers[dq.input[1]]
    assert (weight.data_type, list(weight.dims), list(scale.dims)) == (
        onnx.TensorProto.INT4,
        [64, 256],
        [64, 2],
    )
    assert (_attribute(dq, "axis"), _attribute(dq, "block_size")) == (1, 128)


def test_helper_exports_fp8_nvfp4_autoquant(tmp_path, block_quant_export_only):
    sample_input = torch.randn(2, 128)
    model = _auto_quantize(
        _linears(128).eval(),
        sample_input,
        [mtq.NVFP4_DEFAULT_CFG, mtq.FP8_DEFAULT_CFG],
        effective_bits=6.25,
    )
    exported = _helper_export(model, (sample_input,), 24, tmp_path)

    initializer_dtypes = {initializer.data_type for initializer in exported.graph.initializer}
    assert {onnx.TensorProto.FLOAT4E2M1, onnx.TensorProto.FLOAT8E4M3FN} <= initializer_dtypes
    assert ("trt", "TRT_FP4DynamicQuantize") in {
        (node.domain, node.op_type) for node in exported.graph.node
    }
    assert ("", "QuantizeLinear") in {(node.domain, node.op_type) for node in exported.graph.node}


def test_helper_rejects_uncalibrated_quantizer_before_capture(fail_dynamo_capture):
    inputs = (torch.randn(2, 16),)
    model = _quantize(nn.Linear(16, 16).eval(), inputs, mtq.FP8_DEFAULT_CFG)
    del model.input_quantizer._amax
    with pytest.raises(ValueError, match="has not been calibrated"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


def test_helper_rejects_unsupported_mixed_formats_before_capture(fail_dynamo_capture):
    inputs = (torch.randn(2, 128),)
    model = nn.Sequential(
        _quantize(nn.Linear(128, 128).eval(), inputs, mtq.FP8_DEFAULT_CFG),
        _quantize(nn.Linear(128, 128).eval(), inputs, mtq.INT4_AWQ_CFG),
    )
    with pytest.raises(NotImplementedError, match="mixed AutoQuant only for FP8 with NVFP4"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


def test_helper_rejects_block_activation_rank_before_capture(
    fail_dynamo_capture, block_quant_export_only
):
    inputs = (torch.randn(1, 2, 3, 32),)
    model = _quantize(nn.Linear(32, 32).eval(), inputs, mtq.NVFP4_DEFAULT_CFG)
    with pytest.raises(NotImplementedError, match="supports rank 2 or 3 only"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


def test_helper_rejects_shared_quantized_weights_before_capture(fail_dynamo_capture):
    inputs = (torch.randn(2, 16),)
    model = _quantize(_SharedLinears().eval(), inputs, mtq.FP8_DEFAULT_CFG)

    with pytest.raises(NotImplementedError, match="shared quantized weights"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


@pytest.mark.parametrize("case", ["unsigned_int4", "narrow_int8", "w4a8"])
def test_helper_rejects_unsupported_quantizer_configs_before_capture(case, fail_dynamo_capture):
    inputs = (torch.randn(2, 128),)
    config = mtq.INT8_DEFAULT_CFG if case == "narrow_int8" else mtq.INT4_AWQ_CFG
    model = _quantize(nn.Linear(128, 128).eval(), inputs, config)

    if case == "unsigned_int4":
        model.weight_quantizer.unsigned = True
        message = "signed INT4 only"
    elif case == "narrow_int8":
        model.weight_quantizer.narrow_range = True
        message = "signed narrow-range INT8"
    else:
        quantizer = model.input_quantizer
        quantizer.enable()
        quantizer.num_bits = 8
        quantizer.block_sizes = None
        quantizer.unsigned = False
        quantizer.narrow_range = False
        quantizer._amax = torch.tensor(1.0)
        message = "quantizer combination"

    with pytest.raises(NotImplementedError, match=message):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


def test_helper_rejects_block_size_mismatch_before_capture(
    fail_dynamo_capture, block_quant_export_only
):
    calibration_inputs = (torch.randn(2, 32),)
    model = _quantize(nn.Linear(32, 32).eval(), calibration_inputs, mtq.NVFP4_DEFAULT_CFG)

    with pytest.raises(NotImplementedError, match="must divide the last dimension 30"):
        get_onnx_bytes_and_metadata(model, (torch.randn(2, 30),), dynamo_export=True)


def test_helper_rejects_block_quantized_conv_before_capture(fail_dynamo_capture):
    inputs = (torch.randn(1, 4, 2, 2),)
    model = _quantize(nn.Conv2d(4, 4, 1).eval(), inputs, mtq.FP8_DEFAULT_CFG)
    for quantizer in (model.input_quantizer, model.weight_quantizer):
        quantizer.num_bits = (2, 1)
        quantizer.block_sizes = {
            -1: 16,
            "type": "dynamic",
            "scale_bits": (4, 3),
        }

    with pytest.raises(NotImplementedError, match="block-quantized Conv"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True)


def test_custom_op_schemas_keep_legacy_positional_calls():
    inputs = torch.randn(4, 32)
    amax = torch.tensor(1.0)
    fp8 = torch.ops.tensorrt.quantize_op(inputs, amax, 8, 4, False, False)
    mxfp8 = torch.ops.tensorrt.dynamic_block_quantize_op.overload(inputs, 32, None, 8, 4, 9, 8)

    assert (fp8.shape, fp8.dtype) == (inputs.shape, inputs.dtype)
    assert (mxfp8.shape, mxfp8.dtype) == (inputs.shape, inputs.dtype)


def test_helper_rejects_unsupported_dynamo_opset():
    model, inputs = nn.Linear(128, 64, bias=False), (torch.randn(1, 128),)
    with pytest.raises(ValueError, match="opset 23 or newer"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True, onnx_opset=22)


def test_helper_rejects_dynamic_axes():
    model, inputs = nn.Linear(128, 64, bias=False), (torch.randn(1, 128),)
    with pytest.raises(NotImplementedError, match="dynamic_axes"):
        get_onnx_bytes_and_metadata(
            model,
            inputs,
            dynamo_export=True,
            onnx_opset=23,
            dynamic_axes={"input": {0: "batch"}},
        )


def test_helper_disables_legacy_fallback(monkeypatch):
    if "fallback" not in inspect.signature(torch.onnx.export).parameters:
        pytest.skip("This PyTorch version has no legacy fallback option.")

    def assert_disabled(*args, fallback=True, **kwargs):
        assert fallback is False
        raise RuntimeError("checked")

    monkeypatch.setattr(torch.onnx, "export", assert_disabled)
    with pytest.raises(RuntimeError, match="checked"):
        get_onnx_bytes_and_metadata(
            nn.Identity(), (torch.ones(1),), dynamo_export=True, onnx_opset=23
        )
