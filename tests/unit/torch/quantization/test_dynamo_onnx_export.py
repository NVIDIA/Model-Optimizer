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
from modelopt.onnx.export import INT4QuantExporter, MXFP8QuantExporter, NVFP4QuantExporter
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
        if self.quant_format == "int8":
            return torch.ops.tensorrt.quantize_op.default(
                x, amax, 8, 0, False, False, "Float", None, 0
            )
        if self.quant_format.startswith("nvfp4"):
            return torch.ops.tensorrt.dynamic_block_quantize_op.default(
                x, 16, amax, *_NVFP4_ARGS, self.quant_format.removeprefix("nvfp4_")
            )
        return torch.ops.tensorrt.dynamic_block_quantize_op.overload(
            x, 32, None, *_MXFP8_ARGS, self.quant_format.removeprefix("mxfp8_")
        )


class _INT4ScaleFanout(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight0 = nn.Parameter(torch.randn(64, 128))
        self.weight1 = nn.Parameter(torch.randn(64, 128))
        self.register_buffer("amax0", torch.ones(64, 1))
        self.register_buffer("amax1", torch.ones(64, 1))

    def forward(self, x):
        quantized = [
            torch.ops.tensorrt.quantize_op.default(weight, amax, 4, 0, False, True, "Float", 128, 1)
            for weight, amax in ((self.weight0, self.amax0), (self.weight1, self.amax1))
        ]
        return x @ quantized[0].T + x @ quantized[1].T


def _assert_valid_without_custom_ops(model):
    onnx.checker.check_model(model, full_check=True)
    assert not model.functions
    assert not any(
        node.domain == "tensorrt" or node.op_type in {"quantize_op", "dynamic_block_quantize_op"}
        for node in model.graph.node
    )
    return model


def _helper_export(model, inputs, name, opset, tmp_path):
    payload, _ = get_onnx_bytes_and_metadata(
        model,
        inputs,
        model_name=name,
        dynamo_export=True,
        onnx_opset=opset,
    )
    package = OnnxBytes.from_bytes(payload)
    directory = tmp_path / name
    package.write_to_disk(str(directory))
    path = directory / f"{package.model_name}.onnx"
    return _assert_valid_without_custom_ops(onnx.load(path, load_external_data=True))


def _raw_export(model, inputs, path, opset=21):
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
    return _assert_valid_without_custom_ops(onnx.load(path))


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


def _linears(features, count=2):
    return nn.Sequential(*(nn.Linear(features, features, bias=False) for _ in range(count)))


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


def _weight_dqs(model, op_type):
    initializers = {item.name for item in model.graph.initializer}
    return [
        node
        for node in model.graph.node
        if node.op_type == op_type and node.input[0] in initializers
    ]


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
    ("fp8_none", 21, ("trt", "TRT_FP8QuantizeLinear")),
    ("int8", 21, ("", "QuantizeLinear")),
    ("nvfp4_dynamic", 21, ("trt", "TRT_FP4DynamicQuantize")),
    ("nvfp4_static", 21, ("trt", "TRT_FP4QDQ")),
    ("mxfp8_dynamic", 21, ("trt", "TRT_MXFP8DynamicQuantize")),
    ("mxfp8_static", 24, ("trt", "TRT_MXFP8DequantizeLinear")),
]


@pytest.mark.parametrize(("quant_format", "opset", "expected_node"), _STRICT_CASES)
def test_private_table_strict_formats(tmp_path, quant_format, opset, expected_node):
    sample_input = torch.randn(4, 32)
    amax = torch.ones(4, 1) if quant_format == "int8" else torch.tensor(1.0)
    exported_program = torch.export.export(
        _StrictQuantOp(quant_format), (sample_input, amax), strict=True
    )
    exported = _raw_export(exported_program, (), tmp_path / f"{quant_format}_{opset}.onnx", opset)
    assert expected_node in {(node.domain, node.op_type) for node in exported.graph.node}
    assert {item.domain: item.version for item in exported.opset_import}[""] == opset


def test_int8_fp16_bias_preserves_source_dtype(tmp_path):
    inputs = (torch.randn(2, 32, dtype=torch.float16),)
    model = _quantize(nn.Linear(32, 32, dtype=torch.float16).eval(), inputs, mtq.INT8_DEFAULT_CFG)
    exported = _helper_export(model, inputs, "int8_fp16", 21, tmp_path)
    assert exported.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16


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


@pytest.mark.parametrize("quant_format", ["int4", "mxfp8"])
def test_postprocessors_split_shared_scales(tmp_path, quant_format):
    if quant_format == "int4":
        inputs = (torch.randn(2, 128),)
        model = _INT4ScaleFanout().eval()
        exporter, op_type = INT4QuantExporter, "DequantizeLinear"
    else:
        inputs = (torch.randn(2, 64),)
        model = _quantize(_linears(64, 3).eval(), inputs, mtq.MXFP8_DEFAULT_CFG)
        exporter, op_type = MXFP8QuantExporter, "TRT_MXFP8DequantizeLinear"

    exported = _raw_export(model, inputs, tmp_path / f"{quant_format}_shared_scale.onnx")
    weight_dqs = _weight_dqs(exported, op_type)
    assert len(weight_dqs) >= 2 and len({node.input[1] for node in weight_dqs}) == 1

    exported = _assert_valid_without_custom_ops(exporter.process_model(exported))
    weight_dqs = _weight_dqs(exported, op_type)
    assert len({node.input[1] for node in weight_dqs}) == len(weight_dqs)


def test_helper_exports_biased_nvfp4_with_default_fp32(tmp_path, block_quant_export_only):
    inputs = (SimpleLinear.get_input(),)
    model = _quantize(SimpleLinear().eval(), inputs, mtq.NVFP4_DEFAULT_CFG)

    exported = _helper_export(model, inputs, "biased_nvfp4", 21, tmp_path)

    _assert_fp16_gemm_with_float_io(exported)


def test_helper_exports_mxfp8_at_opset21(tmp_path, block_quant_export_only):
    inputs = (torch.randn(2, 64),)
    model = _quantize(nn.Linear(64, 64).eval(), inputs, mtq.MXFP8_DEFAULT_CFG)
    exported = _helper_export(model, inputs, "mxfp8", 21, tmp_path)
    assert any(node.op_type == "TRT_MXFP8DequantizeLinear" for node in exported.graph.node)
    assert {onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.UINT8} <= {
        item.data_type for item in exported.graph.initializer
    }
    _assert_fp16_gemm_with_float_io(exported)


@pytest.mark.parametrize("opset", [21, 24])
def test_helper_exports_fp8_autoquant(tmp_path, opset):
    sample_input = torch.randn(2, 16)
    model = _auto_quantize(
        _linears(16).eval(),
        sample_input,
        [mtq.FP8_DEFAULT_CFG],
        effective_bits=8.0,
    )
    exported = _helper_export(model, (sample_input,), f"two_linear_fp8_{opset}", opset, tmp_path)
    assert {"QuantizeLinear", "DequantizeLinear"} <= {node.op_type for node in exported.graph.node}
    fp8_weights = {
        item.name
        for item in exported.graph.initializer
        if item.data_type == onnx.TensorProto.FLOAT8E4M3FN and list(item.dims) == [16, 16]
    }
    assert len(fp8_weights) == 2


def test_helper_exports_fp16_int4_awq_at_opset21(tmp_path):
    inputs = (torch.randn(2, 256, dtype=torch.float16),)
    model = _quantize(
        nn.Linear(256, 64, bias=False, dtype=torch.float16).eval(),
        inputs,
        mtq.INT4_AWQ_CFG,
    )

    exported = _helper_export(model, inputs, "linear_int4", 21, tmp_path)

    dq = next(
        node
        for node in exported.graph.node
        if node.domain == "trt" and node.op_type == "DequantizeLinear"
    )
    initializers = {initializer.name: initializer for initializer in exported.graph.initializer}
    weight, scale = initializers[dq.input[0]], initializers[dq.input[1]]
    assert (weight.data_type, list(weight.dims), list(scale.dims)) == (
        onnx.TensorProto.INT4,
        [64, 256],
        [64, 2],
    )


def test_helper_exports_mixed_autoquant_without_markers(tmp_path):
    sample_input = torch.randn(2, 128)
    model = _auto_quantize(
        _linears(128).eval(),
        sample_input,
        [mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, mtq.FP8_DEFAULT_CFG],
        effective_bits=6.0,
    )

    exported = _helper_export(model, (sample_input,), "two_linear_mixed", 24, tmp_path)

    initializer_dtypes = {initializer.data_type for initializer in exported.graph.initializer}
    assert {onnx.TensorProto.INT4, onnx.TensorProto.FLOAT8E4M3FN} <= initializer_dtypes
    assert not any(node.op_type.startswith("TRT_") for node in exported.graph.node)


def test_custom_op_schemas_keep_legacy_positional_calls():
    inputs = torch.randn(4, 32)
    amax = torch.tensor(1.0)
    fp8 = torch.ops.tensorrt.quantize_op(inputs, amax, 8, 4, False, False)
    mxfp8 = torch.ops.tensorrt.dynamic_block_quantize_op.overload(inputs, 32, None, 8, 4, 9, 8)

    assert (fp8.shape, fp8.dtype) == (inputs.shape, inputs.dtype)
    assert (mxfp8.shape, mxfp8.dtype) == (inputs.shape, inputs.dtype)


def test_nvfp4_rejects_marker_output_fanout():
    weight = onnx.numpy_helper.from_array(torch.ones(4, 32).numpy(), "weight")
    nodes = [
        onnx.helper.make_node("TRT_FP4QDQ", ["weight"], ["weight_dq"], block_size=16),
        onnx.helper.make_node("Identity", ["weight_dq"], ["output0"]),
        onnx.helper.make_node("Identity", ["weight_dq"], ["output1"]),
    ]
    graph = onnx.helper.make_graph(nodes, "fanout", [], [], [weight])
    with pytest.raises(NotImplementedError, match="expected one consumer"):
        NVFP4QuantExporter.pre_process(onnx.helper.make_model(graph))


def test_helper_rejects_unsupported_dynamo_options():
    model, inputs = nn.Linear(128, 64, bias=False), (torch.randn(1, 128),)
    with pytest.raises(ValueError, match="opset 21 or newer"):
        get_onnx_bytes_and_metadata(model, inputs, dynamo_export=True, onnx_opset=20)
    with pytest.raises(NotImplementedError, match="dynamic_axes"):
        get_onnx_bytes_and_metadata(
            model,
            inputs,
            dynamo_export=True,
            onnx_opset=21,
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
            nn.Identity(), (torch.ones(1),), dynamo_export=True, onnx_opset=21
        )
