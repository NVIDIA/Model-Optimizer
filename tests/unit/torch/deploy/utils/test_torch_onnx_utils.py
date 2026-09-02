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

import json
from contextlib import nullcontext

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.nn as nn
from _test_utils.torch.deploy.lib_test_models import BaseDeployModel, get_deploy_models

import modelopt.torch._deploy.utils.torch_onnx as torch_onnx
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.tensor_quant as tensor_quant
from modelopt.onnx.autocast.convert import convert_to_fp32
from modelopt.onnx.export.base_exporter import ONNXQuantExporter
from modelopt.onnx.utils import get_batch_size_from_bytes, validate_batch_size
from modelopt.torch._deploy.utils import (
    OnnxBytes,
    flatten_tree,
    generate_onnx_input,
    get_onnx_bytes_and_metadata,
)
from modelopt.torch._deploy.utils.torch_onnx import (
    _get_autocast_context,
    _override_onnx_quantizer_precision,
    _to_expected_onnx_type,
)
from modelopt.torch.quantization.nn import TensorQuantizer
from modelopt.torch.utils import standardize_model_args, unflatten_tree

deploy_benchmark_all = get_deploy_models()
deploy_benchmark_dynamo = get_deploy_models(dynamic_control_flow=False)

# `torch.onnx.export(dynamo=True)` is expensive (~1.5s/export), so the dynamo matrix is
# trimmed to representatives that cover each distinct input/output container shape plus
# the compile-failure path. Full structural and numeric-type coverage still runs in the
# (cheap) non-dynamo ``test_onnx_export_and_inputs`` below.
_DYNAMO_REPRESENTATIVE_MODELS = {
    "TensorModel",  # plain single tensor
    "ListMultiModel",  # list of tensors (arg flattening)
    "ListDictModel",  # mixed list + dict nesting
    "NestedModel",  # deeply nested inputs
    "DictMultiModel",  # dict inputs
    "ArgsKwargsModel1",  # args + kwargs (success)
    "ArgsKwargsModel2",  # args + kwargs (compile_fail path)
    "TwoOutModel",  # multiple outputs
    "NestedOutModel",  # nested outputs
}
deploy_benchmark_dynamo = {
    k: v for k, v in deploy_benchmark_dynamo.items() if k in _DYNAMO_REPRESENTATIVE_MODELS
}

_ONNX_DTYPE_BY_NAME = {
    "fp32": onnx.TensorProto.FLOAT,
    "fp16": onnx.TensorProto.FLOAT16,
    "bf16": onnx.TensorProto.BFLOAT16,
}
_QUANTIZED_LINEAR_CASES = {
    "int4": (mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG, 128),
    "mxfp8": (mtq.MXFP8_DEFAULT_CFG, 32),
    "nvfp4": (mtq.NVFP4_DEFAULT_CFG, 32),
    "int8": (mtq.INT8_DEFAULT_CFG, 32),
}
_NATIVE_QUANTIZED_LINEAR_CASES = {
    "fp8": (mtq.FP8_DEFAULT_CFG, 32),
    **_QUANTIZED_LINEAR_CASES,
}


def _export_fp8_linear(source_dtype, weights_dtype):
    model = nn.Sequential(nn.Linear(4, 4, bias=False)).eval().to(source_dtype)
    sample_input = torch.arange(4, dtype=source_dtype).reshape(1, 4)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        model_name="fp8_linear",
        weights_dtype=weights_dtype,
        dq_only=False,
        onnx_opset=23,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    return onnx.load_model_from_string(onnx_bytes_obj.get_onnx_model_file_bytes())


def _export_model(model, sample_input, weights_dtype, onnx_opset=20):
    weights_dtype_kwargs = {} if weights_dtype is None else {"weights_dtype": weights_dtype}
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model.eval(),
        (sample_input,),
        onnx_opset=onnx_opset,
        **weights_dtype_kwargs,
    )
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    return onnx.load_model_from_string(onnx_bytes_obj.get_onnx_model_file_bytes())


def _tensor_dtype_map(model):
    dtype_map = {
        value.name: value.type.tensor_type.elem_type
        for value in (*model.graph.input, *model.graph.output, *model.graph.value_info)
        if value.type.HasField("tensor_type")
    }
    dtype_map.update(
        {initializer.name: initializer.data_type for initializer in model.graph.initializer}
    )
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        value = next(
            (attribute.t for attribute in node.attribute if attribute.name == "value"), None
        )
        if value is not None:
            dtype_map[node.output[0]] = value.data_type
    return dtype_map


def _assert_runtime_io_dtype(model, expected_dtype):
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    runtime_io = [
        *[value for value in model.graph.input if value.name not in initializer_names],
        *model.graph.output,
    ]
    assert runtime_io
    assert all(value.type.tensor_type.elem_type == expected_dtype for value in runtime_io)


def test_autocast_prefers_floating_input_device(monkeypatch):
    autocast_args = {}

    def capture_autocast(*, device_type, dtype):
        autocast_args.update(device_type=device_type, dtype=dtype)
        return nullcontext()

    monkeypatch.setattr(torch, "autocast", capture_autocast)
    flat_input = [
        torch.ones(1, dtype=torch.int64, device="meta"),
        torch.ones(1, dtype=torch.float32),
    ]

    _get_autocast_context(nn.Identity(), flat_input, torch.bfloat16)

    assert autocast_args == {"device_type": "cpu", "dtype": torch.bfloat16}


def test_quant_exporter_preserves_legacy_post_process_signature():
    class LegacyExporter(ONNXQuantExporter):
        pre_process = compute_scales = compress_weights = staticmethod(lambda model: model)

        @staticmethod
        def post_process(model):
            return model

    model = onnx.helper.make_model(onnx.helper.make_graph([], "graph", [], []))

    assert LegacyExporter.process_model(model) is model


@pytest.mark.parametrize(
    "model", deploy_benchmark_dynamo.values(), ids=deploy_benchmark_dynamo.keys()
)
def test_onnx_dynamo_export(skip_on_windows, model: BaseDeployModel):
    # One numeric type is enough here — numeric-type coverage is in test_onnx_export_and_inputs.
    for active in range(1):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, _ = get_onnx_bytes_and_metadata(model, args, dynamo_export=True)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            model_bytes = onnx_bytes_obj.get_onnx_model_file_bytes()

        if model.compile_fail:
            continue

        assert model_bytes != b""
        assert onnx.load_model_from_string(model_bytes)


@pytest.mark.parametrize("model", deploy_benchmark_all.values(), ids=deploy_benchmark_all.keys())
def test_onnx_export_and_inputs(model: BaseDeployModel):
    # try it for all potential numeric types
    for active in range(model.get.num_choices):
        # retrieve args
        model.get.active = active
        model.get.set_default_counter()
        args = model.get_args()

        with pytest.raises(AssertionError) if model.compile_fail else nullcontext():
            onnx_bytes, metadata = get_onnx_bytes_and_metadata(model, args)
            onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
            onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

        if model.compile_fail:
            continue

        assert onnx_bytes != b""
        assert onnx.load_model_from_string(onnx_bytes)

        # check correct naming assignment of ops by running a onnx inference session
        ort_session = ort.InferenceSession(onnx_bytes)
        ort_inputs = [inp.name for inp in ort_session.get_inputs()]

        print(ort_session)

        # NOTE: for dict inputs the order is determined by the order of the keys!
        # So if we change the order of the keys in the input this check might fail
        assert ort_inputs == model.onnx_input_names()

        # check correct naming assignments of outputs
        ort_outputs = [out.name for out in ort_session.get_outputs()]
        assert ort_outputs == model.onnx_output_names()

        # check correct output structure
        assert json.dumps(metadata["output_tree_spec"].spec) == json.dumps(model.output_spec())

        if not model.check_input_option(active):
            continue

        # run inference in ORT session with hand-generated input
        model.get.set_default_counter()
        out_ort_flat = ort_session.run(None, {k: np.asarray(model.get()) for k in ort_inputs})

        # run inference with torch model and flatten them
        out_torch = model(*standardize_model_args(model, args))
        out_torch_flat, out_torch_tree_spec = flatten_tree(out_torch)

        # making sure we have pytorch output type as expected ...
        out_torch_flat = [_to_expected_onnx_type(x) for x in out_torch_flat]

        print(out_ort_flat, out_torch_flat)

        # compare flat ORT and torch results
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort_flat)
        )

        # run inference with properly generated onnx inputs and fill data structure
        inputs_generated = generate_onnx_input(metadata, args)

        if model.invalid_device_input:
            continue

        inputs_generated = {k: v.cpu().numpy() for k, v in inputs_generated.items()}
        out_ort2 = unflatten_tree(ort_session.run(None, inputs_generated), out_torch_tree_spec)

        # now flatten both and compare
        out_ort2_flat, _ = flatten_tree(out_ort2)
        print(out_torch_flat, out_ort2_flat)
        assert all(
            torch.allclose(ot, torch.from_numpy(oo).to(ot))
            for ot, oo in zip(out_torch_flat, out_ort2_flat)
        )


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_scale_dtype", "expected_io_dtypes"),
    [
        pytest.param(
            torch.bfloat16,
            "native",
            onnx.TensorProto.FLOAT,
            (onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT),
            id="native-bf16",
        ),
        pytest.param(
            torch.float32,
            "fp16",
            onnx.TensorProto.FLOAT16,
            (onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT16),
            id="fp32-to-fp16",
        ),
        pytest.param(
            torch.float32,
            "bf16",
            onnx.TensorProto.BFLOAT16,
            (onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16),
            id="fp32-to-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            "bf16",
            onnx.TensorProto.BFLOAT16,
            (onnx.TensorProto.BFLOAT16, onnx.TensorProto.BFLOAT16),
            id="bf16-to-bf16",
        ),
        pytest.param(
            torch.bfloat16,
            "fp32",
            onnx.TensorProto.FLOAT,
            (onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT),
            id="bf16-to-fp32",
        ),
    ],
)
def test_fp8_export_with_supported_weights_dtype(
    source_dtype, weights_dtype, expected_scale_dtype, expected_io_dtypes
):
    exported_model = _export_fp8_linear(source_dtype, weights_dtype)

    onnx.checker.check_model(exported_model)
    assert not any(
        node.op_type in {"TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear"}
        for node in exported_model.graph.node
    )
    initializer_by_name = {
        initializer.name: initializer for initializer in exported_model.graph.initializer
    }
    fp8_weight_dq_nodes = [
        node
        for node in exported_model.graph.node
        if node.op_type == "DequantizeLinear"
        and node.input[0] in initializer_by_name
        and initializer_by_name[node.input[0]].data_type == onnx.TensorProto.FLOAT8E4M3FN
    ]
    assert fp8_weight_dq_nodes
    assert all(
        initializer_by_name[node.input[1]].data_type == expected_scale_dtype
        for node in fp8_weight_dq_nodes
    )
    graph_io = [*exported_model.graph.input, *exported_model.graph.output]
    assert tuple(value.type.tensor_type.elem_type for value in graph_io) == expected_io_dtypes


@pytest.mark.parametrize("format_name", _QUANTIZED_LINEAR_CASES)
@pytest.mark.parametrize("weights_dtype", _ONNX_DTYPE_BY_NAME)
def test_quantized_linear_export_uses_requested_weights_dtype(
    monkeypatch, format_name, weights_dtype
):
    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", lambda inputs, *args: inputs)
    quantization_config, features = _QUANTIZED_LINEAR_CASES[format_name]
    source_dtype = torch.bfloat16 if weights_dtype == "fp32" else torch.float32
    model = nn.Sequential(nn.Linear(features, 8, bias=False)).eval().to(source_dtype)
    sample_input = torch.ones(1, features, dtype=source_dtype)
    model = mtq.quantize(
        model,
        quantization_config,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    exported_model = _export_model(model, sample_input, weights_dtype, onnx_opset=23)
    expected_dtype = _ONNX_DTYPE_BY_NAME[weights_dtype]

    onnx.checker.check_model(exported_model)
    inferred_model = onnx.shape_inference.infer_shapes(
        exported_model, check_type=True, strict_mode=True
    )
    _assert_runtime_io_dtype(exported_model, expected_dtype)
    dtype_map = _tensor_dtype_map(inferred_model)
    matmul = next(node for node in inferred_model.graph.node if node.op_type == "MatMul")
    assert all(dtype_map[input_name] == expected_dtype for input_name in matmul.input)

    initializer_map = {
        initializer.name: initializer for initializer in exported_model.graph.initializer
    }
    cast_nodes = [node for node in exported_model.graph.node if node.op_type == "Cast"]

    if format_name == "int4":
        weight = next(
            initializer
            for initializer in initializer_map.values()
            if initializer.data_type == onnx.TensorProto.INT4
        )
        weight_dq = next(
            node
            for node in exported_model.graph.node
            if node.op_type == "DequantizeLinear" and node.input[0] == weight.name
        )
        assert initializer_map[weight_dq.input[1]].data_type == expected_dtype
        assert dtype_map[weight_dq.output[0]] == expected_dtype
        assert not cast_nodes
    elif format_name == "mxfp8":
        initializer_dtypes = {initializer.data_type for initializer in initializer_map.values()}
        assert onnx.TensorProto.FLOAT8E4M3FN in initializer_dtypes
        assert onnx.TensorProto.UINT8 in initializer_dtypes
        dq_nodes = [
            node
            for node in exported_model.graph.node
            if node.op_type == "TRT_MXFP8DequantizeLinear"
        ]
        assert dq_nodes
        assert all(
            next(attribute.i for attribute in node.attribute if attribute.name == "output_dtype")
            == expected_dtype
            for node in dq_nodes
        )
        assert not cast_nodes
    elif format_name == "nvfp4":
        initializer_dtypes = {initializer.data_type for initializer in initializer_map.values()}
        assert {
            onnx.TensorProto.FLOAT4E2M1,
            onnx.TensorProto.FLOAT8E4M3FN,
            onnx.TensorProto.FLOAT,
        } <= initializer_dtypes
        cast_dtypes = {
            next(attribute.i for attribute in node.attribute if attribute.name == "to")
            for node in cast_nodes
        }
        assert cast_dtypes == ({expected_dtype} if weights_dtype != "fp32" else set())
    else:
        q_nodes = [node for node in inferred_model.graph.node if node.op_type == "QuantizeLinear"]
        dq_nodes = [
            node for node in inferred_model.graph.node if node.op_type == "DequantizeLinear"
        ]
        assert q_nodes and dq_nodes
        assert all(dtype_map[node.output[0]] == onnx.TensorProto.INT8 for node in q_nodes)
        assert all(dtype_map[node.input[1]] == expected_dtype for node in dq_nodes)
        assert all(dtype_map[node.output[0]] == expected_dtype for node in dq_nodes)
        assert not cast_nodes


@pytest.mark.parametrize("format_name", _NATIVE_QUANTIZED_LINEAR_CASES)
def test_quantized_linear_default_preserves_native_behavior(monkeypatch, format_name):
    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", lambda inputs, *args: inputs)
    quantization_config, features = _NATIVE_QUANTIZED_LINEAR_CASES[format_name]
    model = nn.Sequential(nn.Linear(features, 8, bias=False)).eval().to(torch.bfloat16)
    sample_input = torch.ones(1, features, dtype=torch.bfloat16)
    model = mtq.quantize(
        model,
        quantization_config,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )

    exported_model = _export_model(model, sample_input, None, onnx_opset=23)
    onnx.checker.check_model(exported_model)
    inferred_model = onnx.shape_inference.infer_shapes(
        exported_model, check_type=True, strict_mode=True
    )
    initializer_names = {initializer.name for initializer in exported_model.graph.initializer}
    runtime_inputs = [
        value for value in exported_model.graph.input if value.name not in initializer_names
    ]
    assert runtime_inputs
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16 for value in runtime_inputs
    )
    expected_boundary_dtype = (
        onnx.TensorProto.BFLOAT16 if format_name == "nvfp4" else onnx.TensorProto.FLOAT
    )
    assert all(
        value.type.tensor_type.elem_type == expected_boundary_dtype
        for value in exported_model.graph.output
    )
    dtype_map = _tensor_dtype_map(inferred_model)
    matmul = next(node for node in inferred_model.graph.node if node.op_type == "MatMul")
    assert all(dtype_map[input_name] == expected_boundary_dtype for input_name in matmul.input)

    initializer_map = {
        initializer.name: initializer for initializer in exported_model.graph.initializer
    }
    initializer_dtypes = {initializer.data_type for initializer in initializer_map.values()}
    if format_name == "fp8":
        activation_q_nodes = [
            node for node in inferred_model.graph.node if node.op_type == "QuantizeLinear"
        ]
        cast_by_output = {
            node.output[0]: node for node in inferred_model.graph.node if node.op_type == "Cast"
        }
        assert activation_q_nodes
        assert all(
            dtype_map[node.input[0]] == onnx.TensorProto.FLOAT
            and dtype_map[node.input[1]] == onnx.TensorProto.FLOAT
            for node in activation_q_nodes
        )
        assert all(node.input[0] in cast_by_output for node in activation_q_nodes)
        assert all(
            next(
                attribute.i
                for attribute in cast_by_output[node.input[0]].attribute
                if attribute.name == "to"
            )
            == onnx.TensorProto.FLOAT
            for node in activation_q_nodes
        )
        weight_dq_nodes = [
            node
            for node in inferred_model.graph.node
            if node.op_type == "DequantizeLinear"
            and node.input[0] in initializer_map
            and initializer_map[node.input[0]].data_type == onnx.TensorProto.FLOAT8E4M3FN
        ]
        assert weight_dq_nodes
        dq_nodes = [
            node for node in inferred_model.graph.node if node.op_type == "DequantizeLinear"
        ]
        assert all(
            dtype_map[node.input[1]] == onnx.TensorProto.FLOAT
            and dtype_map[node.output[0]] == onnx.TensorProto.FLOAT
            for node in dq_nodes
        )
    elif format_name == "int4":
        assert onnx.TensorProto.INT4 in initializer_dtypes
        weight_dq = next(
            node
            for node in inferred_model.graph.node
            if node.op_type == "DequantizeLinear"
            and node.input[0] in initializer_map
            and initializer_map[node.input[0]].data_type == onnx.TensorProto.INT4
        )
        assert dtype_map[weight_dq.input[1]] == onnx.TensorProto.FLOAT
    elif format_name == "mxfp8":
        assert {onnx.TensorProto.FLOAT8E4M3FN, onnx.TensorProto.UINT8} <= initializer_dtypes
        dq_nodes = [
            node
            for node in inferred_model.graph.node
            if node.op_type == "TRT_MXFP8DequantizeLinear"
        ]
        assert dq_nodes
        assert all(
            next(attribute.i for attribute in node.attribute if attribute.name == "output_dtype")
            == onnx.TensorProto.FLOAT16
            for node in dq_nodes
        )
    elif format_name == "nvfp4":
        assert {
            onnx.TensorProto.FLOAT4E2M1,
            onnx.TensorProto.FLOAT8E4M3FN,
            onnx.TensorProto.FLOAT,
        } <= initializer_dtypes
    else:
        q_nodes = [node for node in inferred_model.graph.node if node.op_type == "QuantizeLinear"]
        dq_nodes = [
            node for node in inferred_model.graph.node if node.op_type == "DequantizeLinear"
        ]
        assert q_nodes and dq_nodes
        assert all(dtype_map[node.output[0]] == onnx.TensorProto.INT8 for node in q_nodes)
        assert all(dtype_map[node.input[1]] == onnx.TensorProto.FLOAT for node in dq_nodes)


@pytest.mark.parametrize("weights_dtype", _ONNX_DTYPE_BY_NAME)
def test_fp8_conv_export_uses_requested_weights_dtype(weights_dtype):
    source_dtype = torch.bfloat16 if weights_dtype == "fp32" else torch.float32
    model = nn.Conv2d(3, 4, kernel_size=3, bias=False).eval().to(source_dtype)
    sample_input = torch.ones(1, 3, 8, 8, dtype=source_dtype)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )
    exported_model = _export_model(model, sample_input, weights_dtype, onnx_opset=23)
    expected_dtype = _ONNX_DTYPE_BY_NAME[weights_dtype]

    onnx.checker.check_model(exported_model)
    inferred_model = onnx.shape_inference.infer_shapes(
        exported_model, check_type=True, strict_mode=True
    )
    _assert_runtime_io_dtype(exported_model, expected_dtype)
    assert not any(
        node.op_type in {"TRT_FP8QuantizeLinear", "TRT_FP8DequantizeLinear"}
        for node in exported_model.graph.node
    )

    initializer_map = {
        initializer.name: initializer for initializer in exported_model.graph.initializer
    }
    conv = next(node for node in exported_model.graph.node if node.op_type == "Conv")
    weight_dq = next(
        node
        for node in exported_model.graph.node
        if node.op_type == "DequantizeLinear" and node.output[0] == conv.input[1]
    )
    weight = initializer_map[weight_dq.input[0]]
    assert weight.data_type == onnx.TensorProto.FLOAT8E4M3FN
    assert initializer_map[weight_dq.input[1]].data_type == expected_dtype
    assert _tensor_dtype_map(inferred_model)[weight_dq.output[0]] == expected_dtype
    assert not any(node.op_type == "Cast" for node in exported_model.graph.node)


class MixedPrecisionLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fp32_weight = nn.Parameter(torch.eye(4, dtype=torch.float32))
        self.bf16_weight = nn.Parameter(torch.eye(4, dtype=torch.bfloat16))

    def forward(self, x):
        return torch.matmul(x, self.fp32_weight) + torch.matmul(x, self.bf16_weight)


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_onnx_dtype"),
    [
        (torch.float32, "bf16", onnx.TensorProto.BFLOAT16),
        (torch.bfloat16, "fp32", onnx.TensorProto.FLOAT),
        (torch.float32, "fp16", onnx.TensorProto.FLOAT16),
    ],
)
def test_parameterless_model_uses_explicit_weights_dtype(
    source_dtype, weights_dtype, expected_onnx_dtype
):
    exported_model = _export_model(
        nn.Identity(), torch.ones(1, 4, dtype=source_dtype), weights_dtype
    )

    graph_io = [*exported_model.graph.input, *exported_model.graph.output]
    assert all(value.type.tensor_type.elem_type == expected_onnx_dtype for value in graph_io)


def test_mixed_parameter_model_uses_explicit_weights_dtype():
    model = MixedPrecisionLinear()
    exported_model = _export_model(model, torch.ones(1, 4), "bf16")

    assert model.fp32_weight.dtype == torch.float32
    assert model.bf16_weight.dtype == torch.bfloat16
    assert exported_model.graph.initializer
    assert all(
        initializer.data_type == onnx.TensorProto.BFLOAT16
        for initializer in exported_model.graph.initializer
    )


def test_onnx_quantizer_precision_is_restored_after_failure():
    quantizers = nn.ModuleList([TensorQuantizer(), TensorQuantizer()])
    quantizers[0].trt_high_precision_dtype = "Float"
    del quantizers[1]._trt_high_precision_dtype

    with _override_onnx_quantizer_precision(quantizers, None):
        assert quantizers[0].trt_high_precision_dtype == "Float"
        assert not hasattr(quantizers[1], "_trt_high_precision_dtype")
    assert quantizers[0].trt_high_precision_dtype == "Float"
    assert not hasattr(quantizers[1], "_trt_high_precision_dtype")

    with _override_onnx_quantizer_precision(quantizers, "Half"):
        assert all(q.trt_high_precision_dtype == "Half" for q in quantizers)
    assert quantizers[0].trt_high_precision_dtype == "Float"
    assert not hasattr(quantizers[1], "_trt_high_precision_dtype")

    with (
        pytest.raises(RuntimeError, match="export failed"),
        _override_onnx_quantizer_precision(quantizers, "BFloat16"),
    ):
        assert all(q.trt_high_precision_dtype == "BFloat16" for q in quantizers)
        raise RuntimeError("export failed")

    assert quantizers[0].trt_high_precision_dtype == "Float"
    assert not hasattr(quantizers[1], "_trt_high_precision_dtype")


def _identity_onnx_model(dtype=onnx.TensorProto.FLOAT):
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [onnx.helper.make_tensor_value_info("input", dtype, [1, 4])],
        [onnx.helper.make_tensor_value_info("output", dtype, [1, 4])],
    )
    return onnx.helper.make_model(graph)


@pytest.mark.parametrize("weights_dtype", ["fp32", "fp16", "bf16"])
def test_onnx_load_path_rejects_non_native_weights_dtype(tmp_path, weights_dtype):
    onnx_path = tmp_path / "identity.onnx"
    onnx.save(_identity_onnx_model(), onnx_path)

    with pytest.raises(ValueError, match="weights_dtype must be 'native'"):
        get_onnx_bytes_and_metadata(
            nn.Identity(),
            (torch.ones(1, 4),),
            onnx_load_path=str(onnx_path),
            weights_dtype=weights_dtype,
        )


def test_onnx_load_path_preserves_native_model(tmp_path):
    onnx_path = tmp_path / "identity.onnx"
    onnx.save(_identity_onnx_model(), onnx_path)

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        nn.Identity(),
        (torch.ones(1, 4),),
        onnx_load_path=str(onnx_path),
        weights_dtype="native",
    )

    loaded = OnnxBytes.from_bytes(onnx_bytes)
    assert onnx.load_model_from_string(loaded.get_onnx_model_file_bytes())


def _make_bf16_tensor(name, values):
    values = np.asarray(values, dtype=np.float32)
    tensor = onnx.TensorProto(name=name, data_type=onnx.TensorProto.BFLOAT16)
    tensor.dims.extend(values.shape)
    tensor.raw_data = (values.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    return tensor


def test_convert_to_fp32_recurses_through_graphs_functions_and_attributes():
    branch_value = _make_bf16_tensor("branch_value", [2.0])
    weight = _make_bf16_tensor("weight", [1.0])
    weight.doc_string = "weight metadata"
    weight.metadata_props.add(key="source", value="test")
    branch = onnx.helper.make_graph(
        [onnx.helper.make_node("Constant", [], ["branch_output"], value=branch_value)],
        "branch",
        [],
        [onnx.helper.make_tensor_value_info("branch_output", onnx.TensorProto.BFLOAT16, [1])],
    )
    custom_node = onnx.helper.make_node(
        "CustomOp",
        ["input"],
        ["custom_output"],
        domain="test",
        dtype=onnx.TensorProto.DOUBLE,
        output_dtype=onnx.TensorProto.BFLOAT16,
    )
    custom_node.attribute.append(
        onnx.helper.make_attribute(
            "type", onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT16, [1])
        )
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "If", ["condition"], ["output"], then_branch=branch, else_branch=branch
            ),
            onnx.helper.make_node("Cast", ["input"], ["cast_output"], to=onnx.TensorProto.FLOAT16),
            custom_node,
        ],
        "recursive",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.BFLOAT16, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.DOUBLE, [1])],
        initializer=[
            onnx.numpy_helper.from_array(np.array(True), "condition"),
            weight,
            onnx.numpy_helper.from_array(np.array([3.0], dtype=np.float64), "double_weight"),
        ],
        value_info=[
            onnx.helper.make_tensor_value_info("custom_output", onnx.TensorProto.FLOAT16, [1])
        ],
    )
    function = onnx.helper.make_function(
        "test",
        "LocalCast",
        ["x"],
        ["y"],
        [onnx.helper.make_node("Cast", ["x"], ["y"], to=onnx.TensorProto.BFLOAT16)],
        [onnx.helper.make_opsetid("", 20)],
        value_info=[onnx.helper.make_tensor_value_info("y", onnx.TensorProto.BFLOAT16, [1])],
    )
    model = onnx.helper.make_model(graph, functions=[function])

    assert convert_to_fp32(model) is model

    graph = model.graph
    function = model.functions[0]
    custom_node = next(node for node in graph.node if node.op_type == "CustomOp")
    assert all(
        value.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        for value in (*graph.input, *graph.output, *graph.value_info)
    )
    initializer_map = {initializer.name: initializer for initializer in graph.initializer}
    assert initializer_map["weight"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["double_weight"].data_type == onnx.TensorProto.FLOAT
    assert initializer_map["weight"].doc_string == "weight metadata"
    assert initializer_map["weight"].metadata_props[0].key == "source"
    np.testing.assert_array_equal(
        np.frombuffer(initializer_map["weight"].raw_data, dtype=np.float32),
        np.array([1.0], dtype=np.float32),
    )
    attribute_map = {attribute.name: attribute for attribute in custom_node.attribute}
    assert attribute_map["dtype"].i == onnx.TensorProto.DOUBLE
    assert attribute_map["output_dtype"].i == onnx.TensorProto.BFLOAT16
    assert attribute_map["type"].tp.tensor_type.elem_type == onnx.TensorProto.FLOAT
    for attribute in graph.node[0].attribute:
        branch_graph = attribute.g
        assert branch_graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        assert branch_graph.node[0].attribute[0].t.data_type == onnx.TensorProto.FLOAT
    assert function.value_info[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert function.node[0].attribute[0].i == onnx.TensorProto.FLOAT


def test_convert_to_fp32_handles_vetted_dtype_attributes():
    nodes = [
        onnx.helper.make_node("HannWindow", [], [], output_datatype=onnx.TensorProto.FLOAT16),
        onnx.helper.make_node("LayerNormalization", [], [], stash_type=onnx.TensorProto.BFLOAT16),
        onnx.helper.make_node("Attention", [], [], softmax_precision=onnx.TensorProto.DOUBLE),
        onnx.helper.make_node("QuantizeLinear", [], [], precision=onnx.TensorProto.FLOAT16),
        onnx.helper.make_node("Cast", [], [], to=onnx.TensorProto.BFLOAT16),
        onnx.helper.make_node(
            "TRTCustom", [], [], domain="trt", output_dtype=onnx.TensorProto.BFLOAT16
        ),
        onnx.helper.make_node(
            "CustomOp",
            [],
            [],
            domain="test",
            dtype=onnx.TensorProto.BFLOAT16,
            precision=onnx.TensorProto.BFLOAT16,
            to=onnx.TensorProto.BFLOAT16,
        ),
    ]
    model = onnx.helper.make_model(onnx.helper.make_graph(nodes, "dtype_attributes", [], []))

    convert_to_fp32(model)

    for node in model.graph.node[:6]:
        assert node.attribute[0].i == onnx.TensorProto.FLOAT
    assert all(attr.i == onnx.TensorProto.BFLOAT16 for attr in model.graph.node[6].attribute)


def _make_function_with_referenced_to(op_type):
    to_attribute = onnx.AttributeProto(
        name="to",
        ref_attr_name="target_dtype",
        type=onnx.AttributeProto.INT,
    )
    node = onnx.helper.make_node(op_type, ["x"], ["y"])
    node.attribute.append(to_attribute)
    return onnx.helper.make_function(
        "test",
        f"Referenced{op_type}",
        ["x"],
        ["y"],
        [node],
        [onnx.helper.make_opsetid("", 26)],
        attribute_protos=[onnx.helper.make_attribute("target_dtype", onnx.TensorProto.BFLOAT16)],
    )


def test_convert_to_fp32_converts_function_cast_attribute_default():
    function = _make_function_with_referenced_to("Cast")
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "function_cast", [], []), functions=[function]
    )

    convert_to_fp32(model)

    assert model.functions[0].attribute_proto[0].i == onnx.TensorProto.FLOAT


def test_convert_to_fp32_rejects_function_bitcast_attribute_default():
    function = _make_function_with_referenced_to("BitCast")
    model = onnx.helper.make_model(
        onnx.helper.make_graph([], "function_bitcast", [], []), functions=[function]
    )

    with pytest.raises(ValueError, match="BitCast targets cannot be converted safely"):
        convert_to_fp32(model)


def test_convert_to_fp32_rejects_low_precision_bitcast():
    bitcast = onnx.helper.make_node("BitCast", ["input"], ["output"], to=onnx.TensorProto.FLOAT16)
    graph = onnx.helper.make_graph(
        [bitcast],
        "bitcast",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT16, [1])],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [1])],
    )

    with pytest.raises(ValueError, match="BitCast targets cannot be converted safely"):
        convert_to_fp32(onnx.helper.make_model(graph))


def test_convert_to_fp32_rejects_segmented_tensor():
    weight = onnx.numpy_helper.from_array(np.array([1.0], dtype=np.float16), "weight")
    weight.segment.begin = 0
    weight.segment.end = 1
    graph = onnx.helper.make_graph(
        [],
        "segmented",
        [],
        [],
        initializer=[weight],
    )

    with pytest.raises(ValueError, match="Segmented tensors are not supported"):
        convert_to_fp32(onnx.helper.make_model(graph))


def test_convert_to_fp32_handles_loaded_external_data(tmp_path):
    tensor = onnx.numpy_helper.from_array(np.array([1.0, -2.0], dtype=np.float16), "weight")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["weight"], ["output"])],
        "external",
        [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT16, [2])],
        [tensor],
    )
    onnx_path = tmp_path / "external.onnx"
    onnx.save_model(
        onnx.helper.make_model(graph),
        onnx_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="weights.data",
        size_threshold=0,
    )

    unloaded_model = onnx.load(onnx_path, load_external_data=False)
    with pytest.raises(ValueError, match="External tensor data must be loaded"):
        convert_to_fp32(unloaded_model)

    loaded_model = onnx.load(onnx_path, load_external_data=True)
    convert_to_fp32(loaded_model)
    converted_weight = loaded_model.graph.initializer[0]
    assert converted_weight.data_type == onnx.TensorProto.FLOAT
    assert converted_weight.data_location == onnx.TensorProto.DEFAULT
    assert not converted_weight.external_data
    np.testing.assert_array_equal(
        onnx.numpy_helper.to_array(converted_weight), np.array([1.0, -2.0], dtype=np.float32)
    )


def test_save_onnx_model_externalizes_large_attribute_and_replaces_shards(tmp_path, monkeypatch):
    values = np.arange(300, dtype=np.float32)
    constant = onnx.helper.make_node(
        "Constant",
        [],
        ["output"],
        value=onnx.numpy_helper.from_array(values),
    )
    graph = onnx.helper.make_graph(
        [constant],
        "external_attribute",
        [],
        [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [300])],
    )
    model = onnx.helper.make_model(graph)
    onnx_path = tmp_path / "model.onnx"
    onnx.save_model(model, onnx_path)
    external_data_path = tmp_path / "model.onnx_data"
    external_data_path.write_bytes(b"stale")
    previous_shard = tmp_path / "previous.data"
    previous_shard.write_bytes(b"old")
    monkeypatch.setattr(torch_onnx, "TWO_GB", 1)

    torch_onnx._save_onnx_model(model, str(onnx_path), "model")

    assert external_data_path.stat().st_size == values.nbytes
    assert not previous_shard.exists()
    unloaded_model = onnx.load(onnx_path, load_external_data=False)
    value = next(attr.t for attr in unloaded_model.graph.node[0].attribute if attr.name == "value")
    assert onnx.external_data_helper.uses_external_data(value)
    assert next(prop.value for prop in value.external_data if prop.key == "location") == (
        "model.onnx_data"
    )
    loaded_model = onnx.load(onnx_path, load_external_data=True)
    loaded_value = next(
        attr.t for attr in loaded_model.graph.node[0].attribute if attr.name == "value"
    )
    np.testing.assert_array_equal(onnx.numpy_helper.to_array(loaded_value), values)


class SingleArgModel(nn.Module):
    def forward(self, x: torch.Tensor):
        return torch.add(x, x) - x


class DoubleArgModel(nn.Module):
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return torch.add(x, y) - x


@pytest.mark.parametrize(
    ("model", "n_args", "batch_size"),
    [
        (SingleArgModel(), 1, 1),
        (SingleArgModel(), 1, 2),
        (DoubleArgModel(), 2, 1),
        (DoubleArgModel(), 2, 2),
    ],
)
def test_get_and_validate_batch_size(model, n_args, batch_size):
    inputs = (torch.randn([batch_size, 3, 32, 32]),) * n_args
    onnx_bytes, _ = get_onnx_bytes_and_metadata(model, inputs)
    onnx_bytes_obj = OnnxBytes.from_bytes(onnx_bytes)
    onnx_bytes = onnx_bytes_obj.onnx_model[f"{onnx_bytes_obj.model_name}.onnx"]

    assert validate_batch_size(onnx_bytes, batch_size)
    assert validate_batch_size(onnx_bytes, 3) is False

    assert batch_size == get_batch_size_from_bytes(onnx_bytes)
