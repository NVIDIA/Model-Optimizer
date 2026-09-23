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

"""Unit tests for ONNX export for CPU quantization."""

import inspect
import io
from copy import deepcopy

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.onnx_export import TEST_MODELS, onnx_export_tester
from onnx import TensorProto, helper, numpy_helper

import modelopt.torch._deploy.utils.torch_onnx as torch_onnx
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.tensor_quant as tensor_quant
from modelopt.onnx import utils
from modelopt.onnx.export import NVFP4QuantExporter
from modelopt.onnx.export.nvfp4_exporter import _encode_nvfp4_block_scale
from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import is_quantized_linear


def _export_to_onnx(model, sample_input, **kwargs):
    buffer = io.BytesIO()
    if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
        kwargs["enable_onnx_checker"] = False
    torch.onnx.export(model, sample_input, buffer, dynamo=False, **kwargs)
    buffer.seek(0)
    return onnx.load_model_from_string(buffer.read())


@pytest.mark.parametrize("model_cls", TEST_MODELS)
@pytest.mark.parametrize(
    ("num_bits", "per_channel_quantization", "constant_folding"),
    [
        (8, True, True),
        (8, False, True),
        (8, True, False),
        (8, False, False),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_onnx_export_cpu(model_cls, num_bits, per_channel_quantization, constant_folding, dtype):
    # TODO: ORT output correctness tests sometimes fails due to random seed.
    # It needs to be investigated closer (lower priority). Lets set a seed for now.
    set_seed(90)
    onnx_export_tester(
        model_cls(), "cpu", num_bits, per_channel_quantization, constant_folding, dtype
    )


def test_fp8_conv_export_preserves_custom_qdq_and_kernel_shape():
    model = torch.nn.Conv2d(3, 4, 3, bias=False).eval()
    sample_input = torch.randn(1, 3, 8, 8)
    model = mtq.quantize(
        model,
        mtq.FP8_DEFAULT_CFG,
        forward_loop=lambda quantized_model: quantized_model(sample_input),
    )

    exported_model = _export_to_onnx(model, sample_input, opset_version=20)
    producers = {output: node for node in exported_model.graph.node for output in node.output}
    conv = next(node for node in exported_model.graph.node if node.op_type == "Conv")

    for conv_input in conv.input[:2]:
        dequantize = producers[conv_input]
        quantize = producers[dequantize.input[0]]
        assert dequantize.op_type == "TRT_FP8DequantizeLinear"
        assert quantize.op_type == "TRT_FP8QuantizeLinear"

    value_info = {value.name: value for value in exported_model.graph.value_info}
    weight_dequantize = producers[conv.input[1]]
    weight_quantize = producers[weight_dequantize.input[0]]
    for value_name in (*weight_quantize.output, *weight_dequantize.output):
        shape = [
            dimension.dim_value for dimension in value_info[value_name].type.tensor_type.shape.dim
        ]
        assert shape == [4, 3, 3, 3]

    kernel_shape = next(
        attribute for attribute in conv.attribute if attribute.name == "kernel_shape"
    )
    assert list(kernel_shape.ints) == [3, 3]
    onnx.checker.check_model(exported_model)


class _NVFP4LinearWithExplicitBias(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False, dtype=dtype)
        self.norm = torch.nn.LayerNorm(16, dtype=dtype)
        self.projection = torch.nn.Linear(16, 16, bias=False, dtype=dtype)
        self.bias = torch.nn.Parameter(torch.ones(16, dtype=dtype))

    def forward(self, inputs):
        return self.projection(self.norm(self.linear(inputs))) + self.bias


class _NVFP4MixedPrecisionLinear(torch.nn.Module):
    def __init__(self, return_boundaries=False):
        super().__init__()
        self.fp32_linear = torch.nn.Linear(16, 16, dtype=torch.float32)
        self.bf16_linear = torch.nn.Linear(16, 16, dtype=torch.bfloat16)
        self.return_boundaries = return_boundaries

    def forward(self, inputs):
        fp32_output = self.fp32_linear(inputs)
        bf16_output = self.bf16_linear(inputs.to(torch.bfloat16))
        bf16_output_as_fp32 = bf16_output.float()
        output = fp32_output + bf16_output_as_fp32
        return (output, bf16_output, bf16_output_as_fp32) if self.return_boundaries else output


def _make_cpu_nvfp4_model(
    monkeypatch, model, sample_input, disable_input_quantizers=False, quant_config=None
):
    def forward_loop(model):
        model(sample_input)

    def cpu_dynamic_block_quantize(inputs, *args):
        return inputs

    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", cpu_dynamic_block_quantize)
    model = mtq.quantize(model, quant_config or mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)

    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
        if isinstance(module, torch.nn.Linear):
            if disable_input_quantizers:
                module.input_quantizer.disable()
            module.weight_quantizer._onnx_quantizer_type = "static"

    return model


def _export_deploy_onnx_with_types(model, sample_input, weights_dtype, dynamic_axes=None):
    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        weights_dtype=weights_dtype,
        dynamic_axes=dynamic_axes or {},
    )
    exported_model = onnx.load_model_from_string(
        OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    )
    onnx.checker.check_model(exported_model, full_check=True)
    inferred_model = onnx.shape_inference.infer_shapes(exported_model, strict_mode=True)
    tensor_types = {
        initializer.name: initializer.data_type for initializer in inferred_model.graph.initializer
    }
    for value in [
        *inferred_model.graph.input,
        *inferred_model.graph.value_info,
        *inferred_model.graph.output,
    ]:
        if value.type.HasField("tensor_type"):
            tensor_types[value.name] = value.type.tensor_type.elem_type
    return exported_model, tensor_types


def test_nvfp4_exported_onnx_is_topologically_sorted(monkeypatch):
    model = SimpleLinear().eval()
    sample_input = model.get_input()
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input, disable_input_quantizers=True)

    exported_model = _export_to_onnx(
        model,
        sample_input,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=21,
    )
    assert any(node.op_type == "TRT_FP4QDQ" for node in exported_model.graph.node)

    converted_model = NVFP4QuantExporter.process_model(exported_model)
    assert not any(node.op_type == "TRT_FP4QDQ" for node in converted_model.graph.node)
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_dtype"),
    [
        (torch.float32, "fp32", TensorProto.FLOAT),
        (torch.float32, "fp16", TensorProto.FLOAT16),
        (torch.float32, "bf16", TensorProto.BFLOAT16),
        (torch.bfloat16, "fp16", TensorProto.FLOAT16),
        (torch.bfloat16, "bf16", TensorProto.BFLOAT16),
    ],
    ids=["fp32-preserved", "fp32-to-fp16", "fp32-to-bf16", "bf16-to-fp16", "bf16-noop"],
)
def test_nvfp4_deploy_export_has_consistent_elementwise_types(
    monkeypatch, source_dtype, weights_dtype, expected_dtype
):
    model = _NVFP4LinearWithExplicitBias(source_dtype).eval()
    sample_input = torch.ones(1, 2, 16, dtype=source_dtype)
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input)
    # Swin leaves LayerNorm input quantizers disabled on its high-rank activation paths.
    model.norm.input_quantizer.disable()

    exported_model, tensor_types = _export_deploy_onnx_with_types(
        model, sample_input, weights_dtype
    )

    assert utils.get_opset_version(exported_model) >= 23
    assert exported_model.graph.input[0].type.tensor_type.elem_type == expected_dtype
    assert exported_model.graph.output[0].type.tensor_type.elem_type == expected_dtype
    assert any(
        node.op_type == "DequantizeLinear"
        and any(attribute.name == "block_size" for attribute in node.attribute)
        for node in exported_model.graph.node
    )
    assert any(node.op_type == "TRT_FP4DynamicQuantize" for node in exported_model.graph.node)

    dynamic_quantize = next(
        node for node in exported_model.graph.node if node.op_type == "TRT_FP4DynamicQuantize"
    )
    assert [tensor_types[output] for output in dynamic_quantize.output] == [
        TensorProto.FLOAT4E2M1,
        TensorProto.FLOAT8E4M3FN,
    ]

    floating_types = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}
    elementwise_nodes = [
        node
        for node in exported_model.graph.node
        if node.op_type in {"Add", "Sub", "Mul", "Div", "Pow"}
    ]
    assert any(node.op_type == "Add" for node in elementwise_nodes)
    for node in elementwise_nodes:
        input_types = [tensor_types[input_name] for input_name in node.input]
        assert len(set(input_types) & floating_types) <= 1, (
            node.name,
            [TensorProto.DataType.Name(input_type) for input_type in input_types],
        )

    add_node = next(node for node in elementwise_nodes if node.op_type == "Add")
    assert [tensor_types[input_name] for input_name in add_node.input] == [
        expected_dtype,
        expected_dtype,
    ]


@pytest.mark.parametrize(
    ("weights_dtype", "expected_dtype"),
    [
        ("fp32", TensorProto.FLOAT),
        ("fp16", TensorProto.FLOAT16),
        ("bf16", TensorProto.BFLOAT16),
    ],
)
@pytest.mark.parametrize("with_fp8_branch", [False, True])
def test_nvfp4_deploy_export_preserves_mixed_precision_boundaries(
    monkeypatch, weights_dtype, expected_dtype, with_fp8_branch
):
    model = _NVFP4MixedPrecisionLinear().eval()
    sample_input = torch.ones(1, 16)
    quant_config = deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if with_fp8_branch:
        quant_config["quant_cfg"].append(
            {"quantizer_name": "bf16_linear*quantizer", "cfg": {"num_bits": (4, 3), "axis": None}}
        )
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input, quant_config=quant_config)

    quantized_types = {TensorProto.FLOAT4E2M1, TensorProto.FLOAT8E4M3FN}
    expected_payloads = {}
    quantize_weights = torch_onnx.quantize_weights

    def quantized_payloads(graph):
        return {
            initializer.name: (
                initializer.data_type,
                tuple(initializer.dims),
                initializer.raw_data,
            )
            for initializer in graph.graph.initializer
            if initializer.data_type in quantized_types
        }

    def capture_quantized_payloads(model, graph):
        graph = quantize_weights(model, graph)
        expected_payloads.update(quantized_payloads(graph))
        return graph

    monkeypatch.setattr(torch_onnx, "quantize_weights", capture_quantized_payloads)
    exported_model, tensor_types = _export_deploy_onnx_with_types(
        model,
        sample_input,
        weights_dtype,
        dynamic_axes=(
            {"inputs": {0: "batch"}, "out": {0: "batch"}} if weights_dtype != "fp32" else None
        ),
    )
    assert {payload[0] for payload in expected_payloads.values()} == quantized_types
    assert all(payload[2] for payload in expected_payloads.values())
    assert quantized_payloads(exported_model) == expected_payloads
    assert exported_model.graph.input[0].type.tensor_type.elem_type == expected_dtype
    assert exported_model.graph.output[0].type.tensor_type.elem_type == expected_dtype
    if weights_dtype != "fp32":
        for value in [*exported_model.graph.input, *exported_model.graph.output]:
            assert value.type.tensor_type.shape.dim[0].dim_param == "batch"
            assert value.type.tensor_type.shape.dim[1].dim_value == 16

    add_node = next(node for node in exported_model.graph.node if node.op_type == "Add")
    assert [tensor_types[input_name] for input_name in add_node.input] == [
        expected_dtype,
        expected_dtype,
    ]
    if with_fp8_branch and weights_dtype != "fp32":
        quantize_nodes = [
            node for node in exported_model.graph.node if node.op_type == "QuantizeLinear"
        ]
        assert quantize_nodes
        for node in quantize_nodes:
            assert tensor_types[node.input[1]] == expected_dtype


def test_nvfp4_deploy_export_preserves_bf16_graph_outputs(monkeypatch):
    model = _NVFP4MixedPrecisionLinear(return_boundaries=True).eval()
    sample_input = torch.ones(1, 16)
    with torch.no_grad():
        source_outputs = model(sample_input)
    assert [output.dtype for output in source_outputs] == [
        torch.float32,
        torch.bfloat16,
        torch.float32,
    ]
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input)
    exported_model, _ = _export_deploy_onnx_with_types(model, sample_input, "fp32")
    assert [value.type.tensor_type.elem_type for value in exported_model.graph.output] == [
        TensorProto.FLOAT,
        TensorProto.BFLOAT16,
        TensorProto.FLOAT,
    ]

    producers = {output: node for node in exported_model.graph.node for output in node.output}
    native_output = exported_model.graph.output[1].name
    cast_output = exported_model.graph.output[2].name
    assert producers[native_output].op_type in {"Gemm", "MatMul"}
    cast_node = producers[cast_output]
    assert cast_node.op_type == "Cast"
    assert list(cast_node.input) == [native_output]
    assert helper.get_attribute_value(next(a for a in cast_node.attribute if a.name == "to")) == (
        TensorProto.FLOAT
    )


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_gemm_dtype", "expected_output_dtype"),
    [
        (torch.float32, "fp32", TensorProto.FLOAT16, TensorProto.FLOAT),
        (torch.float32, "fp16", TensorProto.FLOAT16, TensorProto.FLOAT16),
        (torch.float32, "bf16", TensorProto.BFLOAT16, TensorProto.BFLOAT16),
        (torch.bfloat16, "fp16", TensorProto.FLOAT16, TensorProto.FLOAT16),
        (torch.bfloat16, "bf16", TensorProto.BFLOAT16, TensorProto.BFLOAT16),
    ],
    ids=["fp32-preserved", "fp32-to-fp16", "fp32-to-bf16", "bf16-to-fp16", "bf16-noop"],
)
def test_nvfp4_deploy_export_has_consistent_gemm_types(
    monkeypatch, source_dtype, weights_dtype, expected_gemm_dtype, expected_output_dtype
):
    model = torch.nn.Linear(16, 16, dtype=source_dtype).eval()
    sample_input = torch.ones(2, 16, dtype=source_dtype)
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input)

    exported_model, tensor_types = _export_deploy_onnx_with_types(
        model, sample_input, weights_dtype
    )

    assert utils.get_opset_version(exported_model) >= 23
    assert any(node.op_type == "TRT_FP4DynamicQuantize" for node in exported_model.graph.node)

    gemm_node = next(node for node in exported_model.graph.node if node.op_type == "Gemm")
    assert [tensor_types[input_name] for input_name in gemm_node.input] == [expected_gemm_dtype] * 3
    assert tensor_types[gemm_node.output[0]] == expected_gemm_dtype
    assert tensor_types[exported_model.graph.output[0].name] == expected_output_dtype


@pytest.mark.parametrize(
    ("convert", "deprecated"),
    [
        pytest.param(NVFP4QuantExporter.process_model, False, id="exporter"),
        pytest.param(fp4qdq_to_2dq, True, id="deprecated-shim"),
    ],
)
@pytest.mark.parametrize(
    "scale_case", ["fp8-rounding", "fp32-tensor-arithmetic", "fp32-block-arithmetic"]
)
def test_nvfp4_packed_weights_match_eager(scale_case, convert, deprecated):
    block_size = 16
    if scale_case == "fp8-rounding":
        weight = np.zeros((4, block_size), dtype=np.float16)
        weight[0, 0] = 6.0
        weight[1, :2] = [0.231689453125, 0.0297393798828125]
        expected_scale = np.array([[448.0], [18.0], [1.0], [1.0]], dtype=np.float32)
    elif scale_case == "fp32-tensor-arithmetic":
        weight = np.zeros((2, block_size), dtype=np.float16)
        weight[0, 0] = 3.296875
        weight[1, 0] = 2.099609375
        weight[1, -2:] = [-1.501953125, 0.6181640625]
        expected_scale = np.array([[448.0], [288.0]], dtype=np.float32)
    else:
        weight = np.zeros((2, block_size), dtype=np.float16)
        weight[:, 0] = [2.24609375, 2.515625]
        expected_scale = np.array([[384.0], [448.0]], dtype=np.float32)

    weight_dq = helper.make_tensor_value_info("weight_dq", TensorProto.FLOAT, list(weight.shape))
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "TRT_FP4QDQ",
                    ["weight"],
                    ["weight_dq"],
                    name="weight_qdq",
                    block_size=block_size,
                ),
                helper.make_node(
                    "MatMul",
                    ["activation", "weight_dq"],
                    ["output"],
                    name="matmul",
                ),
            ],
            "nvfp4_packed_weights",
            [
                helper.make_tensor_value_info(
                    "activation", TensorProto.FLOAT16, [1, weight.shape[0]]
                )
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, weight.shape[1]])],
            [numpy_helper.from_array(weight, "weight")],
            value_info=[weight_dq],
        ),
        opset_imports=[helper.make_opsetid("", 23)],
    )

    if deprecated:
        with pytest.warns(DeprecationWarning):
            converted = convert(model)
    else:
        converted = convert(model)
    eager_qtensor, eager_scale, _ = NVFP4QTensor.quantize(
        torch.from_numpy(weight.astype(np.float32)), block_size
    )

    np.testing.assert_array_equal(eager_scale.float().cpu().numpy(), expected_scale)

    fp8_scale = next(
        initializer
        for initializer in converted.graph.initializer
        if initializer.name == "weight_f8_scale"
    )
    np.testing.assert_array_equal(
        np.frombuffer(fp8_scale.raw_data, dtype=np.uint8),
        eager_scale.view(torch.uint8).cpu().numpy().reshape(-1),
    )

    fp4_weight = next(
        initializer
        for initializer in converted.graph.initializer
        if initializer.name == "weight_f4"
    )
    np.testing.assert_array_equal(
        np.frombuffer(fp4_weight.raw_data, dtype=np.uint8),
        eager_qtensor._quantized_data.cpu().numpy().reshape(-1),
    )


@pytest.mark.parametrize("scale", [np.nan, np.inf, -1.0])
def test_nvfp4_rejects_invalid_block_scale(scale):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _encode_nvfp4_block_scale(np.array([scale], dtype=np.float32))


def test_nvfp4_shared_activation_reuses_cast():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [4, 32])
    outputs = [
        helper.make_tensor_value_info("output0", TensorProto.FLOAT, [4, 64]),
        helper.make_tensor_value_info("output1", TensorProto.FLOAT, [4, 64]),
    ]
    nodes = []
    initializers = []
    value_info = []

    for index in range(2):
        weight_name = f"linear{index}.weight"
        fp4qdq_output = f"fp4qdq_output{index}"
        initializers.append(
            numpy_helper.from_array(
                np.linspace(-1.0, 1.0, num=32 * 64, dtype=np.float32).reshape(32, 64),
                weight_name,
            )
        )
        value_info.append(helper.make_tensor_value_info(fp4qdq_output, TensorProto.FLOAT, [32, 64]))
        nodes.extend(
            [
                helper.make_node(
                    "TRT_FP4QDQ",
                    inputs=[weight_name],
                    outputs=[fp4qdq_output],
                    name=f"weight{index}_fp4qdq",
                    block_size=16,
                ),
                helper.make_node(
                    "MatMul",
                    inputs=["input", fp4qdq_output],
                    outputs=[f"output{index}"],
                    name=f"matmul{index}",
                ),
            ]
        )

    model = helper.make_model(
        helper.make_graph(
            nodes,
            "shared_activation_nvfp4",
            [input_tensor],
            outputs,
            initializers,
            value_info=value_info,
        )
    )

    converted_model = NVFP4QuantExporter.process_model(model)
    activation_casts = [
        node
        for node in converted_model.graph.node
        if node.op_type == "Cast" and node.input == ["input"]
    ]
    assert len(activation_casts) == 1
    assert activation_casts[0].output == ["input_f16"]
    assert all(
        node.input[0] == "input_f16"
        for node in converted_model.graph.node
        if node.op_type == "MatMul"
    )
    onnx.checker.check_model(converted_model)


def test_topologically_sort_graph_nodes_accounts_for_subgraph_captures():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    cond_tensor = helper.make_tensor_value_info("cond", TensorProto.BOOL, [])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    then_output = helper.make_tensor_value_info("then_output", TensorProto.FLOAT, [1])
    else_output = helper.make_tensor_value_info("else_output", TensorProto.FLOAT, [1])

    then_graph = helper.make_graph(
        [helper.make_node("Identity", ["captured"], ["then_output"], name="then_use_captured")],
        "then_branch",
        [],
        [then_output],
    )
    else_graph = helper.make_graph(
        [helper.make_node("Identity", ["captured"], ["else_output"], name="else_use_captured")],
        "else_branch",
        [],
        [else_output],
    )
    if_node = helper.make_node(
        "If",
        ["cond"],
        ["output"],
        name="if_uses_captured",
        then_branch=then_graph,
        else_branch=else_graph,
    )
    producer = helper.make_node("Identity", ["input"], ["captured"], name="producer")
    model = helper.make_model(
        helper.make_graph(
            [if_node, producer],
            "outer_scope_capture",
            [input_tensor, cond_tensor],
            [output_tensor],
        )
    )

    utils.topologically_sort_graph_nodes(model.graph)

    assert [node.name for node in model.graph.node] == ["producer", "if_uses_captured"]
    onnx.checker.check_model(model)
