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

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")
from _test_utils.torch.misc import set_seed
from _test_utils.torch.quantization.models import SimpleLinear
from _test_utils.torch.quantization.onnx_export import TEST_MODELS, onnx_export_tester
from onnx import TensorProto, helper, numpy_helper

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.tensor_quant as tensor_quant
from modelopt.onnx import utils
from modelopt.onnx.export import NVFP4QuantExporter
from modelopt.onnx.export.nvfp4_exporter import _encode_nvfp4_block_scale
from modelopt.onnx.quantization.qdq_utils import fp4qdq_to_2dq
from modelopt.torch._deploy.utils import OnnxBytes, get_onnx_bytes_and_metadata
from modelopt.torch.quantization.qtensor import NVFP4QTensor
from modelopt.torch.quantization.utils import is_quantized_linear


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


class _NVFP4LinearWithExplicitBias(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.linear = torch.nn.Linear(16, 16, bias=False, dtype=dtype)
        self.bias = torch.nn.Parameter(torch.ones(16, dtype=dtype))

    def forward(self, inputs):
        return self.linear(inputs) + self.bias


def _make_cpu_nvfp4_model(monkeypatch, model, sample_input, disable_input_quantizers=False):
    def forward_loop(model):
        model(sample_input)

    def cpu_dynamic_block_quantize(inputs, *args):
        return inputs

    monkeypatch.setattr(tensor_quant, "dynamic_block_quantize_op", cpu_dynamic_block_quantize)
    model = mtq.quantize(model, mtq.NVFP4_DEFAULT_CFG, forward_loop=forward_loop)

    for module in model.modules():
        assert not isinstance(module, torch.nn.Linear) or is_quantized_linear(module)
        if isinstance(module, torch.nn.Linear):
            if disable_input_quantizers:
                module.input_quantizer.disable()
            module.weight_quantizer._onnx_quantizer_type = "static"

    return model


def test_nvfp4_exported_onnx_is_topologically_sorted(monkeypatch):
    model = SimpleLinear().eval()
    sample_input = model.get_input()
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input, disable_input_quantizers=True)

    buffer = io.BytesIO()
    if "enable_onnx_checker" in inspect.signature(torch.onnx.export).parameters:
        kwargs = {"enable_onnx_checker": False}
    else:
        kwargs = {}

    torch.onnx.export(
        model,
        sample_input,
        buffer,
        input_names=["input"],
        output_names=["output"],
        export_params=True,
        opset_version=21,
        dynamo=False,
        **kwargs,
    )

    buffer.seek(0)
    exported_model = onnx.load_model_from_string(buffer.read())
    assert any(node.op_type == "TRT_FP4QDQ" for node in exported_model.graph.node)

    converted_model = NVFP4QuantExporter.process_model(exported_model)
    assert not any(node.op_type == "TRT_FP4QDQ" for node in converted_model.graph.node)
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize(
    ("source_dtype", "weights_dtype", "expected_dtype"),
    [
        (torch.float32, "fp32", TensorProto.FLOAT),
        (torch.float32, "fp16", TensorProto.FLOAT16),
        (torch.bfloat16, "bf16", TensorProto.BFLOAT16),
    ],
    ids=["fp32-preserved", "fp32-to-fp16", "bf16-noop"],
)
def test_nvfp4_deploy_export_has_consistent_elementwise_types(
    monkeypatch, source_dtype, weights_dtype, expected_dtype
):
    model = _NVFP4LinearWithExplicitBias(source_dtype).eval()
    sample_input = torch.ones(1, 2, 16, dtype=source_dtype)
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input)

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        weights_dtype=weights_dtype,
    )
    exported_model = onnx.load_model_from_string(
        OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    )

    assert utils.get_opset_version(exported_model) >= 23
    onnx.checker.check_model(exported_model, full_check=True)
    assert any(
        node.op_type == "DequantizeLinear"
        and any(attribute.name == "block_size" for attribute in node.attribute)
        for node in exported_model.graph.node
    )
    assert any(node.op_type == "TRT_FP4DynamicQuantize" for node in exported_model.graph.node)

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

    floating_types = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.BFLOAT16}
    elementwise_nodes = [
        node
        for node in inferred_model.graph.node
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
    ("source_dtype", "weights_dtype", "expected_gemm_dtype", "expected_output_dtype"),
    [
        (torch.float32, "fp32", TensorProto.FLOAT16, TensorProto.FLOAT),
        (torch.float32, "fp16", TensorProto.FLOAT16, TensorProto.FLOAT16),
        (torch.bfloat16, "bf16", TensorProto.BFLOAT16, TensorProto.BFLOAT16),
    ],
    ids=["fp32-preserved", "fp32-to-fp16", "bf16-noop"],
)
def test_nvfp4_deploy_export_has_consistent_gemm_types(
    monkeypatch, source_dtype, weights_dtype, expected_gemm_dtype, expected_output_dtype
):
    model = torch.nn.Linear(16, 16, dtype=source_dtype).eval()
    sample_input = torch.ones(2, 16, dtype=source_dtype)
    model = _make_cpu_nvfp4_model(monkeypatch, model, sample_input)

    onnx_bytes, _ = get_onnx_bytes_and_metadata(
        model,
        (sample_input,),
        weights_dtype=weights_dtype,
    )
    exported_model = onnx.load_model_from_string(
        OnnxBytes.from_bytes(onnx_bytes).get_onnx_model_file_bytes()
    )

    assert utils.get_opset_version(exported_model) >= 23
    onnx.checker.check_model(exported_model, full_check=True)
    assert any(node.op_type == "TRT_FP4DynamicQuantize" for node in exported_model.graph.node)
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

    gemm_node = next(node for node in inferred_model.graph.node if node.op_type == "Gemm")
    assert [tensor_types[input_name] for input_name in gemm_node.input] == [expected_gemm_dtype] * 3
    assert tensor_types[gemm_node.output[0]] == expected_gemm_dtype
    assert tensor_types[inferred_model.graph.output[0].name] == expected_output_dtype


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
