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

from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnx_graphsurgeon")
pytest.importorskip("diffusers")
from onnx import TensorProto, helper, numpy_helper

from examples.diffusers.quantization.onnx_utils import export as diffusion_export
from modelopt.onnx.export import NVFP4QuantExporter


class _Quantizer:
    def __init__(self, num_bits, *, enabled=True, amax=448.0):
        self._num_bits = num_bits
        self._amax = torch.tensor(amax)
        self.is_enabled = enabled

    @property
    def num_bits(self):
        return self._num_bits


def _make_raw_fp4_model(*, qdq_consumer="Conv", marker_block_size=16, marker_initializer=True):
    dynamic_scale = numpy_helper.from_array(
        np.array(1.0 / 2688.0, dtype=np.float32), "dynamic_scale_value"
    )
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["dynamic_quantize_scale"],
            name="dynamic_quantize_scale",
            value=dynamic_scale,
        ),
        helper.make_node(
            "Constant",
            [],
            ["dynamic_dequantize_scale"],
            name="dynamic_dequantize_scale",
            value=dynamic_scale,
        ),
    ]
    initializers = []
    inputs = [helper.make_tensor_value_info("activation", TensorProto.FLOAT16, [1, 16])]
    outputs = [helper.make_tensor_value_info("linear_output", TensorProto.FLOAT16, [1, 16])]
    value_info = [helper.make_tensor_value_info("fp4_weight_dq", TensorProto.FLOAT16, [16, 16])]

    if marker_initializer:
        initializers.append(
            numpy_helper.from_array(
                np.linspace(-1.0, 1.0, 16 * 16, dtype=np.float16).reshape(16, 16),
                "fp4_weight",
            )
        )
    nodes.extend(
        [
            helper.make_node(
                "TRT_FP4DynamicQuantize",
                ["activation", "dynamic_quantize_scale"],
                ["activation_fp4", "activation_scale_fp8"],
                name="activation_quantize",
                domain="trt",
                axis=-1,
                block_size=marker_block_size,
                scale_type=TensorProto.FLOAT8E4M3FN,
            ),
            helper.make_node(
                "DequantizeLinear",
                ["activation_scale_fp8", "dynamic_dequantize_scale"],
                ["activation_scale_dq"],
                name="activation_scale_dequantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["activation_fp4", "activation_scale_dq"],
                ["activation_dq"],
                name="activation_dequantize",
                axis=-1,
                block_size=marker_block_size,
            ),
            helper.make_node(
                "Cast",
                ["activation_dq"],
                ["activation_dq_fp16"],
                name="activation_cast",
                to=TensorProto.FLOAT16,
            ),
            helper.make_node(
                "TRT_FP4QDQ",
                ["fp4_weight"],
                ["fp4_weight_dq"],
                name="fp4_weight_qdq",
                domain="trt",
                block_size=marker_block_size,
            ),
            helper.make_node(
                "MatMul",
                ["activation_dq_fp16", "fp4_weight_dq"],
                ["linear_output"],
                name="fp4_matmul",
            ),
        ]
    )

    scale = numpy_helper.from_array(np.array(0.25, dtype=np.float16), "qdq_scale_value")
    zero = numpy_helper.from_array(np.array(0, dtype=np.int8), "qdq_zero_value")
    nodes.extend(
        [
            helper.make_node("Constant", [], ["qdq_scale"], name="qdq_scale", value=scale),
            helper.make_node("Constant", [], ["qdq_zero"], name="qdq_zero", value=zero),
            helper.make_node(
                "QuantizeLinear",
                ["qdq_weight", "qdq_scale", "qdq_zero"],
                ["quantized_weight"],
                name="weight_quantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized_weight", "qdq_scale", "qdq_zero"],
                ["qdq_weight_dq"],
                name="weight_dequantize",
            ),
            helper.make_node(
                "QuantizeLinear",
                ["qdq_activation", "qdq_scale", "qdq_zero"],
                ["quantized_activation"],
                name="activation_fp8_quantize",
            ),
            helper.make_node(
                "DequantizeLinear",
                ["quantized_activation", "qdq_scale", "qdq_zero"],
                ["qdq_activation_dq"],
                name="activation_fp8_dequantize",
            ),
        ]
    )

    if qdq_consumer == "Conv":
        initializers.append(
            numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float16), "qdq_weight")
        )
        inputs.append(
            helper.make_tensor_value_info("qdq_activation", TensorProto.FLOAT16, [1, 1, 2, 2])
        )
        outputs.append(
            helper.make_tensor_value_info("qdq_output", TensorProto.FLOAT16, [1, 1, 2, 2])
        )
        nodes.append(
            helper.make_node(
                "Conv",
                ["qdq_activation_dq", "qdq_weight_dq"],
                ["qdq_output"],
                name="fp8_conv",
            )
        )
    else:
        initializers.append(
            numpy_helper.from_array(np.ones((4, 4), dtype=np.float16), "qdq_weight")
        )
        inputs.append(helper.make_tensor_value_info("qdq_activation", TensorProto.FLOAT16, [1, 4]))
        outputs.append(helper.make_tensor_value_info("qdq_output", TensorProto.FLOAT16, [1, 4]))
        nodes.append(
            helper.make_node(
                qdq_consumer,
                ["qdq_activation_dq", "qdq_weight_dq"],
                ["qdq_output"],
                name=f"fp8_{qdq_consumer.lower()}",
            )
        )

    graph = helper.make_graph(
        nodes,
        "mixed_fp4_fp8",
        inputs,
        outputs,
        initializers,
        value_info=value_info,
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("trt", 1)],
    )


def _tensor_dtype(onnx_model, tensor_name):
    for initializer in onnx_model.graph.initializer:
        if initializer.name == tensor_name:
            return initializer.data_type
    node = next(node for node in onnx_model.graph.node if tensor_name in node.output)
    value = next(attribute for attribute in node.attribute if attribute.name == "value").t
    return value.data_type


def _insert_passthrough_before_weight_quantize(onnx_model):
    weight = next(
        initializer
        for initializer in onnx_model.graph.initializer
        if initializer.name == "qdq_weight"
    )
    weight.name = "qdq_weight_source"
    quantize_node = next(node for node in onnx_model.graph.node if node.name == "weight_quantize")
    onnx_model.graph.node.insert(
        0,
        helper.make_node("Identity", ["qdq_weight_source"], ["qdq_weight"], name="weight_identity"),
    )
    assert quantize_node.input[0] == "qdq_weight"


def _insert_weight_cast_before_quantize(onnx_model, dtype):
    quantize_node = next(node for node in onnx_model.graph.node if node.name == "weight_quantize")
    quantize_node.input[0] = "qdq_weight_cast"
    onnx_model.graph.node.insert(
        0,
        helper.make_node(
            "Cast",
            ["qdq_weight"],
            ["qdq_weight_cast"],
            name="weight_cast",
            to=dtype,
        ),
    )


def _make_quantized_backbone(linear_count, conv_count):
    backbone = torch.nn.Module()
    for index in range(linear_count):
        linear = torch.nn.Linear(16, 16, bias=False)
        linear.input_quantizer = _Quantizer((2, 1))
        linear.weight_quantizer = _Quantizer((2, 1))
        backbone.add_module(f"linear_{index}", linear)
    for index in range(conv_count):
        conv = torch.nn.Conv2d(1, 1, 1, bias=False)
        conv.input_quantizer = _Quantizer((4, 3))
        conv.weight_quantizer = _Quantizer((4, 3))
        backbone.add_module(f"conv_{index}", conv)
    return backbone


def _make_external_data_model(fill_value):
    weight = numpy_helper.from_array(
        np.full((1024,), fill_value, dtype=np.float32), "external_weight"
    )
    graph = helper.make_graph(
        [helper.make_node("Identity", ["external_weight"], ["output"])],
        "external_data",
        [],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1024])],
        [weight],
    )
    return helper.make_model(graph)


def test_fp8_scale_workaround_can_target_only_enabled_conv_quantizers():
    conv = torch.nn.Conv2d(1, 1, 1)
    conv.input_quantizer = _Quantizer((4, 3))
    conv.weight_quantizer = _Quantizer((4, 3), enabled=False)
    linear = torch.nn.Linear(1, 1)
    linear.input_quantizer = _Quantizer((2, 1))
    linear.weight_quantizer = _Quantizer((4, 3))
    model = torch.nn.Sequential(conv, linear)

    diffusion_export.generate_fp8_scales(model, conv_only=True)

    assert conv.input_quantizer.num_bits == 8
    assert conv.input_quantizer._amax == 127.0
    assert conv.weight_quantizer.num_bits == (4, 3)
    assert linear.input_quantizer.num_bits == (2, 1)
    assert linear.weight_quantizer.num_bits == (4, 3)

    diffusion_export.generate_fp8_scales(model)

    assert linear.weight_quantizer.num_bits == 8
    assert linear.weight_quantizer._amax == 127.0


def test_mixed_sdxl_graph_preserves_fp8_conv_and_lowers_exact_nvfp4_topology():
    raw_model = _make_raw_fp4_model()

    converted_model = diffusion_export._process_fp4_onnx_graph(raw_model, "sdxl-1.0")

    assert not any(node.op_type == "TRT_FP4QDQ" for node in converted_model.graph.node)
    assert (
        sum(
            initializer.data_type == TensorProto.FLOAT4E2M1
            for initializer in converted_model.graph.initializer
        )
        == 1
    )
    assert _tensor_dtype(converted_model, "qdq_zero") == TensorProto.FLOAT8E4M3FN
    assert any(
        node.op_type == "Conv" and node.name == "fp8_conv" for node in converted_model.graph.node
    )
    assert next(opset.version for opset in converted_model.opset_import if not opset.domain) >= 23
    onnx.checker.check_model(converted_model)


@pytest.mark.parametrize(
    ("linear_count", "conv_count", "error"),
    [
        (2, 1, "found 1 TRT_FP4QDQ weight markers, expected 2 enabled Linear pairs"),
        (
            1,
            2,
            "found 1 initializer-backed FP8 Conv weight Q/DQ pairs, expected 2 enabled Conv2d pairs",
        ),
    ],
)
def test_raw_sdxl_graph_counts_match_enabled_quantizer_pairs(linear_count, conv_count, error):
    expected_linear_count, expected_conv_count = diffusion_export._get_sdxl_fp4_expected_counts(
        _make_quantized_backbone(linear_count, conv_count)
    )

    with pytest.raises(ValueError, match=error):
        diffusion_export._process_fp4_onnx_graph(
            _make_raw_fp4_model(),
            "sdxl-1.0",
            expected_linear_count=expected_linear_count,
            expected_fp8_conv_count=expected_conv_count,
        )


def test_final_sdxl_graph_count_matches_enabled_conv_pairs():
    converted_model = diffusion_export._process_fp4_onnx_graph(_make_raw_fp4_model(), "sdxl-1.0")
    _, expected_conv_count = diffusion_export._get_sdxl_fp4_expected_counts(
        _make_quantized_backbone(1, 2)
    )

    with pytest.raises(
        ValueError,
        match="found 1 initializer-backed FP8 Conv weight Q/DQ pairs, expected 2 enabled Conv2d pairs",
    ):
        diffusion_export._validate_final_fp4_graph(
            converted_model,
            expected_weight_count=1,
            allow_fp8_conv=True,
            expected_fp8_conv_count=expected_conv_count,
        )


@pytest.mark.parametrize("pre_q_passthrough", [False, True])
def test_non_sdxl_fp4_graph_rejects_static_qdq_conv_weights(pre_q_passthrough):
    model = _make_raw_fp4_model()
    if pre_q_passthrough:
        _insert_passthrough_before_weight_quantize(model)

    with pytest.raises(ValueError, match="disallowed initializer-backed Q/DQ"):
        diffusion_export._process_fp4_onnx_graph(model, "flux-dev")


def test_sdxl_fp4_graph_requires_static_fp8_conv_weight_qdq():
    model = _make_raw_fp4_model()
    quantize_node = next(node for node in model.graph.node if node.name == "weight_quantize")
    quantize_node.input[0] = "qdq_activation"

    with pytest.raises(ValueError, match="no initializer-backed FP8 Conv weight Q/DQ"):
        diffusion_export._process_fp4_onnx_graph(model, "sdxl-1.0")


@pytest.mark.parametrize(
    ("marker_initializer", "marker_block_size", "error"),
    [
        (False, 16, "not backed by a weight initializer"),
        (True, 32, "block_size=32, expected 16"),
    ],
)
def test_raw_fp4_validation_rejects_invalid_markers(marker_initializer, marker_block_size, error):
    model = _make_raw_fp4_model(
        marker_initializer=marker_initializer, marker_block_size=marker_block_size
    )

    with pytest.raises(ValueError, match=error):
        diffusion_export._validate_raw_fp4_graph(model)


@pytest.mark.parametrize("consumer_op", ["Add", "Gemm", "MatMul"])
def test_raw_fp4_validation_rejects_static_qdq_non_conv_weights(consumer_op):
    model = _make_raw_fp4_model(qdq_consumer=consumer_op)

    with pytest.raises(ValueError, match="disallowed initializer-backed Q/DQ"):
        diffusion_export._validate_raw_fp4_graph(model, allow_fp8_conv=True)


@pytest.mark.parametrize("consumer_op", ["Add", None])
def test_raw_fp4_validation_rejects_marker_without_weight_consumer(consumer_op):
    model = _make_raw_fp4_model()
    matmul = next(node for node in model.graph.node if node.name == "fp4_matmul")
    if consumer_op is None:
        model.graph.node.remove(matmul)
    else:
        matmul.op_type = consumer_op

    with pytest.raises(ValueError, match="does not reach a Gemm/MatMul weight input"):
        diffusion_export._validate_raw_fp4_graph(model, allow_fp8_conv=True)


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("block-scale-dtype", "FLOAT8E4M3FN block-scale initializer"),
        ("axis", "does not use axis=-1"),
        ("fp8-zero-dtype", "zero point is not FLOAT8E4M3FN"),
        ("weight-consumer", "does not reach a Gemm/MatMul weight input"),
    ],
)
def test_final_fp4_validation_rejects_invalid_double_dq(corruption, error):
    raw_model = _make_raw_fp4_model()
    expected_weight_count = diffusion_export._validate_raw_fp4_graph(raw_model, allow_fp8_conv=True)
    normalized_model = diffusion_export._normalize_fp8_qdq(raw_model)
    converted_model = NVFP4QuantExporter.process_model(normalized_model)
    fp4_weight = next(
        initializer.name
        for initializer in converted_model.graph.initializer
        if initializer.data_type == TensorProto.FLOAT4E2M1
    )
    weight_dq = next(
        node
        for node in converted_model.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == fp4_weight
    )
    scale_dq = next(
        node for node in converted_model.graph.node if weight_dq.input[1] in node.output
    )
    if corruption == "block-scale-dtype":
        fp8_scale = next(
            initializer
            for initializer in converted_model.graph.initializer
            if initializer.name == scale_dq.input[0]
        )
        fp8_scale.data_type = TensorProto.FLOAT16
    elif corruption == "axis":
        axis = next(attribute for attribute in weight_dq.attribute if attribute.name == "axis")
        axis.i = 0
    elif corruption == "fp8-zero-dtype":
        zero_node = next(node for node in converted_model.graph.node if node.name == "qdq_zero")
        value = next(attribute for attribute in zero_node.attribute if attribute.name == "value")
        value.t.data_type = TensorProto.INT8
    else:
        matmul = next(node for node in converted_model.graph.node if node.name == "fp4_matmul")
        matmul.op_type = "Add"

    with pytest.raises(ValueError, match=error):
        diffusion_export._validate_final_fp4_graph(
            converted_model, expected_weight_count, allow_fp8_conv=True
        )


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("extra-input", "must be a two-input DequantizeLinear"),
        ("axis", "must not use axis or block_size"),
        ("block-size", "must not use axis or block_size"),
        ("fp8-scale-fanout", "must be consumed only by"),
        ("global-scale-dtype", "does not use a FLOAT global-scale initializer"),
        ("global-scale-zero", "global scale must be a finite positive scalar constant"),
        ("scale-output-fanout", "output must be consumed only by"),
    ],
)
def test_final_fp4_validation_rejects_invalid_weight_scale_dq(corruption, error):
    converted_model = diffusion_export._process_fp4_onnx_graph(_make_raw_fp4_model(), "sdxl-1.0")
    fp4_weight = next(
        initializer.name
        for initializer in converted_model.graph.initializer
        if initializer.data_type == TensorProto.FLOAT4E2M1
    )
    weight_dq = next(
        node
        for node in converted_model.graph.node
        if node.op_type == "DequantizeLinear" and node.input[0] == fp4_weight
    )
    scale_dq = next(
        node for node in converted_model.graph.node if weight_dq.input[1] in node.output
    )
    if corruption == "extra-input":
        scale_dq.input.append(scale_dq.input[1])
    elif corruption in {"axis", "block-size"}:
        scale_dq.attribute.append(helper.make_attribute(corruption.replace("-", "_"), 0))
    elif corruption == "fp8-scale-fanout":
        converted_model.graph.node.append(
            helper.make_node(
                "Identity",
                [scale_dq.input[0]],
                ["extra_block_scale_use"],
                name="extra_block_scale_use",
            )
        )
    elif corruption == "global-scale-dtype":
        global_scale = next(
            initializer
            for initializer in converted_model.graph.initializer
            if initializer.name == scale_dq.input[1]
        )
        global_scale.data_type = TensorProto.FLOAT16
    elif corruption == "global-scale-zero":
        global_scale = next(
            initializer
            for initializer in converted_model.graph.initializer
            if initializer.name == scale_dq.input[1]
        )
        global_scale.CopyFrom(
            numpy_helper.from_array(np.array(0.0, dtype=np.float32), global_scale.name)
        )
    else:
        converted_model.graph.node.append(
            helper.make_node(
                "Identity",
                [scale_dq.output[0]],
                ["extra_weight_scale_use"],
                name="extra_weight_scale_use",
            )
        )

    with pytest.raises(ValueError, match=error):
        diffusion_export._validate_final_fp4_graph(
            converted_model, expected_weight_count=1, allow_fp8_conv=True
        )


def test_final_fp4_validation_rejects_extra_float4_weight_consumer():
    raw_model = _make_raw_fp4_model()
    expected_weight_count = diffusion_export._validate_raw_fp4_graph(raw_model, allow_fp8_conv=True)
    converted_model = diffusion_export._process_fp4_onnx_graph(raw_model, "sdxl-1.0")
    fp4_weight = next(
        initializer.name
        for initializer in converted_model.graph.initializer
        if initializer.data_type == TensorProto.FLOAT4E2M1
    )
    converted_model.graph.node.append(
        helper.make_node("Identity", [fp4_weight], ["extra_weight_use"], name="extra_weight_use")
    )

    with pytest.raises(ValueError, match="must feed exactly one weight DequantizeLinear"):
        diffusion_export._validate_final_fp4_graph(
            converted_model, expected_weight_count, allow_fp8_conv=True
        )


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("mismatched-scale", "do not share scale and zero point"),
        ("axis", "without an axis"),
        ("nonpositive-scale", "finite positive scalar constant"),
        ("wrong-scale-dtype", "FP8 scale must use a floating-point dtype"),
        ("nonzero-zero", "zero point must be a scalar zero"),
        ("missing-activation", "has no FP8 activation Q/DQ"),
        ("activation-fanout", "has non-Conv activation consumers"),
        ("weight-fanout", "weight DQ must feed exactly one FP8 Conv input 1"),
    ],
)
def test_final_fp4_validation_rejects_invalid_fp8_conv_qdq(corruption, error):
    converted_model = diffusion_export._process_fp4_onnx_graph(_make_raw_fp4_model(), "sdxl-1.0")
    weight_quantize = next(
        node for node in converted_model.graph.node if node.name == "weight_quantize"
    )
    weight_dequantize = next(
        node for node in converted_model.graph.node if node.name == "weight_dequantize"
    )
    if corruption == "mismatched-scale":
        converted_model.graph.initializer.append(
            numpy_helper.from_array(np.array(0.5, dtype=np.float16), "other_fp8_scale")
        )
        weight_dequantize.input[1] = "other_fp8_scale"
    elif corruption == "axis":
        weight_quantize.attribute.append(helper.make_attribute("axis", 0))
    elif corruption == "nonpositive-scale":
        scale_node = next(node for node in converted_model.graph.node if node.name == "qdq_scale")
        scale_value = next(
            attribute for attribute in scale_node.attribute if attribute.name == "value"
        )
        scale_value.t.CopyFrom(
            numpy_helper.from_array(np.array(0.0, dtype=np.float16), "qdq_scale_value")
        )
    elif corruption == "wrong-scale-dtype":
        scale_node = next(node for node in converted_model.graph.node if node.name == "qdq_scale")
        scale_value = next(
            attribute for attribute in scale_node.attribute if attribute.name == "value"
        )
        scale_value.t.CopyFrom(
            numpy_helper.from_array(np.array(1, dtype=np.int32), "qdq_scale_value")
        )
    elif corruption == "nonzero-zero":
        zero_node = next(node for node in converted_model.graph.node if node.name == "qdq_zero")
        zero_value = next(
            attribute for attribute in zero_node.attribute if attribute.name == "value"
        )
        zero_value.t.raw_data = b"\x38"
    elif corruption == "missing-activation":
        conv = next(node for node in converted_model.graph.node if node.name == "fp8_conv")
        conv.input[0] = "qdq_activation"
    elif corruption == "activation-fanout":
        converted_model.graph.node.append(
            helper.make_node(
                "Identity",
                ["qdq_activation_dq"],
                ["extra_fp8_activation_use"],
                name="extra_fp8_activation_use",
            )
        )
    else:
        converted_model.graph.node.append(
            helper.make_node(
                "Conv",
                ["qdq_activation_dq", "qdq_weight_dq"],
                ["extra_fp8_conv_output"],
                name="extra_fp8_conv",
            )
        )

    with pytest.raises(ValueError, match=error):
        diffusion_export._validate_final_fp4_graph(
            converted_model, expected_weight_count=1, allow_fp8_conv=True
        )


@pytest.mark.parametrize(
    ("corruption", "error"),
    [
        ("block-size", "activation_quantize does not use block_size=16"),
        ("mismatched-scale", "quantize and dequantize global scales do not match"),
        ("wrong-domain", "must use the trt domain"),
        ("static-input", "input 0 must be a dynamic activation"),
        ("fanout", "must feed exactly one Gemm/MatMul activation input"),
        ("bypass", "dynamic NVFP4 activation paths do not match FLOAT4 weight consumers"),
        ("extra-scale-consumer", "FP8 scale output must feed exactly one DequantizeLinear"),
    ],
)
def test_final_fp4_validation_rejects_invalid_dynamic_activation(corruption, error):
    converted_model = diffusion_export._process_fp4_onnx_graph(_make_raw_fp4_model(), "sdxl-1.0")
    dynamic_quantize = next(
        node for node in converted_model.graph.node if node.name == "activation_quantize"
    )
    if corruption == "block-size":
        block_size = next(
            attribute for attribute in dynamic_quantize.attribute if attribute.name == "block_size"
        )
        block_size.i = 32
    elif corruption == "mismatched-scale":
        scale_node = next(
            node for node in converted_model.graph.node if node.name == "dynamic_dequantize_scale"
        )
        scale_value = next(
            attribute for attribute in scale_node.attribute if attribute.name == "value"
        )
        scale_value.t.CopyFrom(
            numpy_helper.from_array(np.array(2.0 / 2688.0, dtype=np.float32), "scale_value")
        )
    elif corruption == "wrong-domain":
        dynamic_quantize.domain = "other"
    elif corruption == "static-input":
        dynamic_quantize.input[0] = "dynamic_quantize_scale"
    elif corruption == "fanout":
        converted_model.graph.node.append(
            helper.make_node(
                "MatMul",
                ["activation_dq_fp16", "fp4_weight_dq"],
                ["extra_linear_output"],
                name="extra_fp4_matmul",
            )
        )
    elif corruption == "bypass":
        matmul = next(node for node in converted_model.graph.node if node.name == "fp4_matmul")
        matmul.input[0] = "activation"
    else:
        converted_model.graph.node.append(
            helper.make_node(
                "Identity",
                [dynamic_quantize.output[1]],
                ["extra_dynamic_scale_use"],
                name="extra_dynamic_scale_use",
            )
        )

    with pytest.raises(ValueError, match=error):
        diffusion_export._validate_final_fp4_graph(
            converted_model, expected_weight_count=1, allow_fp8_conv=True
        )


def test_final_fp4_validation_accepts_bfloat16_fp8_conv_scales():
    converted_model = diffusion_export._process_fp4_onnx_graph(_make_raw_fp4_model(), "sdxl-1.0")
    qdq_weight = next(
        initializer
        for initializer in converted_model.graph.initializer
        if initializer.name == "qdq_weight"
    )
    qdq_weight.data_type = TensorProto.BFLOAT16
    qdq_activation = next(
        value for value in converted_model.graph.input if value.name == "qdq_activation"
    )
    qdq_activation.type.tensor_type.elem_type = TensorProto.BFLOAT16
    scale_node = next(node for node in converted_model.graph.node if node.name == "qdq_scale")
    scale_value = next(attribute for attribute in scale_node.attribute if attribute.name == "value")
    scale_value.t.data_type = TensorProto.BFLOAT16

    diffusion_export._validate_final_fp4_graph(
        converted_model, expected_weight_count=1, allow_fp8_conv=True
    )


def test_final_fp4_validation_uses_cast_output_dtype_for_fp8_conv_scale():
    raw_model = _make_raw_fp4_model()
    _insert_weight_cast_before_quantize(raw_model, TensorProto.BFLOAT16)
    activation = next(value for value in raw_model.graph.input if value.name == "qdq_activation")
    activation.type.tensor_type.elem_type = TensorProto.BFLOAT16
    scale_node = next(node for node in raw_model.graph.node if node.name == "qdq_scale")
    scale_value = next(attribute for attribute in scale_node.attribute if attribute.name == "value")
    scale_value.t.data_type = TensorProto.BFLOAT16

    converted_model = diffusion_export._process_fp4_onnx_graph(raw_model, "sdxl-1.0")

    diffusion_export._validate_final_fp4_graph(
        converted_model, expected_weight_count=1, allow_fp8_conv=True
    )


def test_non_sdxl_fp4_export_uses_permissive_generic_lowering(monkeypatch, tmp_path):
    raw_model = _make_raw_fp4_model()
    generic_calls = []
    saved_models = []

    def fake_onnx_export(*args, f, **kwargs):
        del args, kwargs
        onnx.save(raw_model, f)

    def generic_process(cls, onnx_model):
        del cls
        generic_calls.append(onnx_model)
        return onnx_model

    monkeypatch.setattr(diffusion_export, "onnx_export", fake_onnx_export)
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, {}),
    )
    monkeypatch.setattr(
        diffusion_export,
        "_process_fp4_onnx_graph",
        lambda *args: pytest.fail("strict SDXL processing must not run for Flux"),
    )
    monkeypatch.setattr(NVFP4QuantExporter, "process_model", classmethod(generic_process))
    monkeypatch.setattr(
        diffusion_export,
        "save_onnx",
        lambda onnx_model, output: saved_models.append((onnx_model, output)),
    )
    monkeypatch.setattr(
        diffusion_export,
        "_save_onnx_atomically",
        lambda *args: pytest.fail("atomic checked publication must be SDXL FP4-only"),
    )

    diffusion_export.modelopt_export_sd(torch.nn.Identity(), tmp_path, "flux-dev", "fp4")

    assert len(generic_calls) == 1
    assert len(saved_models) == 1
    assert (
        next(opset.version for opset in saved_models[0][0].opset_import if not opset.domain) == 20
    )


def test_sdxl_fp4_export_threads_enabled_quantizer_counts(monkeypatch, tmp_path):
    raw_model = _make_raw_fp4_model()
    processed_counts = []
    saved_models = []
    backbone = _make_quantized_backbone(2, 3)

    def fake_onnx_export(*args, f, **kwargs):
        del args, kwargs
        onnx.save(raw_model, f)

    def fake_process(onnx_model, model_name, block_size, **kwargs):
        del model_name, block_size
        processed_counts.append(kwargs)
        return onnx_model

    monkeypatch.setattr(diffusion_export, "onnx_export", fake_onnx_export)
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, {}),
    )
    monkeypatch.setattr(
        diffusion_export, "configure_linear_module_onnx_quantizers", lambda _: nullcontext()
    )
    monkeypatch.setattr(diffusion_export, "_process_fp4_onnx_graph", fake_process)
    monkeypatch.setattr(
        diffusion_export,
        "_save_onnx_atomically",
        lambda onnx_model, output: saved_models.append((onnx_model, output)),
    )
    monkeypatch.setattr(
        diffusion_export,
        "save_onnx",
        lambda *args: pytest.fail("SDXL FP4 must use atomic checked publication"),
    )

    diffusion_export.modelopt_export_sd(backbone, tmp_path, "sdxl-1.0", "fp4")

    assert processed_counts == [{"expected_linear_count": 2, "expected_fp8_conv_count": 3}]
    assert len(saved_models) == 1


@pytest.mark.parametrize("precision", ["fp16", "fp8"])
def test_non_fp4_exports_keep_legacy_save_path(monkeypatch, tmp_path, precision):
    raw_model = _make_raw_fp4_model()
    saved_models = []

    def fake_onnx_export(*args, f, **kwargs):
        del args, kwargs
        onnx.save(raw_model, f)

    monkeypatch.setattr(diffusion_export, "onnx_export", fake_onnx_export)
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, {}),
    )
    monkeypatch.setattr(diffusion_export, "_normalize_fp8_qdq", lambda model: model)
    monkeypatch.setattr(
        diffusion_export,
        "save_onnx",
        lambda onnx_model, output: saved_models.append((onnx_model, output)),
    )
    monkeypatch.setattr(
        diffusion_export,
        "_save_onnx_atomically",
        lambda *args: pytest.fail("atomic checked publication must be SDXL FP4-only"),
    )

    diffusion_export.modelopt_export_sd(torch.nn.Identity(), tmp_path, "sdxl-1.0", precision)

    assert len(saved_models) == 1
    assert (
        next(opset.version for opset in saved_models[0][0].opset_import if not opset.domain) == 20
    )


def test_invalid_fp4_export_preserves_existing_output_and_cleans_raw_temp(monkeypatch, tmp_path):
    output_dir = tmp_path / "onnx"
    output_dir.mkdir()
    output = output_dir / "model.onnx"
    output_data = output_dir / "model.onnx_data"
    output.write_bytes(b"previous-model")
    output_data.write_bytes(b"previous-data")
    raw_dirs = []

    invalid_model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Identity", ["input"], ["output"])],
            "invalid_fp4",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        )
    )

    def fake_onnx_export(*args, f, **kwargs):
        del args, kwargs
        raw_dirs.append(Path(f).parent)
        onnx.save(invalid_model, f)

    monkeypatch.setattr(diffusion_export, "onnx_export", fake_onnx_export)
    monkeypatch.setattr(
        diffusion_export,
        "generate_dummy_kwargs_and_dynamic_axes_and_shapes",
        lambda *args: ({}, {}, {}),
    )

    with pytest.raises(ValueError, match="no TRT_FP4QDQ weight markers"):
        diffusion_export.modelopt_export_sd(torch.nn.Identity(), output_dir, "sdxl-1.0", "fp4")

    assert output.read_bytes() == b"previous-model"
    assert output_data.read_bytes() == b"previous-data"
    assert len(raw_dirs) == 1
    assert not raw_dirs[0].exists()


def test_staged_save_failure_preserves_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "model.onnx"
    output_data = tmp_path / "model.onnx_data"
    output.write_bytes(b"previous-model")
    output_data.write_bytes(b"previous-data")

    def fail_after_partial_save(onnx_model, staged_output, external_data_name=None):
        del onnx_model
        staged_output.write_bytes(b"partial-model")
        (staged_output.parent / external_data_name).write_bytes(b"partial-data")
        raise RuntimeError("save failed")

    monkeypatch.setattr(diffusion_export, "save_onnx", fail_after_partial_save)

    with pytest.raises(RuntimeError, match="save failed"):
        diffusion_export._save_onnx_atomically(object(), output)

    assert output.read_bytes() == b"previous-model"
    assert output_data.read_bytes() == b"previous-data"
    assert not list(tmp_path.glob(".modelopt-export-*"))


def test_staged_checker_failure_preserves_existing_output(monkeypatch, tmp_path):
    output = tmp_path / "model.onnx"
    output_data = tmp_path / "model.onnx_data"
    output.write_bytes(b"previous-model")
    output_data.write_bytes(b"previous-data")
    checked_paths = []

    def fail_check(staged_output):
        checked_paths.append(Path(staged_output))
        raise RuntimeError("checker failed")

    monkeypatch.setattr(diffusion_export.onnx.checker, "check_model", fail_check)

    with pytest.raises(RuntimeError, match="checker failed"):
        diffusion_export._save_onnx_atomically(_make_external_data_model(2.0), output)

    assert len(checked_paths) == 1
    assert checked_paths[0].parent.name.startswith(".modelopt-export-")
    assert output.read_bytes() == b"previous-model"
    assert output_data.read_bytes() == b"previous-data"
    assert not list(tmp_path.glob("model.onnx_data.*"))
    assert not list(tmp_path.glob(".modelopt-export-*"))


def test_publication_failure_restores_previous_output_and_removes_new_data(monkeypatch, tmp_path):
    output = tmp_path / "model.onnx"
    output_data = tmp_path / "model.onnx_data"
    output.write_bytes(b"previous-model")
    output_data.write_bytes(b"previous-data")

    real_replace = diffusion_export.os.replace

    def fail_model_publish(source, destination):
        source = Path(source)
        destination = Path(destination)
        if source.name == output.name and source.parent.name.startswith(".modelopt-export-"):
            raise RuntimeError("publish failed")
        real_replace(source, destination)

    monkeypatch.setattr(diffusion_export.os, "replace", fail_model_publish)

    with pytest.raises(RuntimeError, match="publish failed"):
        diffusion_export._save_onnx_atomically(_make_external_data_model(2.0), output)

    assert output.read_bytes() == b"previous-model"
    assert output_data.read_bytes() == b"previous-data"
    assert not list(tmp_path.glob("model.onnx_data.*"))
    assert not list(tmp_path.glob(".modelopt-export-*"))


def test_atomic_save_publishes_versioned_data_then_removes_old_data(tmp_path):
    output = tmp_path / "model.onnx"
    old_data = tmp_path / "model.onnx_data"
    diffusion_export.save_onnx(_make_external_data_model(1.0), output)
    assert old_data.exists()

    diffusion_export._save_onnx_atomically(_make_external_data_model(2.0), output)

    stored_model = onnx.load(str(output), load_external_data=False)
    locations = {
        entry.value
        for initializer in stored_model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    }
    assert len(locations) == 1
    external_data_name = locations.pop()
    assert external_data_name.startswith("model.onnx_data.")
    assert (tmp_path / external_data_name).exists()
    assert not old_data.exists()
    assert np.all(numpy_helper.to_array(onnx.load(str(output)).graph.initializer[0]) == 2.0)
    assert not list(tmp_path.glob(".modelopt-export-*"))
