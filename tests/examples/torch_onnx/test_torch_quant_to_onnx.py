# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from collections import defaultdict

import onnx
import pytest
from _test_utils.examples.run_command import extend_cmd_parts, run_example_command

# TODO: Add int4_awq once the INT4 exporter supports non-MatMul/Gemm consumer patterns
# (e.g., DQ -> Reshape -> Slice in small ViT / SwinTransformer ONNX graphs).
_QUANT_MODES = ["fp8", "int8", "mxfp8", "nvfp4", "auto"]

_MODELS = {
    "vit_tiny": ("vit_tiny_patch16_224", '{"depth": 1}'),
    "swin_tiny": ("swin_tiny_patch4_window7_224", '{"depths": [1, 1, 1, 1]}'),
    "swinv2_tiny": ("swinv2_tiny_window8_256", '{"depths": [1, 1, 1, 1]}'),
    "resnet50": ("resnet50", None),
}


def _assert_residual_adds_are_quantized(onnx_save_path, quantize_mode):
    model = onnx.load(onnx_save_path)
    initializers = {initializer.name: initializer for initializer in model.graph.initializer}
    consumers = defaultdict(list)
    producers = {}
    for node in model.graph.node:
        for input_name in node.input:
            consumers[input_name].append(node)
        for output_name in node.output:
            producers[output_name] = node

    residual_adds = [
        node
        for node in model.graph.node
        if node.op_type == "Add"
        and [consumer.op_type for consumer in consumers[node.output[0]]] == ["Relu"]
    ]
    assert len(residual_adds) == 16
    for add in residual_adds:
        input_producers = [producers[input_name] for input_name in add.input]
        input_producers = [
            producers[node.input[0]] if node.op_type == "Cast" else node for node in input_producers
        ]
        assert any(
            node.op_type.endswith("DequantizeLinear")
            and producers[node.input[0]].op_type.endswith("QuantizeLinear")
            for node in input_producers
        )

    if quantize_mode not in ("int8", "fp8"):
        return

    activation_quantizers = [
        node
        for node in model.graph.node
        if node.op_type.endswith("QuantizeLinear") and node.input[0] not in initializers
    ]
    assert len(activation_quantizers) == (54 if quantize_mode == "int8" else 52)
    assert len({node.input[0] for node in activation_quantizers}) == len(activation_quantizers)
    assert all(
        producers.get(node.input[0]) is None or producers[node.input[0]].op_type != "Cast"
        for node in activation_quantizers
    )

    dq_fanouts = [
        sorted(consumer.op_type for consumer in consumers[node.output[0]])
        for node in model.graph.node
        if node.op_type.endswith("DequantizeLinear")
    ]
    assert dq_fanouts.count(["Add", "Conv"]) == 12
    assert dq_fanouts.count(["Conv", "Conv"]) == 4
    assert dq_fanouts.count(["Add"]) == 4

    gemm = next(node for node in model.graph.node if node.op_type == "Gemm")
    assert all(
        producers.get(input_name) is None
        or not producers[input_name].op_type.endswith("DequantizeLinear")
        for input_name in gemm.input
    )

    global_pool = next(node for node in model.graph.node if node.op_type == "GlobalAveragePool")
    pool_input_producer = producers[global_pool.input[0]]
    if quantize_mode == "int8":
        assert pool_input_producer.op_type.endswith("DequantizeLinear")
        weight_quantizers = [
            node
            for node in model.graph.node
            if node.op_type.endswith("QuantizeLinear") and node.input[0] in initializers
        ]
        assert len(weight_quantizers) == 53
        assert all(
            initializers[node.input[0]].data_type == onnx.TensorProto.FLOAT16
            for node in weight_quantizers
        )
    else:
        assert pool_input_producer.op_type == "Relu"
        first_conv = next(node for node in model.graph.node if node.op_type == "Conv")
        assert all(
            producers.get(input_name) is None
            or not producers[input_name].op_type.endswith("DequantizeLinear")
            for input_name in first_conv.input
        )


@pytest.mark.parametrize("quantize_mode", _QUANT_MODES)
@pytest.mark.parametrize("model_key", list(_MODELS))
def test_torch_onnx(tmp_path, model_key, quantize_mode):
    timm_model_name, model_kwargs = _MODELS[model_key]
    onnx_save_path = tmp_path / f"{model_key}.{quantize_mode}.onnx"

    cmd_parts = extend_cmd_parts(
        ["python", "torch_quant_to_onnx.py"],
        timm_model_name=timm_model_name,
        model_kwargs=model_kwargs,
        quantize_mode=quantize_mode,
        onnx_save_path=onnx_save_path,
        calibration_data_size="1",
        num_score_steps="1",
    )
    cmd_parts.extend(["--no_pretrained", "--trt_build"])
    run_example_command(cmd_parts, "torch_onnx")

    if model_key == "resnet50":
        _assert_residual_adds_are_quantized(onnx_save_path, quantize_mode)
