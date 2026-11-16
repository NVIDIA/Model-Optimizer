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

import struct

import numpy as np
import onnx
import onnxruntime as ort
import pytest

import modelopt.onnx.quantization as moq


@pytest.mark.parametrize("use_calibration_cache", [False, True])
@pytest.mark.parametrize("high_precision_dtype", ["fp32", "fp16"])
@pytest.mark.parametrize("target_dla", [False, True])
@pytest.mark.parametrize("colliding_parameter", ["scale", "zero_point"])
@pytest.mark.parametrize("collision_shape", ["scalar", "vector"])
def test_quantization_preserves_existing_parameter_initializers(
    tmp_path,
    high_precision_dtype,
    target_dla,
    colliding_parameter,
    collision_shape,
    use_calibration_cache,
):
    """Keep quantization numerically equivalent when parameter names collide."""
    data = np.linspace(-2, 2, 64, dtype=np.float32).reshape(1, 4, 4, 4)
    cache = None
    if use_calibration_cache:
        cache = tmp_path / "calibration.cache"
        names = ["input", "normalized", "left", "right", "combined", "output"]
        encoded_scale = struct.pack("!f", 0.125).hex()
        cache.write_text(
            "TRT-8501-EntropyCalibration2\n"
            + "".join(f"{name}: {encoded_scale}\n" for name in names)
        )
    results = []
    for collide in (False, True):
        names = ["gamma", "normalized_scale_1", "normalized_zero_point_2", "variance", "offset"]
        if collide:
            names[0 if collision_shape == "vector" else 4] = f"normalized_{colliding_parameter}"
        values = [
            np.array([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.zeros(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.array(3.0, dtype=np.float32),
        ]
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "BatchNormalization",
                        ["input", *names[:4]],
                        ["normalized"],
                        name="normalization",
                    ),
                    onnx.helper.make_node(
                        "LeakyRelu", ["normalized"], ["left"], name="left_activation"
                    ),
                    onnx.helper.make_node(
                        "Relu", ["normalized"], ["right"], name="right_activation"
                    ),
                    onnx.helper.make_node("Add", ["left", "right"], ["combined"], name="sum"),
                    onnx.helper.make_node("Add", ["combined", names[4]], ["output"], name="offset"),
                ],
                "normalization",
                [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 4, 4, 4])],
                [
                    onnx.helper.make_tensor_value_info(
                        "output", onnx.TensorProto.FLOAT, [1, 4, 4, 4]
                    )
                ],
                [onnx.numpy_helper.from_array(value, name) for name, value in zip(names, values)],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 19)],
            ir_version=10,
        )
        source = tmp_path / f"source-{collide}.onnx"
        output = tmp_path / f"quantized-{collide}.onnx"
        onnx.save(model, source)
        moq.quantize(
            str(source),
            output_path=str(output),
            quantize_mode="int8",
            high_precision_dtype=high_precision_dtype,
            calibration_data={"input": data},
            calibration_eps=["cpu"],
            calibration_cache_path=str(cache) if cache else None,
            target_dla=target_dla,
            op_types_to_quantize=["BatchNormalization", "LeakyRelu", "Relu", "Add"],
        )
        quantized = onnx.load(output)
        onnx.checker.check_model(quantized)
        initializers = {tensor.name: tensor for tensor in quantized.graph.initializer}
        qdq_nodes = [
            node
            for node in quantized.graph.node
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        ]
        assert qdq_nodes
        if use_calibration_cache:
            quantizer = next(node for node in qdq_nodes if node.name == "normalized_QuantizeLinear")
            assert onnx.numpy_helper.to_array(initializers[quantizer.input[1]]) == 0.125
        for node in qdq_nodes:
            scale, zero = (initializers[name] for name in node.input[1:3])
            assert list(scale.dims) == list(zero.dims), node.name
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(output), sess_options=options, providers=["CPUExecutionProvider"]
        )
        results.append(session.run(None, {"input": data})[0])
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("output_name", ["output", "output_QuantizeLinear_Input"])
def test_calibration_cache_preserves_graph_outputs_with_parameter_collisions(tmp_path, output_name):
    """Restore cached output scales without overwriting renamed per-channel weight scales."""
    data = np.linspace(-1, 1, 8, dtype=np.float32).reshape(2, 4)
    weight = np.eye(4, dtype=np.float32)
    cache = tmp_path / "calibration.cache"
    names = ["input", "product", "shifted", output_name]
    encoded_scale = struct.pack("!f", 0.125).hex()
    results = []
    for collide in (False, True):
        cache_names = [*names, "weight"] if collide else names
        cache.write_text(
            "TRT-8501-EntropyCalibration2\n"
            + "".join(f"{name}: {encoded_scale}\n" for name in cache_names)
        )
        offsets = ["weight_scale", f"{output_name}_scale"] if collide else ["offset_1", "offset_2"]
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                [
                    onnx.helper.make_node(
                        "MatMul", ["input", "weight"], ["product"], name="matmul"
                    ),
                    onnx.helper.make_node(
                        "Add", ["product", offsets[0]], ["shifted"], name="shift"
                    ),
                    onnx.helper.make_node(
                        "Add", ["shifted", offsets[1]], [output_name], name="sum"
                    ),
                ],
                "cached_outputs",
                [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 4])],
                [onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [2, 4])],
                [
                    onnx.numpy_helper.from_array(weight, "weight"),
                    onnx.numpy_helper.from_array(np.array(0.25, np.float32), offsets[0]),
                    onnx.numpy_helper.from_array(np.array(0.5, np.float32), offsets[1]),
                ],
            ),
            opset_imports=[onnx.helper.make_opsetid("", 19)],
            ir_version=10,
        )
        source, output = tmp_path / f"source-{collide}.onnx", tmp_path / f"quantized-{collide}.onnx"
        onnx.save(model, source)
        moq.quantize(
            str(source),
            output_path=str(output),
            quantize_mode="int8",
            high_precision_dtype="fp32",
            calibration_data={"input": data},
            calibration_cache_path=str(cache),
            calibration_eps=["cpu"],
            op_types_to_quantize=["MatMul", "Add"],
            op_types_needing_output_quant=["Add"],
            enable_gemv_detection_for_trt=False,
        )
        quantized = onnx.load(output)
        onnx.checker.check_model(quantized, full_check=True)
        initializers = {tensor.name: tensor for tensor in quantized.graph.initializer}
        producers = {name: node for node in quantized.graph.node for name in node.output}
        dequantizer = producers[output_name]
        assert dequantizer.op_type == "DequantizeLinear"
        output_quantizer = producers[dequantizer.input[0]]
        assert output_quantizer.input[0] == f"{output_name}_QuantizeLinear_Input"
        assert onnx.numpy_helper.to_array(initializers[output_quantizer.input[1]]) == 0.125
        weight_quantizer = next(
            node
            for node in quantized.graph.node
            if node.op_type == "QuantizeLinear" and node.input[0] == "weight"
        )
        np.testing.assert_array_equal(
            onnx.numpy_helper.to_array(initializers[weight_quantizer.input[1]]),
            np.full(4, 1 / 127, dtype=np.float32),
        )
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(
            str(output), sess_options=options, providers=["CPUExecutionProvider"]
        )
        results.append(session.run(None, {"input": data})[0])
    np.testing.assert_array_equal(results[0], results[1])
