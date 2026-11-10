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

"""Regression coverage for the Windows ONNX GenAI quantization example."""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper

from examples.windows.onnx_ptq.genai_llm.quantize_utils import (
    input_shape_profiles,
    make_embedded_input,
    save_quantized_model,
)


def test_embedded_decoder_feed_matches_graph_and_runs():
    inputs = [
        helper.make_tensor_value_info(
            "attention_mask", TensorProto.INT64, ["batch_size", "total_sequence_length"]
        ),
        helper.make_tensor_value_info(
            "inputs_embeds", TensorProto.FLOAT16, ["batch_size", "sequence_length", 2]
        ),
        helper.make_tensor_value_info(
            "position_ids", TensorProto.INT64, [3, "batch_size", "sequence_length"]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.conv_state", TensorProto.FLOAT16, ["batch_size", 4, 3]
        ),
        helper.make_tensor_value_info(
            "past_key_values.0.recurrent_state", TensorProto.FLOAT16, ["batch_size", 2, 2, 2]
        ),
        helper.make_tensor_value_info(
            "past_key_values.1.key",
            TensorProto.FLOAT16,
            ["batch_size", 1, "past_sequence_length", 2],
        ),
        helper.make_tensor_value_info(
            "past_key_values.1.value",
            TensorProto.FLOAT16,
            ["batch_size", 1, "past_sequence_length", 2],
        ),
    ]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["inputs_embeds"], ["output"])],
        "hybrid_decoder",
        inputs,
        [
            helper.make_tensor_value_info(
                "output", TensorProto.FLOAT16, ["batch_size", "sequence_length", 2]
            )
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)])
    model.ir_version = 10
    onnx.checker.check_model(model)

    embedding_weights = torch.arange(12, dtype=torch.bfloat16).reshape(6, 2)
    token_ids = torch.tensor([[1, 3, 5]])
    attention_mask = torch.tensor([[1, 1, 0]])
    zero_cache = {}
    feed = make_embedded_input(token_ids, attention_mask, inputs, embedding_weights, zero_cache)

    assert set(feed) == {value.name for value in inputs}
    assert feed["inputs_embeds"].dtype == np.float16
    np.testing.assert_array_equal(feed["inputs_embeds"], [[[2, 3], [6, 7], [10, 11]]])
    np.testing.assert_array_equal(feed["position_ids"], np.array([[[0, 1, 0]]] * 3))
    assert feed["past_key_values.0.conv_state"].shape == (1, 4, 3)
    assert feed["past_key_values.0.recurrent_state"].shape == (1, 2, 2, 2)
    assert feed["past_key_values.1.key"].shape == (1, 1, 0, 2)
    result = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    ).run(None, feed)
    np.testing.assert_array_equal(result[0], feed["inputs_embeds"])

    second_feed = make_embedded_input(
        token_ids, attention_mask, inputs, embedding_weights, zero_cache
    )
    assert (
        second_feed["past_key_values.0.recurrent_state"]
        is feed["past_key_values.0.recurrent_state"]
    )

    profiles = input_shape_profiles(["NvTensorRtRtx", "cuda"], feed)
    assert profiles[1] == {}
    for bound in ("min", "opt", "max"):
        shapes = profiles[0][f"nv_profile_{bound}_shapes"]
        assert "inputs_embeds:1x3x2" in shapes
        assert "past_key_values.0.conv_state:1x4x3" in shapes
        assert "past_key_values.1.key:1x1x0x2" in shapes


def test_saved_model_keeps_shape_constants_inline(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["input", "weight"], ["sum"]),
            helper.make_node("ReduceMean", ["sum", "axes"], ["output"], keepdims=0),
        ],
        "external_axes",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1024])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [])],
        initializer=[
            numpy_helper.from_array(np.ones(1024, dtype=np.float32), name="weight"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="axes"),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    output_path = tmp_path / "quantized.onnx"

    save_quantized_model(model, output_path)

    saved = onnx.load(output_path, load_external_data=False)
    initializers = {value.name: value for value in saved.graph.initializer}
    assert initializers["axes"].data_location == onnx.TensorProto.DEFAULT
    assert initializers["weight"].data_location == onnx.TensorProto.EXTERNAL
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    np.testing.assert_array_equal(
        session.run(None, {"input": np.ones(1024, dtype=np.float32)})[0],
        np.array(2, dtype=np.float32),
    )
