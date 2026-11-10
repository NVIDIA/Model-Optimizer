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

"""Helpers for the Windows ONNX GenAI quantization example."""

from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import helper


def make_embedded_input(token_ids, attention_mask, graph_inputs, embedding_weights, zero_cache):
    """Match a text-only prefill feed to the ONNX graph's declared inputs."""
    token_ids = token_ids.cpu()
    attention_mask = attention_mask.cpu().numpy()
    batch_size, sequence_length = token_ids.shape
    graph_inputs = {value.name: value for value in graph_inputs}
    required = {"inputs_embeds", "attention_mask", "position_ids"}
    if not required <= graph_inputs.keys():
        raise ValueError(f"Embedded decoder is missing inputs: {required - graph_inputs.keys()}")

    embedding_type = helper.tensor_dtype_to_np_dtype(
        graph_inputs["inputs_embeds"].type.tensor_type.elem_type
    )
    hidden_size = graph_inputs["inputs_embeds"].type.tensor_type.shape.dim[2].dim_value
    if embedding_weights.ndim != 2 or embedding_weights.shape[1] != hidden_size:
        raise ValueError("Embedding weights do not match the ONNX hidden size")
    embedding_dtype = torch.float16 if embedding_type == np.float16 else torch.float32
    embeddings = embedding_weights[token_ids].to(embedding_dtype)
    inputs = {"inputs_embeds": embeddings.numpy(), "attention_mask": attention_mask}

    positions = np.cumsum(attention_mask, axis=-1, dtype=np.int64) - 1
    positions[attention_mask == 0] = 0
    position_shape = graph_inputs["position_ids"].type.tensor_type.shape.dim
    if len(position_shape) != 3 or not position_shape[0].HasField("dim_value"):
        raise ValueError("Expected a fixed-size three-dimensional position_ids input")
    inputs["position_ids"] = np.broadcast_to(
        positions, (position_shape[0].dim_value, batch_size, sequence_length)
    ).copy()

    dimensions = {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "total_sequence_length": sequence_length,
        "past_sequence_length": 0,
    }
    for name, value in graph_inputs.items():
        if name in inputs:
            continue
        if not name.startswith("past_key_values."):
            raise ValueError(f"No calibration value available for ONNX input {name!r}")
        shape = tuple(
            dim.dim_value if dim.HasField("dim_value") else dimensions[dim.dim_param]
            for dim in value.type.tensor_type.shape.dim
        )
        dtype = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        cache_key = (shape, dtype)
        if cache_key not in zero_cache:
            zero_cache[cache_key] = np.zeros(shape, dtype=dtype)
        inputs[name] = zero_cache[cache_key]

    for name, value in graph_inputs.items():
        expected = helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type)
        if inputs[name].dtype != expected:
            raise ValueError(f"ONNX input {name!r} expects {expected}, got {inputs[name].dtype}")
    return inputs


def input_shape_profiles(calibration_eps, inputs):
    """Use the fixed calibration shapes for TensorRT execution providers."""
    shapes = ",".join(
        f"{name}:{'x'.join(str(size) for size in value.shape)}" for name, value in inputs.items()
    )
    profiles = []
    for ep in calibration_eps:
        prefix = "nv" if ep == "NvTensorRtRtx" else "trt" if ep == "trt" else None
        profiles.append(
            {f"{prefix}_profile_{bound}_shapes": shapes for bound in ("min", "opt", "max")}
            if prefix
            else {}
        )
    return profiles


def save_quantized_model(model, output_path):
    """Keep shape constants inline when saving large ONNX initializers externally."""
    output_path = Path(output_path)
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        location=output_path.name + "_data",
        size_threshold=1024,
    )
