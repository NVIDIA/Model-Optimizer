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
import logging
from pathlib import Path

from ..common import read_string
from .constants import INPUT_DATA_KEY, NODE_NAME_DELIMITER, OUTPUT_DATA_KEY, UNNAMED_LAYER_KEY


def _merge_reformatters(layer_latency_dict: dict[str, float]) -> dict[str, float]:
    reformatter_delimiters = [" input reformatter ", " output reformatter ", " to "]
    keys2merge = [k for k in layer_latency_dict if any(rn in k for rn in reformatter_delimiters)]

    for key in keys2merge:
        layer_latency = layer_latency_dict.pop(key)
        for delimiter in reformatter_delimiters:
            if delimiter in key:
                split_key = key.split(delimiter)
                if len(split_key) != 2:
                    logging.error(
                        f"Key splitting in layerwise profiling failed for reformatter node: {key}."
                    )
                else:
                    key = split_key[1] if delimiter == " to " else split_key[0]
                break

        layer_latency_dict[key] = layer_latency_dict.get(key, 0.0) + layer_latency

    return layer_latency_dict


def process_layerwise_result(
    profile_path: str | Path, onnx_node_names: list[str] | None = None
) -> dict[str, float]:
    """This module process the layerwise profiling result to make them mappable to PyTorch.

    Args:
        profile_path: Path to the layerwise profiling dump from trtexec.
        onnx_node_names: List of node names in the onnx model.

    Returns:
        A dictionary mapping layer names to latency in ms. For example,

            .. code-block:: python

                {
                    "Conv_0 + Clip_3": 0.016,
                    "Conv_4": 0.008,
                    "Gemm_169": 0.015,
                    "input_data": 0.010,
                    "output_data": 0.040,
                }
    """
    layerwise_profiling = read_string(profile_path)
    layerwise_results = json.loads(layerwise_profiling)[1:]

    # Just keep averageMs from layerwise profiling result
    layer_latency_dict = {}
    for results in layerwise_results:
        layer_latency_dict[results["name"].replace("onnx::", "")] = results["averageMs"]

    # Merge the input/output reformatter op into its following op
    layer_latency_dict = _merge_reformatters(layer_latency_dict)

    return map_trt_layers_to_onnx(layer_latency_dict, onnx_node_names)


def map_trt_layers_to_onnx(
    layerwise_result: dict[str, float], onnx_node_names: list[str] | None = None
) -> dict[str, float]:
    """This module maps the TensorRT layers from profiling result with onnx nodes.

    Args:
        layerwise_result: Layerwise profiling result.
        onnx_node_names: Onnx node names.

    Returns:
        A dictionary mapping layer names to latency in ms.
    """

    def _group_split(key: str, delimiter: str):
        parenthesis_balance = 0
        key_group = []
        current_key = ""

        for ch in key:
            if ch == "(":
                parenthesis_balance += 1
            elif ch == ")":
                parenthesis_balance -= 1
            elif ch == delimiter and parenthesis_balance == 0:
                key_group.append(current_key.strip())
                current_key = ""
                continue

            current_key += ch

        # Insert the last key in group
        key_group.append(current_key.strip())

        return key_group

    def _remove_non_onnx_nodes(layer: str) -> tuple[str]:
        keys = []

        # To match the longer name first, match from the last node
        for node_name in reversed(onnx_node_names or []):
            if node_name in layer:
                keys.append(node_name)
                layer = layer.replace(node_name, "")  # Skip from further matching

        cleaned_layer = NODE_NAME_DELIMITER.join(reversed(keys))

        # Lets check if the layer has any input/output data timing
        if not cleaned_layer and "input" in layer:
            return (INPUT_DATA_KEY,)
        if not cleaned_layer and "output" in layer:
            return (OUTPUT_DATA_KEY,)

        if not onnx_node_names:  # empty list if omnimizer<=v0.4.1
            return (layer,)

        return (cleaned_layer,) if len(cleaned_layer) > 0 else ()  # type: ignore[return-value]

    def _iterative_split_key(key: str) -> tuple:
        if ": " in key:
            # e.g., 2-layer MLP: Conv_68 + Relu_69 -> Conv_70
            if len(key.split(": ")) != 2:
                logging.error(f"Key splitting in layerwise profiling failed for key: {key}.")
                return _remove_non_onnx_nodes(key)
            return _iterative_split_key(key.split(": ")[1])
        elif key.startswith("PWN"):
            # e.g., PWN(Clip_589 + (Unnamed Layer* 16) [Shuffle], PWN(PWN(PWN(Add_586 + (Unnamed Layer* 11) [Shuffle]
            # + Add_43, Clip_46), Mul_47), Div_49))
            if not key.endswith(")"):
                logging.error(f"Key: {key} starts with 'PWN' but does not end with a ')'.")
                key = key[4:]
            else:
                key = key[4:-1]
            return sum((_iterative_split_key(k) for k in _group_split(key, ",")), ())
        elif " + " in key:
            # e.g. Add_586 + (Unnamed Layer* 11) [Shuffle] + Add_43
            return sum((_iterative_split_key(k) for k in _group_split(key, "+")), ())
        elif " -> " in key:
            # e.g. Conv_68 + Relu_69 -> Conv_70
            key = key.split(" -> ")
            if len(key) != 2:
                logging.error(f"Key splitting in layerwise profiling failed for key: {key}.")
                return _remove_non_onnx_nodes(key)

            covered_keys = [*_group_split(key[0], "+"), key[1]]
            if onnx_node_names:
                idx1, idx2 = onnx_node_names.index(key[0]), onnx_node_names.index(key[1])
                covered_keys = tuple(onnx_node_names[idx1 : idx2 + 1])
            return sum((_iterative_split_key(k) for k in covered_keys), ())

        return _remove_non_onnx_nodes(key)

    mapped_layerwise_result = {}
    for k, v in layerwise_result.items():
        new_key = NODE_NAME_DELIMITER.join(_iterative_split_key(k)) or UNNAMED_LAYER_KEY
        mapped_layerwise_result[new_key] = mapped_layerwise_result.get(new_key, 0.0) + v

    return mapped_layerwise_result
