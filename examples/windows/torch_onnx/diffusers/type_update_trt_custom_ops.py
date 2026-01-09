# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import os

import onnx
from onnx import TensorProto, helper

dtype_map = {
    "output_0": TensorProto.FLOAT4E2M1,
    "output_1": TensorProto.FLOAT8E4M3FN,
}


def get_dtype(out_name):
    res = None
    if "output_0" in out_name:
        res = dtype_map["output_0"]
    elif "output_1" in out_name:
        res = dtype_map["output_1"]
    assert res is not None, "dtype None, output name mismatch"
    return res


def parse_args():
    parser = argparse.ArgumentParser(description="Update ONNX model with TRT custom ops.")
    parser.add_argument("--input_path", required=True, help="Path to input ONNX model file")
    parser.add_argument("--output_path", required=True, help="Path to save updated ONNX model file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path

    model = onnx.load(input_path)
    graph = model.graph

    # constant_names = set(init.name for init in graph.initializer)

    # Iterate through all nodes to find TRT_FP4DynamicQuantize
    for node in graph.node:
        if node.op_type == "TRT_FP4DynamicQuantize":
            assert len(node.output) == 2, "outputs count mismatch"
            for out_name in node.output:
                # Check if value_info for output already exists
                found = False
                for vi in graph.value_info:
                    if vi.name == out_name:
                        vi.type.tensor_type.elem_type = get_dtype(out_name)
                        # Optionally set shape to empty (unknown), or keep as is
                        # vi.type.tensor_type.shape.Clear()
                        found = True
                        break
                # If not found, create new value_info with only type (no shape)
                if not found:
                    value_info = helper.make_tensor_value_info(
                        out_name, get_dtype(out_name), None
                    )  # None for shape
                    graph.value_info.append(value_info)
        elif node.op_type == "DequantizeLinear" and node.domain == "trt":
            out_name = node.output[0]
            for vi in graph.value_info:
                found = False
                if vi.name == out_name:
                    vi.type.tensor_type.elem_type = TensorProto.FLOAT
                    found = True
                    break
            if not found:
                value_info = helper.make_tensor_value_info(
                    out_name, TensorProto.FLOAT, None
                )  # None for shape
                graph.value_info.append(value_info)

    # Save updated model
    external_data_path = os.path.basename(output_path) + "_data"
    onnx.save(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_path,
    )
