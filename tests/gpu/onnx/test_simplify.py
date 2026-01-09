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

import os

import onnx
import onnx_graphsurgeon as gs
import torch
from _test_utils.onnx.lib_test_models import NonSimplifiedModel, export_as_onnx
from _test_utils.onnx.quantization.utils import assert_nodes_are_quantized

from modelopt.onnx.quantization.quantize import quantize


def _create_test_model(onnx_filename):
    opset_version = 13
    in_channels = 16
    model = NonSimplifiedModel(in_channels)
    input_tensor = torch.randn(20, in_channels, 50, 100)
    export_as_onnx(
        model, input_tensor, onnx_filename, opset=opset_version, do_constant_folding=False
    )


def test_onnx_simplification(tmp_path):
    onnx_filename = os.path.join(tmp_path, "model_non_simplified.onnx")
    _create_test_model(onnx_filename)

    with open(onnx_filename) as f:
        graph = gs.import_onnx(onnx.load(f.name))

        # Check that the model contains Identity nodes, indicating that constant folding did not happen.
        identity_nodes = [n for n in graph.nodes if n.op == "Identity"]
        assert identity_nodes, "ONNX model doesn't contain Identity nodes as expected."

        # Quantize model with simplification enabled
        quantize(f.name, simplify=True, keep_intermediate_files=True)

        # Output model should be produced in the same tmp_path
        simplified_onnx_path = f.name.replace(".onnx", "_simp.onnx")
        output_onnx_path = f.name.replace(".onnx", ".quant.onnx")

        # Check that the simplified and quantized explicit models are generated
        assert os.path.isfile(simplified_onnx_path), "Simplified ONNX was not found!"
        assert os.path.isfile(output_onnx_path), "Quantized ONNX was not found!"

        # Load the simplified model and check that the model doesn't contain Identity nodes,
        #   only 2 layers (Conv->Relu).
        graph = gs.import_onnx(onnx.load(simplified_onnx_path))
        identity_nodes = [n for n in graph.nodes if n.op == "Identity"]
        assert not identity_nodes, "Simplified ONNX model contains Identity nodes but it shouldn't."
        assert len(graph.nodes) == 2, (
            f"Number of nodes doesn't match the expected: {len(graph.nodes)} vs 2."
        )
        assert all(n.op in ["Conv", "Relu"] for n in graph.nodes), (
            "Graph contains more ops than expected."
        )

        # Load the output model and check QDQ node placements
        graph = gs.import_onnx(onnx.load(output_onnx_path))

        # Check that the default quantization happened successfully: Conv layer should be quantized
        quantizable_nodes = [n for n in graph.nodes if n.op == "Conv"]
        assert assert_nodes_are_quantized(quantizable_nodes)
