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

import os

import onnx
import onnx_graphsurgeon as gs
from _test_utils.onnx.lib_test_models import export_as_onnx
from _test_utils.torch.vision_models import get_tiny_resnet_and_input

from modelopt.onnx.quantization.graph_utils import (
    build_non_residual_input_map,
    classify_partition_nodes,
    filter_quantizable_kgen_heads,
)
from modelopt.onnx.quantization.partitioning import (
    find_fusible_partitions,
    get_skipped_output_layers,
)


def test_partitioning(tmp_path):
    # Small enough model to verify different statistics manually
    model, args, kwargs = get_tiny_resnet_and_input()
    assert not kwargs
    onnx_path = os.path.join(tmp_path, "tiny_resnet.onnx")
    export_as_onnx(model, args, onnx_filename=onnx_path)

    onnx_model = onnx.load(onnx_path)
    graph = gs.import_onnx(onnx_model)
    quantizable_op_types = [
        "Add",
        "AveragePool",
        "BatchNormalization",
        "Clip",
        "Conv",
        "ConvTranspose",
        "Gemm",
        "GlobalAveragePool",
        "MatMul",
        "MaxPool",
        "Mul",
    ]

    non_residual_inputs, _ = build_non_residual_input_map(graph)

    assert "/layer1/layer1.0/Add" in non_residual_inputs
    assert len(non_residual_inputs) == 5

    partitioned_nodes = set()
    cask_fusible_partitions, kgen_partitions = find_fusible_partitions(
        graph,
        partitioned_nodes,
        non_residual_inputs,
    )
    assert len(cask_fusible_partitions) == 15
    assert len(kgen_partitions) == 0

    non_quantizable_nodes, quantizable_partition_nodes, no_quantize_inputs = (
        classify_partition_nodes(cask_fusible_partitions)
    )
    assert len(non_quantizable_nodes) == 6
    assert len(quantizable_partition_nodes) == 15
    assert len(no_quantize_inputs) == 5

    quantizable_kgen_heads, _ = filter_quantizable_kgen_heads(
        cask_fusible_partitions,
        kgen_partitions,
        quantizable_op_types,
        graph,
    )
    assert not quantizable_kgen_heads

    skip_list = get_skipped_output_layers(graph, [])
    assert not skip_list
