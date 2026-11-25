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

"""The ONNX export of a graph-placed torch model has the ONNX path's Q/DQ placement."""

import ast
from collections import Counter

import pytest
import torch

pytest.importorskip("onnxscript")
torchvision = pytest.importorskip("torchvision")

import modelopt.torch.quantization as mtq
from modelopt.onnx.quantization.placement import find_qdq_placement, read_qdq_placement
from modelopt.torch.quantization._dynamo_onnx import _get_dynamo_onnx_translation_table
from modelopt.torch.quantization.utils import export_torch_mode


def _placement_keys(onnx_model, placement, aliases):
    # Both ONNX models are translated from the same FX graph, so their nodes are compared by the FX
    # node each one was translated from.
    def fx_name(node):
        metadata = {prop.key: prop.value for prop in node.metadata_props}
        name = ast.literal_eval(metadata["pkg.torch.onnx.name_scopes"])[-1]
        return aliases.get(name, name)

    nodes = {node.name: node for node in onnx_model.graph.node}
    return Counter(
        (tensor.is_constant, frozenset(fx_name(nodes[name]) for name, _ in tensor.consumers))
        for tensor in placement.tensors.values()
    )


@pytest.mark.parametrize(
    "model_fn", [torchvision.models.resnet50, torchvision.models.efficientnet_v2_s]
)
def test_graph_placement_matches_onnx_path(model_fn):
    model = model_fn().cuda().eval()
    example = torch.randn(1, 3, 224, 224, device="cuda")

    program = torch.onnx.export(torch.export.export(model, (example,), strict=False), dynamo=True)
    expected = find_qdq_placement(program.model_proto, log_level="WARNING")
    # The exporter folds batch norm into the preceding conv and names the result after the batch
    # norm; quantized conv weights keep it from folding the torch-placed model.
    aliases = {
        node.name: node.args[0].name
        for node in program.exported_program.graph.nodes
        if node.target == torch.ops.aten._native_batch_norm_legit_no_training.default
    }

    graph_module = mtq.quantize_graph(
        model, mtq.INT8_DEFAULT_CFG, (example,), lambda m: m(example), log_level="WARNING"
    )
    with export_torch_mode(), torch.inference_mode():
        quantized = torch.onnx.export(
            graph_module,
            (example,),
            dynamo=True,
            opset_version=23,
            custom_translation_table=_get_dynamo_onnx_translation_table(),
        ).model_proto
    actual = read_qdq_placement(quantized, "int8")

    assert _placement_keys(quantized, actual, aliases) == _placement_keys(
        program.model_proto, expected, aliases
    )
