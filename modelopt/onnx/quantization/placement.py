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

"""Q/DQ placement chosen by the ONNX quantization path, described independently of the graph format.

The placement lets other frontends (e.g. ``modelopt.torch.quantization``) apply the same Q/DQ
placement as the ONNX path without re-implementing its TensorRT-oriented rules.
"""

import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.qdq_graph import get_concat_eliminated_tensors
from modelopt.onnx.quantization.quantize import quantize
from modelopt.onnx.utils import save_onnx

__all__ = ["QdqPlacement", "QuantizedTensor", "find_qdq_placement", "read_qdq_placement"]


@dataclass(frozen=True)
class QuantizedTensor:
    """A tensor quantized by a single Q/DQ pair that all of its ``consumers`` read.

    Args:
        consumers: ``(node name, input index)`` of every node input reading the dequantized tensor.
        axis: Quantization axis for per-channel scales, ``None`` for per-tensor.
    """

    name: str
    consumers: tuple[tuple[str, int], ...]
    axis: int | None
    is_constant: bool


@dataclass
class QdqPlacement:
    """Which tensors of a graph are quantized, for which consumers, and with which precision.

    Args:
        scale_groups: Sets of tensor names that must share one scale (e.g. around a Concat).
    """

    precision: str
    tensors: dict[str, QuantizedTensor]
    scale_groups: list[frozenset[str]] = field(default_factory=list)


def read_qdq_placement(qdq_model: onnx.ModelProto, precision: str) -> QdqPlacement:
    """Read the placement of an ONNX model that already contains QuantizeLinear/DequantizeLinear."""
    graph = gs.import_onnx(qdq_model)
    consumers: dict[str, list[tuple[str, int]]] = defaultdict(list)
    quantize_nodes: dict[str, gs.Node] = {}
    for node in graph.nodes:
        if node.op in ("QuantizeLinear", "DequantizeLinear"):
            continue
        for index, tensor in enumerate(node.inputs):
            if not tensor.inputs or tensor.inputs[0].op != "DequantizeLinear":
                continue
            dequantize = tensor.inputs[0]
            quantized = dequantize.inputs[0]
            if not quantized.inputs or quantized.inputs[0].op != "QuantizeLinear":
                raise ValueError(
                    f"DequantizeLinear '{dequantize.name}' has no QuantizeLinear producer; "
                    "pre-quantized (DQ-only) inputs are not supported."
                )
            quantize_node = quantized.inputs[0]
            source = quantize_node.inputs[0]
            consumers[source.name].append((node.name, index))
            quantize_nodes[source.name] = quantize_node

    tensors = {}
    for name, quantize_node in quantize_nodes.items():
        source, scale = quantize_node.inputs[0], quantize_node.inputs[1]
        per_channel = np.prod(scale.shape or ()) > 1
        tensors[name] = QuantizedTensor(
            name=name,
            consumers=tuple(consumers[name]),
            axis=quantize_node.attrs.get("axis", 1) if per_channel else None,
            is_constant=isinstance(source, gs.Constant)
            or (bool(source.inputs) and source.inputs[0].op == "Constant"),
        )
    return QdqPlacement(precision=precision, tensors=tensors)


def find_qdq_placement(
    onnx_model: onnx.ModelProto, quantize_mode: str = "int8", **quantize_kwargs
) -> QdqPlacement:
    """Return the Q/DQ placement that :func:`modelopt.onnx.quantization.quantize` gives ``onnx_model``.

    The ONNX path is run on CPU with random calibration data, so only the placement is meaningful,
    not the scales. Tensor and node names refer to ``onnx_model``.

    Args:
        quantize_mode: ``"int8"`` or ``"fp8"``.
        quantize_kwargs: Extra placement options for :func:`~modelopt.onnx.quantization.quantize`,
            e.g. ``nodes_to_exclude`` or ``passes``.
    """
    if quantize_mode not in ("int8", "fp8"):
        raise ValueError(f"Unsupported quantize_mode for placement: {quantize_mode!r}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_path = os.path.join(tmp_dir, "model.onnx")
        output_path = os.path.join(tmp_dir, "model.quant.onnx")
        save_onnx(onnx_model, onnx_path)
        quantize(
            onnx_path,
            quantize_mode=quantize_mode,
            calibration_method="max",
            calibration_eps=["cpu"],
            high_precision_dtype="fp32",
            output_path=output_path,
            **quantize_kwargs,
        )
        placement = read_qdq_placement(onnx.load(output_path), quantize_mode)

    _check_names_match(placement, onnx_model)
    if "concat_elimination" in quantize_kwargs.get("passes", ["concat_elimination"]):
        consumer_nodes = {node for t in placement.tensors.values() for node, _ in t.consumers}
        groups = get_concat_eliminated_tensors(onnx_model, list(consumer_nodes))
        placement.scale_groups = list({frozenset(group) for group in groups.values()})
    return placement


def _check_names_match(placement: QdqPlacement, onnx_model: onnx.ModelProto) -> None:
    # Preprocessing in the ONNX path may rename nodes or duplicate shared constants; the placement
    # is only usable if it still refers to ``onnx_model``.
    graph = onnx_model.graph
    node_inputs = {node.name: list(node.input) for node in graph.node}
    for tensor in placement.tensors.values():
        for node_name, index in tensor.consumers:
            inputs = node_inputs.get(node_name)
            if inputs is None or index >= len(inputs) or inputs[index] != tensor.name:
                raise ValueError(
                    f"Q/DQ placement for tensor '{tensor.name}' read by '{node_name}' input {index} "
                    "does not match the input model; the ONNX path renamed it during preprocessing."
                )
