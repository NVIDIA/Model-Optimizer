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

"""Fusion-aware grouping of ONNX nodes based on TensorRT graph.json.

Parses TensorRT's exported layer information (graph.json) to understand
how ONNX operations are fused into TRT layers. Creates FusionGroups that
map back to ONNX node names, enabling subgraph-level QDQ optimization.
"""

import json
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Tuple

import onnx_graphsurgeon as gs

logger = logging.getLogger(__name__)

DEFAULT_QUANTIZABLE_OPS = {"Conv", "ConvTranspose", "MatMul", "Gemm", "Einsum"}
QUANTIZABLE_OPS = DEFAULT_QUANTIZABLE_OPS
FUSIBLE_OPS = {
    "Relu", "Sigmoid", "Tanh", "LeakyRelu", "Clip", "Add", "Abs", "Mul",
    "Sub", "Div", "Sqrt", "Pow", "BatchNormalization", "Softmax",
    "ReduceMean", "ReduceMax", "ReduceSum", "Cast", "Transpose",
    "Reshape", "Squeeze", "Unsqueeze", "Flatten",
}


@dataclass
class TRTLayer:
    """A single TensorRT layer parsed from graph.json."""
    name: str
    layer_type: str
    onnx_nodes: List[str]
    input_names: List[str]
    output_names: List[str]
    metadata_raw: str = ""


@dataclass
class FusionGroup:
    """A group of ONNX nodes that TRT fuses into one or more TRT layers.

    Attributes:
        id: Unique group index.
        trt_layer_name: Name of the TRT layer this group originates from.
        onnx_node_names: ONNX node names (from Metadata) in this group.
        has_quantizable_op: Whether any node is Conv/MatMul/Gemm etc.
        input_tensors: Boundary input tensor names for subgraph extraction.
        output_tensors: Boundary output tensor names for subgraph extraction.
        quantizable_node_names: Subset of onnx_node_names that are quantizable ops.
    """
    id: int
    trt_layer_name: str
    onnx_node_names: List[str] = field(default_factory=list)
    has_quantizable_op: bool = False
    input_tensors: List[str] = field(default_factory=list)
    output_tensors: List[str] = field(default_factory=list)
    quantizable_node_names: List[str] = field(default_factory=list)


def _parse_metadata(metadata_str: str) -> List[str]:
    """Extract ONNX node names from a TRT layer's Metadata string.

    TRT uses ASCII control characters as delimiters:
      \\x1f (US) = nodes within same fusion group
      \\x1e (RS) = different sub-groups
    """
    if not metadata_str:
        return []
    parts = re.split(r"[\x1e\x1f]", metadata_str)
    nodes = []
    for p in parts:
        m = re.search(r"\[ONNX Layer: ([^\]]+)\]", p)
        if m:
            nodes.append(m.group(1))
    return nodes


def parse_graph_json(path: str) -> List[TRTLayer]:
    """Parse a TensorRT graph.json file into TRTLayer objects."""
    with open(path, "r") as f:
        data = json.load(f)

    layers = []
    for entry in data.get("Layers", []):
        name = entry.get("Name", "")
        layer_type = entry.get("LayerType", "")
        metadata = entry.get("Metadata", "")
        onnx_nodes = _parse_metadata(metadata)
        inp_names = [t["Name"] for t in entry.get("Inputs", [])]
        out_names = [t["Name"] for t in entry.get("Outputs", [])]
        layers.append(TRTLayer(
            name=name,
            layer_type=layer_type,
            onnx_nodes=onnx_nodes,
            input_names=inp_names,
            output_names=out_names,
            metadata_raw=metadata,
        ))

    logger.info(f"Parsed {len(layers)} TRT layers from {path}")
    return layers


def _get_op_type_from_node_name(name: str, graph: gs.Graph) -> Optional[str]:
    """Look up the op_type of an ONNX node by name."""
    for node in graph.nodes:
        if node.name == name:
            return node.op
    return None


def create_fusion_groups(
    trt_layers: List[TRTLayer],
    graph: gs.Graph,
    quantizable_ops: Optional[Set[str]] = None,
) -> List[FusionGroup]:
    """Map TRT layers to FusionGroups with ONNX node resolution.

    For each TRT layer that has ONNX Metadata, creates a FusionGroup
    containing the referenced ONNX node names. Identifies which groups
    contain quantizable operations and resolves boundary tensors.

    Args:
        trt_layers: Parsed TRT layer information.
        graph: ONNX graph (graphsurgeon).
        quantizable_ops: Set of ONNX op types to treat as quantizable.
            If None, uses DEFAULT_QUANTIZABLE_OPS.
    """
    if quantizable_ops is None:
        quantizable_ops = QUANTIZABLE_OPS
    node_name_to_node = {n.name: n for n in graph.nodes}
    groups = []

    for idx, trt_layer in enumerate(trt_layers):
        if not trt_layer.onnx_nodes:
            continue

        resolved_names = []
        quantizable_names = []
        has_quant = False

        for onnx_name in trt_layer.onnx_nodes:
            node = node_name_to_node.get(onnx_name)
            if node is None:
                continue
            resolved_names.append(onnx_name)
            if node.op in quantizable_ops:
                has_quant = True
                quantizable_names.append(onnx_name)

        if not resolved_names:
            continue

        group_nodes = [node_name_to_node[n] for n in resolved_names if n in node_name_to_node]
        group_node_set = set(resolved_names)
        input_tensors, output_tensors = _find_boundary_tensors(group_nodes, group_node_set, graph)

        groups.append(FusionGroup(
            id=idx,
            trt_layer_name=trt_layer.name,
            onnx_node_names=resolved_names,
            has_quantizable_op=has_quant,
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            quantizable_node_names=quantizable_names,
        ))

    quant_groups = [g for g in groups if g.has_quantizable_op]
    logger.info(
        f"Created {len(groups)} fusion groups, "
        f"{len(quant_groups)} contain quantizable ops"
    )
    return groups


def _find_boundary_tensors(
    group_nodes: List[gs.Node],
    group_node_names: Set[str],
    graph: gs.Graph,
) -> Tuple[List[str], List[str]]:
    """Find input and output boundary tensors for a node group.

    A boundary input is a Variable tensor that is consumed by a group node
    but produced by a node outside the group (or is a graph input).
    A boundary output is a Variable tensor produced by a group node
    that is consumed by a node outside the group (or is a graph output).
    """
    graph_output_names = {t.name for t in graph.outputs}
    input_tensors = set()
    output_tensors = set()

    for node in group_nodes:
        for inp in node.inputs:
            if isinstance(inp, gs.Constant):
                continue
            if not isinstance(inp, gs.Variable):
                continue
            producer = inp.inputs[0] if inp.inputs else None
            if producer is None or producer.name not in group_node_names:
                input_tensors.add(inp.name)

        for out in node.outputs:
            if not isinstance(out, gs.Variable):
                continue
            if out.name in graph_output_names:
                output_tensors.add(out.name)
                continue
            for consumer in out.outputs:
                if consumer.name not in group_node_names:
                    output_tensors.add(out.name)
                    break

    return sorted(input_tensors), sorted(output_tensors)


def generate_graph_json(
    onnx_path: str,
    output_dir: str,
    plugin_libraries: Optional[List[str]] = None,
    strongly_typed: bool = True,
    extra_trtexec_args: Optional[List[str]] = None,
) -> str:
    """Run trtexec FP16 build to generate graph.json.

    Returns the path to the generated graph.json file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(onnx_path).stem
    graph_json_path = str(output_dir / f"{model_name}.fp16.graph.json")
    log_path = str(output_dir / f"{model_name}.fp16.build.log")

    defaults = {
        "onnx": onnx_path,
        "maxTactics": "1",
        "exportLayerInfo": graph_json_path,
        "profilingVerbosity": "detailed",
    }
    if strongly_typed:
        defaults["stronglyTyped"] = None

    if plugin_libraries:
        for lib in plugin_libraries:
            defaults.setdefault("staticPlugins", [])
            if isinstance(defaults["staticPlugins"], list):
                defaults["staticPlugins"].append(lib)
            else:
                defaults["staticPlugins"] = [lib]

    if extra_trtexec_args:
        for arg in extra_trtexec_args:
            key = arg.lstrip("-").split("=", 1)[0]
            val = arg.split("=", 1)[1] if "=" in arg else None
            defaults[key] = val

    cmd = ["trtexec"]
    for key, val in defaults.items():
        if isinstance(val, list):
            for v in val:
                cmd.append(f"--{key}={v}")
        elif val is None:
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={val}")

    logger.info(f"Running trtexec FP16 build to generate graph.json ...")
    logger.debug(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        with open(log_path, "w") as f:
            f.write(result.stdout)
            f.write(result.stderr)

        if result.returncode != 0:
            logger.error(f"trtexec failed (exit {result.returncode}), see {log_path}")
            raise RuntimeError(f"trtexec FP16 build failed: {log_path}")

        logger.info(f"graph.json generated: {graph_json_path}")
        return graph_json_path

    except subprocess.TimeoutExpired:
        logger.error("trtexec timed out after 600s")
        raise
