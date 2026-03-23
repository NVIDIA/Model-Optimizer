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

"""Subgraph-based QDQ autotune workflow.

Uses fusion-aware subgraph extraction and heuristic QDQ schemes to optimize
Q/DQ placement. Reduces autotune time from ~25 hours to ~30 minutes by:
  1. Grouping ONNX nodes by TRT fusion boundaries (graph.json)
  2. Profiling isolated subgraphs instead of full model
  3. Using domain-informed heuristic schemes instead of random mutation
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import onnx
import onnx_graphsurgeon as gs

from modelopt.onnx.quantization.autotune.fusion_grouping import (
    DEFAULT_QUANTIZABLE_OPS,
    FusionGroup,
    create_fusion_groups,
    generate_graph_json,
    parse_graph_json,
)
from modelopt.onnx.op_types import get_bool_ops, get_comparison_ops, get_value_check_ops

_BOOL_OUTPUT_OPS = get_bool_ops() | get_comparison_ops() | get_value_check_ops()
from modelopt.onnx.quantization.autotune.subgraph_extractor import (
    extract_subgraph,
    extract_subgraph_by_nodes,
)
from modelopt.onnx.quantization.autotune.workflows import benchmark_onnx_model

logger = logging.getLogger(__name__)

def _get_fp8_dtype():
    """Resolve FP8 dtype with explicit fallback warning."""
    try:
        import ml_dtypes  # noqa: F401

        return np.dtype("float8_e4m3fn")
    except (ImportError, TypeError):
        logger.warning(
            "FP8 dtype (float8_e4m3fn) unavailable; install ml_dtypes for native FP8. "
            "Falling back to int8 representation."
        )
        return np.int8


QUANT_DTYPES = {
    "int8": np.int8,
    "fp8": _get_fp8_dtype(),
}

MIN_CHANNELS_FOR_QUANT = 64

CACHE_VERSION = 1


# ── Shape inference helpers ─────────────────────────────────────────────────

def _parse_shape_spec(spec_str: str) -> Dict[str, List[int]]:
    """Parse trtexec shape spec like 'a:1x2x3,b:4x5' into {'a': [1,2,3], 'b': [4,5]}."""
    result: Dict[str, List[int]] = {}
    if not spec_str:
        return result
    for item in spec_str.split(","):
        parts = item.split(":")
        if len(parts) == 2:
            name = parts[0].strip()
            shape = [int(d) for d in parts[1].strip().split("x")]
            result[name] = shape
    return result


def _extract_shape_specs(
    extra_trtexec_args: Optional[List[str]],
) -> Tuple[Dict[str, List[int]], Dict[str, List[int]], Dict[str, List[int]]]:
    """Extract --minShapes/--optShapes/--maxShapes from trtexec arg list."""
    min_s: Dict[str, List[int]] = {}
    opt_s: Dict[str, List[int]] = {}
    max_s: Dict[str, List[int]] = {}
    if not extra_trtexec_args:
        return min_s, opt_s, max_s
    for arg in extra_trtexec_args:
        if arg.startswith("--minShapes="):
            min_s = _parse_shape_spec(arg.split("=", 1)[1])
        elif arg.startswith("--optShapes="):
            opt_s = _parse_shape_spec(arg.split("=", 1)[1])
        elif arg.startswith("--maxShapes="):
            max_s = _parse_shape_spec(arg.split("=", 1)[1])
    return min_s, opt_s, max_s


def _infer_all_tensor_shapes(
    model_path: str, input_shapes: Dict[str, List[int]],
) -> Dict[str, List[int]]:
    """Run ONNX shape inference with concrete input shapes.

    Returns {tensor_name: [d1, d2, ...]} for every tensor whose shape is
    fully resolved (all dims > 0).
    """
    model = onnx.load(model_path)

    for inp in model.graph.input:
        if inp.name not in input_shapes:
            continue
        shape = input_shapes[inp.name]
        dim_proto = inp.type.tensor_type.shape.dim
        while len(dim_proto) > len(shape):
            dim_proto.pop()
        while len(dim_proto) < len(shape):
            dim_proto.add()
        for i, d in enumerate(shape):
            dim_proto[i].ClearField("dim_param")
            dim_proto[i].dim_value = d

    try:
        inferred = onnx.shape_inference.infer_shapes(model, data_prop=True)
    except Exception as e:
        logger.warning(f"Shape inference failed: {e}")
        return {}

    shapes: Dict[str, List[int]] = {}

    def _collect(proto_list):
        for vi in proto_list:
            try:
                if not vi.type.HasField("tensor_type"):
                    continue
                if not vi.type.tensor_type.HasField("shape"):
                    continue
                dims = []
                for dim in vi.type.tensor_type.shape.dim:
                    dims.append(dim.dim_value if dim.dim_value > 0 else -1)
                if dims and all(d > 0 for d in dims):
                    shapes[vi.name] = dims
            except Exception:
                pass

    _collect(inferred.graph.input)
    _collect(inferred.graph.value_info)
    _collect(inferred.graph.output)
    return shapes


def _build_subgraph_shape_args(
    subgraph_input_names: List[str],
    min_shapes: Dict[str, List[int]],
    opt_shapes: Dict[str, List[int]],
    max_shapes: Dict[str, List[int]],
) -> Optional[List[str]]:
    """Construct trtexec --minShapes/--optShapes/--maxShapes for a subgraph."""
    min_specs, opt_specs, max_specs = [], [], []
    for name in subgraph_input_names:
        if name in min_shapes:
            min_specs.append(f"{name}:{'x'.join(str(d) for d in min_shapes[name])}")
        if name in opt_shapes:
            opt_specs.append(f"{name}:{'x'.join(str(d) for d in opt_shapes[name])}")
        if name in max_shapes:
            max_specs.append(f"{name}:{'x'.join(str(d) for d in max_shapes[name])}")

    args: List[str] = []
    if min_specs:
        args.append(f"--minShapes={','.join(min_specs)}")
    if opt_specs:
        args.append(f"--optShapes={','.join(opt_specs)}")
    if max_specs:
        args.append(f"--maxShapes={','.join(max_specs)}")
    return args or None


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class QDQScheme:
    """A heuristic QDQ insertion scheme for a fusion group."""
    name: str
    target_tensors: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class SchemeResult:
    """Result from profiling a single scheme on a subgraph."""
    scheme: QDQScheme
    latency_ms: float
    success: bool
    compute_ms: float = 0.0


@dataclass
class LayerTiming:
    """Per-layer timing from trtexec --exportProfile."""
    name: str
    median_ms: float
    time_pct: float


@dataclass
class GroupResult:
    """Aggregated profiling results for a fusion group."""
    group: FusionGroup
    baseline_latency_ms: float
    best_scheme: Optional[QDQScheme]
    best_latency_ms: float
    all_results: List[SchemeResult] = field(default_factory=list)
    baseline_compute_ms: float = 0.0
    best_compute_ms: float = 0.0

    @property
    def speedup(self) -> float:
        if (
            self.baseline_latency_ms <= 0
            or self.baseline_latency_ms == float("inf")
            or self.best_latency_ms == float("inf")
        ):
            return 0.0
        return self.baseline_latency_ms / self.best_latency_ms

    @property
    def compute_speedup(self) -> float:
        """Speedup based on compute-only layers (excludes reformat overhead)."""
        if self.baseline_compute_ms > 0 and self.best_compute_ms > 0:
            return self.baseline_compute_ms / self.best_compute_ms
        return self.speedup


# ── Profile parsing helpers ────────────────────────────────────────────────

def _is_reformat_layer(name: str) -> bool:
    """Return True for TRT data-layout reformatting layers."""
    return "reformat" in name.lower()


def _parse_profile_json(profile_path: str) -> List[LayerTiming]:
    """Parse trtexec --exportProfile JSON into LayerTiming list."""
    try:
        with open(profile_path) as f:
            data = json.load(f)
        return [
            LayerTiming(
                name=entry["name"],
                median_ms=entry.get("medianMs", entry.get("averageMs", 0)),
                time_pct=entry.get("percentage", 0),
            )
            for entry in data
        ]
    except Exception:
        return []


def _compute_time_from_profile(profile: List[LayerTiming]) -> float:
    """Sum median times of compute layers, excluding reformat overhead."""
    if not profile:
        return 0.0
    return sum(l.median_ms for l in profile if not _is_reformat_layer(l.name))


# ── Cache helpers ──────────────────────────────────────────────────────────

def _save_cache(cache_path: Path, data: dict):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(cache_path)


def _load_cache(cache_path: Path) -> Optional[dict]:
    if not cache_path.exists():
        return None
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return None


def _group_result_to_cache_entry(gr: GroupResult) -> dict:
    return {
        "group_id": gr.group.id,
        "trt_layer_name": gr.group.trt_layer_name,
        "baseline_latency_ms": gr.baseline_latency_ms,
        "best_scheme_name": gr.best_scheme.name if gr.best_scheme else "baseline",
        "best_scheme_target_tensors": (
            gr.best_scheme.target_tensors if gr.best_scheme else []
        ),
        "best_scheme_description": (
            gr.best_scheme.description if gr.best_scheme else ""
        ),
        "best_latency_ms": gr.best_latency_ms,
        "baseline_compute_ms": gr.baseline_compute_ms,
        "best_compute_ms": gr.best_compute_ms,
    }


def _cache_entry_to_group_result(entry: dict) -> GroupResult:
    """Reconstruct a minimal GroupResult from a cache entry (for Phase 3)."""
    group = FusionGroup(
        id=entry["group_id"],
        trt_layer_name=entry["trt_layer_name"],
    )
    best_scheme = None
    if (entry.get("best_scheme_name", "baseline") != "baseline"
            and entry.get("best_scheme_target_tensors")):
        best_scheme = QDQScheme(
            name=entry["best_scheme_name"],
            target_tensors=entry["best_scheme_target_tensors"],
            description=entry.get("best_scheme_description", ""),
        )
    return GroupResult(
        group=group,
        baseline_latency_ms=entry["baseline_latency_ms"],
        best_scheme=best_scheme,
        best_latency_ms=entry["best_latency_ms"],
        baseline_compute_ms=entry.get("baseline_compute_ms", 0),
        best_compute_ms=entry.get("best_compute_ms", 0),
    )


# ── QDQ insertion (simple, direct approach) ─────────────────────────────────

def _get_tensor_dtype(tensor) -> np.dtype:
    """Get the numpy dtype of a gs.Variable or gs.Constant."""
    if isinstance(tensor, gs.Constant):
        return tensor.values.dtype
    if tensor.dtype is not None:
        return np.dtype(tensor.dtype)
    return np.dtype(np.float32)


def _make_scale_dtype(tensor_dtype: np.dtype) -> type:
    name = tensor_dtype.name
    if name == "float16":
        return np.float16
    return np.float32


def derive_nodes_to_quantize(
    graph: gs.Graph,
    target_tensor_names: List[str],
) -> Tuple[List[str], List[Tuple]]:
    """Convert target tensor names into (nodes_to_quantize, no_quantize_inputs).

    Maps the autotune output (which tensors to quantize) into the format
    expected by modelopt's standard quantize() API. This ensures the final
    model uses ORT's QDQQuantizer with per-channel weights, bias exclusion, etc.

    Returns:
        nodes_to_quantize: Node names whose inputs include target tensors.
        no_quantize_inputs: (producer_node, consumer_node, tensor_name) tuples
            for inputs of quantized nodes that should NOT get Q/DQ.
    """
    target_set = set(target_tensor_names)
    nodes_to_quantize: List[str] = []
    no_quantize_inputs: List[Tuple] = []
    seen_nodes: set = set()

    for node in graph.nodes:
        covered_inputs = set()
        for inp_idx, inp in enumerate(node.inputs):
            tname = getattr(inp, "name", None)
            if tname and tname in target_set:
                covered_inputs.add(inp_idx)

        if not covered_inputs:
            continue

        if node.name not in seen_nodes:
            nodes_to_quantize.append(node.name)
            seen_nodes.add(node.name)

            for inp in node.inputs:
                if inp.inputs:
                    producer = inp.inputs[0]
                    if hasattr(producer, "name") and producer.name not in seen_nodes:
                        nodes_to_quantize.append(producer.name)
                        seen_nodes.add(producer.name)

        for inp_idx, inp in enumerate(node.inputs):
            tname = getattr(inp, "name", None)
            if tname and inp_idx not in covered_inputs and inp.inputs:
                no_quantize_inputs.append((inp.inputs[0], node, tname))

    return nodes_to_quantize, no_quantize_inputs


def finalize_with_quantize(
    model_path: str,
    output_path: str,
    target_tensor_names: List[str],
    quant_type: str = "int8",
    calibration_data=None,
    calibration_method: str = "entropy",
    calibration_shapes: Optional[str] = None,
    calibration_eps: Optional[List[str]] = None,
    high_precision_dtype: str = "fp16",
    use_external_data_format: bool = False,
) -> str:
    """Produce the final calibrated model using modelopt's standard quantize().

    Instead of inserting placeholder Q/DQ (scale=0.1), this calls modelopt's
    ORT-based quantization with real calibration, per-channel weights, bias
    exclusion, and all standard TRT-guided options.

    Args:
        model_path: Path to the original (unquantized) ONNX model.
        output_path: Where to save the final calibrated model.
        target_tensor_names: Tensor names to quantize (from autotune).
        quant_type: "int8" or "fp8".
        calibration_data: Calibration data (numpy array or dict).
        calibration_method: ORT calibration method ("entropy", "minmax", etc).
        calibration_shapes: Shape specification string for calibration.
        calibration_eps: Execution providers for calibration.
        high_precision_dtype: Non-quantized precision ("fp16" or "fp32").
        use_external_data_format: Whether to use external data format for large models.

    Returns:
        Path to the final calibrated model.
    """
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)

    nodes_to_quantize, no_quantize_inputs = derive_nodes_to_quantize(
        graph, target_tensor_names,
    )

    logger.info(
        f"Finalizing with modelopt quantize(): "
        f"{len(nodes_to_quantize)} nodes, "
        f"{len(no_quantize_inputs)} excluded inputs"
    )

    from modelopt.onnx.quantization.quantize import quantize

    quantize(
        onnx_path=model_path,
        quantize_mode=quant_type,
        calibration_data=calibration_data,
        calibration_method=calibration_method,
        calibration_shapes=calibration_shapes,
        calibration_eps=calibration_eps or ["cpu", "cuda:0", "trt"],
        nodes_to_quantize=nodes_to_quantize,
        output_path=output_path,
        high_precision_dtype=high_precision_dtype,
        use_external_data_format=use_external_data_format,
        no_quantize_inputs=no_quantize_inputs,
    )

    logger.info(f"Calibrated model saved: {output_path}")
    return output_path


def insert_qdq_on_graph(
    graph: gs.Graph,
    target_tensor_names: List[str],
    quant_type: str = "int8",
    scale: float = 0.1,
    prefix: str = "subq",
    already_qdqd: Optional[Set[str]] = None,
    do_cleanup: bool = True,
) -> gs.Graph:
    """Insert Q/DQ pairs at specified tensor locations directly.

    Unlike the autotuner's pattern-based system, this operates on explicit
    tensor names with no pattern-relative addressing. Suitable for subgraph-level
    insertion where we know exactly which tensors to quantize.

    Args:
        prefix: Unique prefix for generated node/tensor names (avoids collisions
            when multiple groups insert QDQ on the same graph).
        already_qdqd: If provided, tensor names already carrying QDQ are skipped
            and newly processed names are added to this set.
        do_cleanup: Run graph.cleanup().toposort() after insertion. Set False
            when batching multiple insertions.
    """
    tensors = graph.tensors()
    quant_np = QUANT_DTYPES.get(quant_type, np.int8)
    inserted = 0

    for tname in target_tensor_names:
        if already_qdqd is not None and tname in already_qdqd:
            logger.debug(f"Tensor '{tname}' already has QDQ, skipping")
            continue

        tensor = tensors.get(tname)
        if tensor is None:
            logger.debug(f"Tensor '{tname}' not found, skipping")
            continue

        if isinstance(tensor, gs.Variable):
            skip = False
            if tensor.dtype is not None:
                try:
                    if np.dtype(tensor.dtype) not in [
                        np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.float64),
                    ]:
                        logger.debug(f"Tensor '{tname}' has non-float dtype {tensor.dtype}, skipping QDQ")
                        skip = True
                except (TypeError, AttributeError):
                    pass
            if not skip and tensor.inputs:
                for producer in tensor.inputs:
                    if hasattr(producer, "op") and producer.op in _BOOL_OUTPUT_OPS:
                        logger.debug(f"Tensor '{tname}' produced by Bool op '{producer.op}', skipping QDQ")
                        skip = True
                        break
            if skip:
                continue

        is_const = isinstance(tensor, gs.Constant)
        t_dtype = _get_tensor_dtype(tensor)
        scale_dtype = _make_scale_dtype(t_dtype)
        t_shape = tensor.values.shape if is_const else tensor.shape

        safe_name = tname.replace("/", "_").replace(":", "_")

        q_scale_val = np.array([scale], dtype=scale_dtype)
        q_zp_val = np.array([0], dtype=np.int8)

        q_out = gs.Variable(
            f"{tname}_{prefix}_quantized", dtype=quant_np, shape=t_shape,
        )
        q_node = gs.Node(
            op="QuantizeLinear",
            name=f"{prefix}_Q_{safe_name}",
            inputs=[
                tensor,
                gs.Constant(f"{prefix}_scale_{safe_name}", values=q_scale_val),
                gs.Constant(f"{prefix}_zp_{safe_name}", values=q_zp_val),
            ],
            outputs=[q_out],
        )

        dq_out = gs.Variable(
            f"{tname}_{prefix}_dequantized", dtype=t_dtype, shape=t_shape,
        )
        dq_node = gs.Node(
            op="DequantizeLinear",
            name=f"{prefix}_DQ_{safe_name}",
            inputs=[
                q_out,
                gs.Constant(f"{prefix}_dqscale_{safe_name}", values=q_scale_val.copy()),
                gs.Constant(f"{prefix}_dqzp_{safe_name}", values=q_zp_val.copy()),
            ],
            outputs=[dq_out],
        )

        if not is_const:
            consumers = list(tensor.outputs)
        else:
            consumers = [
                n for n in graph.nodes
                if any(
                    (hasattr(inp, "name") and inp.name == tname) for inp in n.inputs
                )
            ]

        for consumer in consumers:
            if consumer is q_node:
                continue
            for i, inp in enumerate(consumer.inputs):
                if hasattr(inp, "name") and inp.name == tname:
                    consumer.inputs[i] = dq_out

        graph.nodes.extend([q_node, dq_node])
        inserted += 1

        if already_qdqd is not None:
            already_qdqd.add(tname)

    if inserted > 0 and do_cleanup:
        graph.cleanup().toposort()
    logger.debug(f"Inserted {inserted} Q/DQ pairs on subgraph")
    return graph


# ── Heuristic scheme generation ─────────────────────────────────────────────

_BIAS_INPUT_INDEX = {
    "Conv": 2,
    "ConvTranspose": 2,
    "Gemm": 2,
}


def _find_weight_tensors(
    graph: gs.Graph, node_names: List[str],
    quantizable_ops: Set[str] = DEFAULT_QUANTIZABLE_OPS,
) -> List[str]:
    """Find weight (Constant) input tensor names for given quantizable nodes.

    Bias inputs are excluded to match modelopt's default behavior
    (QuantizeBias=False).  For Conv/ConvTranspose/Gemm, input index 2
    is the bias; for other ops all Constant inputs are treated as weights.
    """
    node_set = set(node_names)
    result = []
    for node in graph.nodes:
        if node.name not in node_set:
            continue
        if node.op not in quantizable_ops:
            continue
        bias_idx = _BIAS_INPUT_INDEX.get(node.op)
        for inp_idx, inp in enumerate(node.inputs):
            if isinstance(inp, gs.Constant):
                if bias_idx is not None and inp_idx == bias_idx:
                    continue
                result.append(inp.name)
    return result


def _find_activation_tensors(
    graph: gs.Graph, node_names: List[str],
    quantizable_ops: Set[str] = DEFAULT_QUANTIZABLE_OPS,
) -> List[str]:
    """Find activation (Variable) input tensor names for given quantizable nodes."""
    node_set = set(node_names)
    result = []
    seen = set()
    for node in graph.nodes:
        if node.name not in node_set:
            continue
        if node.op not in quantizable_ops:
            continue
        for inp in node.inputs:
            if isinstance(inp, gs.Variable) and inp.name not in seen:
                if inp.inputs:
                    producer = inp.inputs[0]
                    if hasattr(producer, "op") and producer.op in _BOOL_OUTPUT_OPS:
                        continue
                if inp.dtype is not None:
                    try:
                        if np.dtype(inp.dtype) not in [
                            np.dtype(np.float32), np.dtype(np.float16), np.dtype(np.float64),
                        ]:
                            continue
                    except (TypeError, AttributeError):
                        pass
                result.append(inp.name)
                seen.add(inp.name)
    return result


def _find_small_channel_nodes(
    graph: gs.Graph, node_names: List[str],
    quantizable_ops: Set[str] = DEFAULT_QUANTIZABLE_OPS,
) -> Set[str]:
    """Find quantizable nodes with channel count < MIN_CHANNELS_FOR_QUANT."""
    node_set = set(node_names)
    small = set()
    for node in graph.nodes:
        if node.name not in node_set or node.op not in quantizable_ops:
            continue
        for inp in node.inputs:
            if isinstance(inp, gs.Constant) and inp.values.ndim >= 2:
                out_channels = inp.values.shape[0]
                if out_channels < MIN_CHANNELS_FOR_QUANT:
                    small.add(node.name)
                    break
    return small


def generate_heuristic_schemes(
    graph: gs.Graph,
    group: FusionGroup,
) -> List[QDQScheme]:
    """Generate domain-informed QDQ schemes for a fusion group.

    Schemes:
      0) No QDQ (subgraph baseline)
      1) Full QDQ: weights + activations
      2) Weight-only QDQ
      3) Full QDQ minus small-channel ops
      4) Activation-only QDQ
    """
    q_nodes = group.quantizable_node_names
    weight_tensors = _find_weight_tensors(graph, q_nodes)
    act_tensors = _find_activation_tensors(graph, q_nodes)

    schemes = [
        QDQScheme(name="baseline", target_tensors=[], description="No QDQ"),
    ]

    if weight_tensors or act_tensors:
        schemes.append(QDQScheme(
            name="full_qdq",
            target_tensors=weight_tensors + act_tensors,
            description="Full QDQ on all weights + activations",
        ))

    if weight_tensors:
        schemes.append(QDQScheme(
            name="weight_only",
            target_tensors=weight_tensors,
            description="Weight-only QDQ",
        ))

    small_nodes = _find_small_channel_nodes(graph, q_nodes)
    if small_nodes and len(small_nodes) < len(q_nodes):
        big_nodes = [n for n in q_nodes if n not in small_nodes]
        big_w = _find_weight_tensors(graph, big_nodes)
        big_a = _find_activation_tensors(graph, big_nodes)
        if big_w or big_a:
            schemes.append(QDQScheme(
                name="skip_small_channels",
                target_tensors=big_w + big_a,
                description=f"Full QDQ skipping {len(small_nodes)} small-channel ops",
            ))

    if act_tensors:
        schemes.append(QDQScheme(
            name="activation_only",
            target_tensors=act_tensors,
            description="Activation-only QDQ",
        ))

    return schemes


# ── Subgraph workflow ───────────────────────────────────────────────────────

def subgraph_autotuning_workflow(
    model_path: str,
    output_dir: Path,
    graph_json_path: str = None,
    quant_type: str = "int8",
    plugin_libraries: Optional[List[str]] = None,
    schemes_per_group: int = 5,
    strongly_typed: bool = True,
    extra_trtexec_args: Optional[List[str]] = None,
    incremental_validation: bool = True,
    quantizable_ops: Optional[Set[str]] = None,
    calibration_data=None,
    calibration_method: str = "entropy",
    calibration_shapes: Optional[str] = None,
    calibration_eps: Optional[List[str]] = None,
    high_precision_dtype: str = "fp16",
    use_external_data_format: bool = False,
    skip_calibration: bool = False,
) -> str:
    """Run subgraph-based QDQ autotune workflow.

    Phase 1: Parse graph.json (or generate via trtexec) and create fusion groups.
    Phase 2: For each quantizable group, extract subgraph, test heuristic QDQ
             schemes using per-layer profiling (excludes reformat overhead).
    Phase 3: Merge best schemes into the full model.  When
             *incremental_validation* is enabled (default), each group is
             validated one-by-one against the full model; groups that cause
             a regression are rejected.
    Phase 4: Produce the final calibrated model using modelopt's standard
             quantize() with proper per-channel weights, bias exclusion, and
             real calibration data (unless *skip_calibration* is True).

    Outputs:
        optimized_raw.onnx       – all qualifying QDQ groups applied (placeholder Q/DQ).
        optimized_final.onnx     – incrementally validated model (placeholder Q/DQ).
        optimized_calibrated.onnx – calibrated model via modelopt quantize()
                                    (only when calibration_data is provided).

    Caching: intermediate results are persisted to ``autotune_cache.json``
    inside *output_dir*.  If the process is interrupted, re-running with the
    same *output_dir* will resume from the last checkpoint.

    Args:
        model_path: Path to the input ONNX model.
        output_dir: Directory for output files.
        graph_json_path: Path to an existing graph.json (FP16 baseline).
                         If None, one will be generated via trtexec.
        quant_type: "int8" or "fp8".
        plugin_libraries: Optional TRT plugin .so paths.
        schemes_per_group: Max schemes to test per group (capped by heuristics).
        strongly_typed: Use --stronglyTyped for trtexec graph.json generation.
        extra_trtexec_args: Optional extra arguments to pass to trtexec.
        incremental_validation: If True (default), Phase 3 validates groups
            one-by-one against the full model, rejecting regressions.
        quantizable_ops: Custom set of ONNX op types to consider quantizable.
        calibration_data: Numpy array or dict of numpy arrays for calibration.
            If None and skip_calibration is False, random calibration is used.
        calibration_method: ORT calibration method ("entropy", "minmax", etc).
        calibration_shapes: Shape specification, e.g. "input:1x3x224x224".
        calibration_eps: Execution providers for calibration.
        high_precision_dtype: Non-quantized precision ("fp16" or "fp32").
        use_external_data_format: Use external data format for large models.
        skip_calibration: If True, skip Phase 4 calibration and only output
            the placeholder Q/DQ model (for latency-only analysis).

    Returns:
        Path to the final optimized ONNX model (calibrated if available).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    subgraph_dir = output_dir / "subgraphs"
    subgraph_dir.mkdir(exist_ok=True)
    cache_path = output_dir / "autotune_cache.json"

    t_start = time.time()

    # ─── Load cache ──────────────────────────────────────────────────────
    cache = _load_cache(cache_path) or {
        "version": CACHE_VERSION, "model_path": model_path,
    }
    cached_p2: dict = cache.get("phase2", {})
    cached_p2_ids: Set[int] = {
        e["group_id"] for e in cached_p2.get("results", [])
    }
    cached_p3: dict = cache.get("phase3", {})

    if cached_p2_ids:
        logger.info(f"Cache: {len(cached_p2_ids)} Phase-2 groups already profiled")
    if cached_p3.get("validated"):
        logger.info(
            f"Cache: Phase-3 progress – "
            f"{len(cached_p3['validated'])} groups validated"
        )

    # ─── Phase 1: Fusion-Aware Grouping ──────────────────────────────────

    logger.info("=" * 60)
    logger.info("Phase 1: Fusion-Aware Grouping")
    logger.info("=" * 60)

    logger.info(f"Loading model: {model_path}")
    model = onnx.load(model_path)
    graph = gs.import_onnx(model)
    logger.info(f"Model has {len(graph.nodes)} nodes")

    if graph_json_path is None:
        logger.info("No graph.json provided, generating via trtexec FP16 build...")
        graph_json_path = generate_graph_json(
            model_path, str(output_dir), plugin_libraries, strongly_typed,
            extra_trtexec_args=extra_trtexec_args,
        )

    trt_layers = parse_graph_json(graph_json_path)
    all_groups = create_fusion_groups(trt_layers, graph, quantizable_ops=quantizable_ops)
    quant_groups = [g for g in all_groups if g.has_quantizable_op]

    logger.info(f"Phase 1 complete: {len(quant_groups)} quantizable groups "
                f"(out of {len(all_groups)} total)")

    # ─── Infer intermediate tensor shapes for dynamic-shape support ─────

    min_spec, opt_spec, max_spec = _extract_shape_specs(extra_trtexec_args)
    has_dynamic_shapes = bool(min_spec or opt_spec or max_spec)

    min_tensor_shapes: Dict[str, List[int]] = {}
    opt_tensor_shapes: Dict[str, List[int]] = {}
    max_tensor_shapes: Dict[str, List[int]] = {}

    if has_dynamic_shapes:
        logger.info("Inferring intermediate tensor shapes for subgraph profiling...")
        if min_spec:
            min_tensor_shapes = _infer_all_tensor_shapes(model_path, min_spec)
        if opt_spec:
            opt_tensor_shapes = _infer_all_tensor_shapes(model_path, opt_spec)
        if max_spec:
            max_tensor_shapes = _infer_all_tensor_shapes(model_path, max_spec)
        logger.info(
            f"Shape inference resolved {len(opt_tensor_shapes or min_tensor_shapes)} "
            f"intermediate tensor shapes"
        )

    # ─── Phase 2: Subgraph Profiling (with per-layer comparison) ─────────

    logger.info("=" * 60)
    logger.info("Phase 2: Subgraph Profiling (per-layer comparison)")
    logger.info("=" * 60)

    group_results: List[GroupResult] = []
    skipped = 0
    cached_hit = 0

    for gi, group in enumerate(quant_groups):
        logger.info(
            f"[{gi+1}/{len(quant_groups)}] Group '{group.trt_layer_name}' "
            f"({len(group.onnx_node_names)} nodes, "
            f"{len(group.quantizable_node_names)} quantizable)"
        )

        # ── Cache hit: reuse previous profiling result ──
        if group.id in cached_p2_ids:
            entry = next(
                e for e in cached_p2["results"] if e["group_id"] == group.id
            )
            gr = _cache_entry_to_group_result(entry)
            gr.group = group  # attach the real FusionGroup (has node lists)
            group_results.append(gr)
            cached_hit += 1
            logger.info(
                f"  [cached] best={entry.get('best_scheme_name','baseline')} "
                f"compute_speedup={gr.compute_speedup:.3f}x"
            )
            continue

        if not group.input_tensors or not group.output_tensors:
            logger.warning("  Skipping: no boundary tensors resolved")
            skipped += 1
            continue

        # Extract subgraph
        try:
            sub_bytes = extract_subgraph(
                graph, group.input_tensors, group.output_tensors,
            )
        except Exception as e:
            logger.warning(f"  Subgraph extraction failed: {e}")
            skipped += 1
            continue

        # Build per-subgraph shape args only if the subgraph has dynamic inputs
        subgraph_extra_args = None
        if has_dynamic_shapes:
            sub_model_proto = onnx.load_from_string(sub_bytes)
            has_dynamic_inputs = any(
                any(
                    (not dim.dim_value and dim.dim_value != 0)
                    for dim in inp.type.tensor_type.shape.dim
                )
                for inp in sub_model_proto.graph.input
                if inp.type.HasField("tensor_type")
                and inp.type.tensor_type.HasField("shape")
            )
            if has_dynamic_inputs:
                subgraph_extra_args = _build_subgraph_shape_args(
                    group.input_tensors,
                    min_tensor_shapes, opt_tensor_shapes, max_tensor_shapes,
                )
                if subgraph_extra_args:
                    logger.debug(f"  Subgraph shape args: {subgraph_extra_args}")
            else:
                logger.debug("  Subgraph has static inputs, skipping shape args")
            del sub_model_proto

        # Generate heuristic schemes
        schemes = generate_heuristic_schemes(graph, group)
        schemes = schemes[:schemes_per_group]
        logger.info(f"  Testing {len(schemes)} schemes")

        results: List[SchemeResult] = []
        baseline_lat = float("inf")
        baseline_compute = 0.0

        for si, scheme in enumerate(schemes):
            if scheme.target_tensors:
                sub_model = onnx.load_from_string(sub_bytes)
                sub_graph = gs.import_onnx(sub_model)
                insert_qdq_on_graph(sub_graph, scheme.target_tensors, quant_type)
                modified_model = gs.export_onnx(sub_graph)
                model_input = modified_model.SerializeToString()
            else:
                model_input = sub_bytes

            log_file = str(logs_dir / f"group_{group.id}_scheme_{si}.log")
            profile_file = str(
                logs_dir / f"group_{group.id}_scheme_{si}.profile.json"
            )
            latency = benchmark_onnx_model(
                model_input, log_file,
                strip_shape_args=True, extra_run_args=subgraph_extra_args,
                export_profile_path=profile_file,
            )

            success = latency != float("inf")
            profile = _parse_profile_json(profile_file) if success else []
            compute_ms = _compute_time_from_profile(profile)

            results.append(SchemeResult(
                scheme=scheme, latency_ms=latency,
                success=success, compute_ms=compute_ms,
            ))

            if si == 0:
                baseline_lat = latency
                baseline_compute = compute_ms

            status = f"{latency:.3f}ms" if success else "FAIL"
            comp_str = f" compute={compute_ms:.4f}ms" if compute_ms > 0 else ""
            logger.info(
                f"    [{si}] {scheme.name}: {status}{comp_str}"
                f"  ({scheme.description})"
            )

        # Pick best scheme – prefer compute-layer time when available
        valid = [r for r in results if r.success]
        if valid:
            has_compute = baseline_compute > 0 and any(
                r.compute_ms > 0 for r in valid
            )
            if has_compute:
                best_result = min(
                    valid,
                    key=lambda r: r.compute_ms if r.compute_ms > 0 else float("inf"),
                )
            else:
                best_result = min(valid, key=lambda r: r.latency_ms)
            best_scheme = best_result.scheme
            best_lat = best_result.latency_ms
            best_compute = best_result.compute_ms
        else:
            best_scheme = None
            best_lat = float("inf")
            best_compute = 0.0

        gr = GroupResult(
            group=group,
            baseline_latency_ms=baseline_lat,
            best_scheme=best_scheme,
            best_latency_ms=best_lat,
            all_results=results,
            baseline_compute_ms=baseline_compute,
            best_compute_ms=best_compute,
        )
        group_results.append(gr)

        if best_scheme and best_scheme.name != "baseline":
            logger.info(
                f"  Best: {best_scheme.name} ({best_lat:.3f}ms, "
                f"speedup {gr.speedup:.3f}x, "
                f"compute_speedup {gr.compute_speedup:.3f}x)"
            )
        else:
            logger.info(f"  Best: baseline (no QDQ benefit)")

        # ── Persist Phase-2 cache after each group ──
        cached_p2.setdefault("results", [])
        cached_p2["results"].append(_group_result_to_cache_entry(gr))
        cache["phase2"] = cached_p2
        _save_cache(cache_path, cache)

    logger.info(
        f"Phase 2 complete: profiled {len(group_results)} groups "
        f"(cached {cached_hit}, skipped {skipped})"
    )

    # ─── Phase 3: Full-Model Validation ──────────────────────────────────

    MIN_SPEEDUP = 1.02

    logger.info("=" * 60)
    logger.info("Phase 3: Full-Model Validation")
    logger.info("=" * 60)
    logger.info(f"Minimum compute-speedup threshold: {MIN_SPEEDUP:.2f}x")

    # Measure FP16 baseline on full model
    logger.info("Measuring full-model FP16 baseline...")
    baseline_log = str(logs_dir / "full_baseline.log")
    full_baseline = benchmark_onnx_model(model_path, baseline_log)
    logger.info(f"Full-model FP16 baseline: {full_baseline:.3f}ms")

    # Filter and sort candidates by compute_speedup
    candidates = [
        gr for gr in group_results
        if gr.best_scheme is not None
        and gr.best_scheme.name != "baseline"
        and gr.best_scheme.target_tensors
        and gr.compute_speedup >= MIN_SPEEDUP
    ]
    candidates.sort(key=lambda g: g.compute_speedup, reverse=True)

    noise_filtered = sum(
        1 for gr in group_results
        if gr.best_scheme and gr.best_scheme.name != "baseline"
        and gr.best_scheme.target_tensors
        and 1.0 < gr.compute_speedup < MIN_SPEEDUP
    )
    if noise_filtered:
        logger.info(
            f"Filtered {noise_filtered} groups below "
            f"{MIN_SPEEDUP:.2f}x compute-speedup threshold"
        )
    logger.info(f"Candidates to apply: {len(candidates)}")

    # ── 3a: Build & save raw model (all candidates applied at once) ──

    logger.info("Building raw optimized model (all QDQ applied)...")
    raw_graph = gs.import_onnx(onnx.load(model_path))
    raw_applied = 0
    raw_qdqd: Set[str] = set()

    for gr in candidates:
        try:
            insert_qdq_on_graph(
                raw_graph, gr.best_scheme.target_tensors, quant_type,
                prefix=f"g{gr.group.id}",
                already_qdqd=raw_qdqd, do_cleanup=False,
            )
            raw_applied += 1
        except Exception as e:
            logger.warning(
                f"  Raw-model QDQ failed for group "
                f"'{gr.group.trt_layer_name}': {e}"
            )

    if raw_applied > 0:
        raw_graph.cleanup().toposort()

    raw_model_path = str(output_dir / "optimized_raw.onnx")
    onnx.save(gs.export_onnx(raw_graph), raw_model_path)
    logger.info(f"Saved raw optimized model ({raw_applied} groups): {raw_model_path}")

    raw_log = str(logs_dir / "full_raw.log")
    raw_latency = benchmark_onnx_model(raw_model_path, raw_log)
    if raw_latency != float("inf") and full_baseline > 0:
        logger.info(
            f"Raw model latency: {raw_latency:.3f}ms "
            f"(speedup {full_baseline / raw_latency:.3f}x)"
        )
    del raw_graph

    # ── 3b: Incremental validation (optional, default on) ──

    if not incremental_validation:
        final_model_path = raw_model_path
        final_latency = raw_latency
        applied_groups = raw_applied
        logger.info("Incremental validation disabled – using raw model as final")
    else:
        logger.info("=" * 60)
        logger.info("Phase 3b: Incremental Validation")
        logger.info("=" * 60)

        # Check for Phase-3 cache
        p3_validated: List[dict] = cached_p3.get("validated", [])
        p3_done_ids = {v["group_id"] for v in p3_validated}
        p3_kept_ids = {v["group_id"] for v in p3_validated if v["kept"]}
        p3_rejected_ids = {v["group_id"] for v in p3_validated if not v["kept"]}
        current_latency = cached_p3.get("current_latency_ms", full_baseline)

        if p3_done_ids:
            logger.info(
                f"Resuming incremental validation: "
                f"{len(p3_kept_ids)} kept, {len(p3_rejected_ids)} rejected, "
                f"current latency {current_latency:.3f}ms"
            )

        # Replay kept groups onto fresh graph to reconstruct state
        validated_graph = gs.import_onnx(onnx.load(model_path))
        already_qdqd_v: Set[str] = set()
        for gr in candidates:
            if gr.group.id in p3_kept_ids:
                try:
                    insert_qdq_on_graph(
                        validated_graph, gr.best_scheme.target_tensors,
                        quant_type, prefix=f"g{gr.group.id}",
                        already_qdqd=already_qdqd_v, do_cleanup=False,
                    )
                except Exception:
                    pass

        applied_groups = len(p3_kept_ids)
        rejected_groups = len(p3_rejected_ids)

        for ci, gr in enumerate(candidates):
            if gr.group.id in p3_done_ids:
                continue

            group_prefix = f"g{gr.group.id}"

            # Apply QDQ on a temporary copy to test
            test_bytes = gs.export_onnx(validated_graph).SerializeToString()
            test_model = onnx.load_from_string(test_bytes)
            test_graph = gs.import_onnx(test_model)
            test_qdqd = set(already_qdqd_v)

            try:
                insert_qdq_on_graph(
                    test_graph, gr.best_scheme.target_tensors, quant_type,
                    prefix=group_prefix,
                    already_qdqd=test_qdqd, do_cleanup=False,
                )
            except Exception as e:
                logger.warning(
                    f"  [{ci+1}/{len(candidates)}] SKIP group "
                    f"'{gr.group.trt_layer_name}': QDQ insertion failed: {e}"
                )
                p3_validated.append({
                    "group_id": gr.group.id, "kept": False,
                    "latency_ms": current_latency, "reason": str(e),
                })
                rejected_groups += 1
                _save_p3_cache(
                    cache, cache_path, full_baseline, candidates,
                    p3_validated, current_latency,
                )
                continue

            test_graph.cleanup().toposort()
            test_model_bytes = gs.export_onnx(test_graph).SerializeToString()

            val_log = str(logs_dir / f"incr_val_group_{gr.group.id}.log")
            new_latency = benchmark_onnx_model(
                test_model_bytes, val_log,
            )

            kept = (
                new_latency != float("inf")
                and new_latency < current_latency
            )

            if kept:
                validated_graph = test_graph
                already_qdqd_v = test_qdqd
                current_latency = new_latency
                applied_groups += 1
                logger.info(
                    f"  [{ci+1}/{len(candidates)}] KEPT "
                    f"'{gr.group.trt_layer_name}' "
                    f"({gr.best_scheme.name}, "
                    f"compute_speedup {gr.compute_speedup:.3f}x) "
                    f"-> full model {new_latency:.3f}ms"
                )
            else:
                rejected_groups += 1
                lat_str = (
                    f"{new_latency:.3f}ms" if new_latency != float("inf")
                    else "FAIL"
                )
                logger.info(
                    f"  [{ci+1}/{len(candidates)}] REJECTED "
                    f"'{gr.group.trt_layer_name}' "
                    f"({gr.best_scheme.name}) "
                    f"-> {lat_str} (was {current_latency:.3f}ms)"
                )

            p3_validated.append({
                "group_id": gr.group.id, "kept": kept,
                "latency_ms": float(new_latency),
            })
            _save_p3_cache(
                cache, cache_path, full_baseline, candidates,
                p3_validated, current_latency,
            )

        # Save final validated model
        if applied_groups > 0:
            validated_graph.cleanup().toposort()
        final_model_path = str(output_dir / "optimized_final.onnx")
        onnx.save(gs.export_onnx(validated_graph), final_model_path)
        logger.info(f"Saved validated model: {final_model_path}")

        final_log = str(logs_dir / "full_final.log")
        final_latency = benchmark_onnx_model(final_model_path, final_log)

        # Mark Phase-3 complete in cache
        cache.setdefault("phase3", {})["completed"] = True
        _save_cache(cache_path, cache)

    # ── Phase 4: Calibrated export via modelopt quantize() ──────────────

    calibrated_model_path = None
    if not skip_calibration and applied_groups > 0:
        logger.info("=" * 60)
        logger.info("Phase 4: Calibrated Export (modelopt quantize)")
        logger.info("=" * 60)

        all_target_tensors: List[str] = []
        if incremental_validation:
            for gr in candidates:
                if gr.group.id in p3_kept_ids:
                    all_target_tensors.extend(gr.best_scheme.target_tensors)
        else:
            for gr in candidates:
                all_target_tensors.extend(gr.best_scheme.target_tensors)

        all_target_tensors = list(dict.fromkeys(all_target_tensors))

        if all_target_tensors:
            calibrated_model_path = str(
                output_dir / "optimized_calibrated.onnx"
            )
            try:
                finalize_with_quantize(
                    model_path=model_path,
                    output_path=calibrated_model_path,
                    target_tensor_names=all_target_tensors,
                    quant_type=quant_type,
                    calibration_data=calibration_data,
                    calibration_method=calibration_method,
                    calibration_shapes=calibration_shapes,
                    calibration_eps=calibration_eps,
                    high_precision_dtype=high_precision_dtype,
                    use_external_data_format=use_external_data_format,
                )
                logger.info(
                    f"Calibrated model saved: {calibrated_model_path}"
                )
            except Exception as e:
                logger.error(f"Phase 4 calibration failed: {e}")
                calibrated_model_path = None
    elif skip_calibration:
        logger.info(
            "Phase 4 skipped (skip_calibration=True): "
            "final model uses placeholder Q/DQ (scale=0.1)"
        )

    elapsed = time.time() - t_start

    # ─── Summary ─────────────────────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
    logger.info(f"Groups profiled: {len(group_results)}")
    logger.info(f"Groups applied: {applied_groups}")
    logger.info(f"FP16 baseline: {full_baseline:.3f}ms")

    if raw_latency != float("inf") and full_baseline > 0:
        logger.info(
            f"Raw (all QDQ): {raw_latency:.3f}ms "
            f"(speedup {full_baseline / raw_latency:.3f}x) -> {raw_model_path}"
        )

    if final_latency != float("inf") and full_baseline > 0:
        speedup = full_baseline / final_latency
        logger.info(
            f"Final (placeholder QDQ): {final_latency:.3f}ms "
            f"(speedup {speedup:.3f}x) -> {final_model_path}"
        )
    else:
        logger.warning("Final TRT build failed")

    if calibrated_model_path:
        logger.info(
            f"Calibrated: {calibrated_model_path} "
            "(per-channel weights, bias excluded, real scale/zp)"
        )

    _log_per_group_report(group_results)

    return calibrated_model_path or final_model_path


def _save_p3_cache(
    cache: dict, cache_path: Path,
    full_baseline: float, candidates: List[GroupResult],
    validated: List[dict], current_latency: float,
):
    """Persist Phase-3 incremental-validation progress."""
    cache["phase3"] = {
        "full_baseline_ms": full_baseline,
        "candidates": [gr.group.id for gr in candidates],
        "validated": validated,
        "current_latency_ms": current_latency,
        "completed": False,
    }
    _save_cache(cache_path, cache)


def _log_per_group_report(results: List[GroupResult]):
    """Log a summary table of per-group profiling results."""
    if not results:
        return

    logger.info("")
    logger.info("Per-Group Results:")
    logger.info(
        f"{'Group':<50} {'Baseline':>10} {'Best':>10} "
        f"{'Speedup':>8} {'CompSpd':>8} {'Scheme':<20}"
    )
    logger.info("-" * 110)

    sorted_results = sorted(results, key=lambda r: r.compute_speedup, reverse=True)
    for gr in sorted_results:
        name = gr.group.trt_layer_name[:48]
        bl = f"{gr.baseline_latency_ms:.3f}" if gr.baseline_latency_ms != float("inf") else "N/A"
        bt = f"{gr.best_latency_ms:.3f}" if gr.best_latency_ms != float("inf") else "N/A"
        sp = f"{gr.speedup:.3f}x" if gr.speedup > 0 else "N/A"
        cs = f"{gr.compute_speedup:.3f}x" if gr.compute_speedup > 0 else "N/A"
        scheme_name = gr.best_scheme.name if gr.best_scheme else "none"
        logger.info(
            f"{name:<50} {bl:>10} {bt:>10} {sp:>8} {cs:>8} {scheme_name:<20}"
        )
