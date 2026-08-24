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

"""Core ONNX quantization sensitivity primitive: :func:`score`.

For every quantization target (an op type or a single node), inserts calibrated Q/DQ nodes on just
that target via the standard :func:`modelopt.onnx.quantization.quantize` entry point, runs the
resulting ONNX and the unquantized reference through ONNXRuntime on the same calibration inputs,
and computes a proxy metric between the two graph-output activation sets. Higher score means the
target degrades the model more if quantized -- so callers keep high scores at higher precision.
"""

from __future__ import annotations

import glob
import os
import re
import tempfile
import time
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
import onnx

from modelopt.onnx.logging_config import logger
from modelopt.onnx.op_types import (
    get_activation_ops,
    is_copy_op,
    is_default_quantizable_op_by_ort,
    is_fusible_reduction_op,
    is_normalization_op,
)
from modelopt.onnx.quantization.ort_utils import create_inference_session
from modelopt.onnx.quantization.quantize import quantize
from modelopt.onnx.quantization.sensitivity.metrics import cos_dist, kl_div, mse
from modelopt.onnx.utils import gen_random_inputs, get_input_names, get_op_types_in_graph

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = ["CalibrationSource", "Granularity", "Metric", "score"]


class Metric(str, Enum):
    """Proxy metrics between FP16 and quantized activations."""

    KL_DIV = "kl_div"
    MSE = "mse"
    COS = "cos"


class Granularity(str, Enum):
    """Enumeration granularity for sensitivity targets."""

    OP_TYPE = "op_type"
    NODE = "node"


class CalibrationSource(str, Enum):
    """Origin of the calibration data used for scoring."""

    REAL = "real"
    SYNTHETIC = "synthetic"


_METRIC_FUNCS: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    Metric.KL_DIV.value: kl_div,
    Metric.MSE.value: mse,
    Metric.COS.value: cos_dist,
}

# Fixed seed for the synthetic-random calibration fallback so that repeated invocations produce
# identical inputs and, therefore, comparable rankings within one machine.
_SYNTHETIC_SEED = 0


def _default_op_types_scope(onnx_model: onnx.ModelProto) -> set[str]:
    """Return op types worth probing by default: present in the graph AND known-quantizable.

    Intersects the set of op types actually present in the graph with the union of ORT's default
    quantizable ops, activation ops, normalization ops, and fusible reduction ops. Layout / copy
    ops (``Transpose`` / ``Reshape`` / ``Concat`` / ...) are then excluded via
    :func:`is_copy_op` -- they show up in ORT's default quantizable set but their sensitivity
    signal reflects Q/DQ insertion at data-movement boundaries rather than any INT8-kernel
    trade-off, and TensorRT never actually produces INT8 kernels for them, so ranking them
    clutters the output with "don't do this anyway" entries. Graph plumbing (``Cast`` /
    ``Constant`` / ``Shape`` / ...) not on any of the above lists is also skipped.

    Args:
        onnx_model: Loaded ONNX model to enumerate.

    Returns:
        Set of op-type strings to probe.
    """
    activation_ops = get_activation_ops()
    return {
        op
        for op in get_op_types_in_graph(onnx_model)
        if (
            is_default_quantizable_op_by_ort(op)
            or op in activation_ops
            or is_normalization_op(op)
            or is_fusible_reduction_op(op)
        )
        and not is_copy_op(op)
    }


def score(
    onnx_path: str,
    calibration_data: (
        Sequence[dict[str, np.ndarray]] | dict[str, np.ndarray] | np.ndarray | str | None
    ) = None,
    *,
    num_synthetic_samples: int = 100,
    target_precision: str = "int8",
    granularity: str = "op_type",
    metric: str = "kl_div",
    calibration_method: str = "entropy",
    calibration_eps: Sequence[str] = ("cuda:0", "cpu"),
    op_types_scope: Sequence[str] | None = None,
    work_dir: str | None = None,
) -> dict:
    """Rank quantization targets by their impact on model output.

    Runs one reference forward pass over calibration data on the unquantized ``onnx_path``, then for
    each target (op type or node) invokes :func:`modelopt.onnx.quantization.quantize` to insert
    calibrated Q/DQ nodes on just that target, re-runs the model, and computes ``metric`` between
    the reference and quantized graph outputs. Scores are summed across output tensors and averaged
    across the calibration samples inside each metric function; higher score means more accuracy
    loss if the target is quantized.

    Args:
        onnx_path: Path to the ONNX model to score. The model is treated as the FP-precision
            reference and is quantized once per target below.
        calibration_data: Calibration inputs. Accepts a ``dict[str, np.ndarray]`` (batch-first),
            a ``Sequence[dict[str, np.ndarray]]`` of single-sample dicts, a raw ``np.ndarray``
            (single-input models only), or a path to real data on disk (``.npy`` file, ``.npz``
            file, or directory of ``.npz`` files). Passing ``None`` falls back to synthetic random
            tensors of the ONNX's declared input shapes. Synthetic random calibration produces
            directional rankings only; see :class:`CalibrationSource` in the returned dict.
        num_synthetic_samples: Number of synthetic samples generated when
            ``calibration_data is None``. Ignored otherwise.
        target_precision: Quantization mode passed through to
            :func:`modelopt.onnx.quantization.quantize` for each per-target probe. Supported values
            are ``"int8"`` and ``"fp8"``.
        granularity: ``"op_type"`` scores each quantizable op type once (one probe per type);
            ``"node"`` scores each individual quantizable node (one probe per node), which is
            substantially more expensive but pinpoints single-node offenders.
        metric: One of :class:`Metric` values -- ``"kl_div"`` (default), ``"mse"``, or ``"cos"``.
        calibration_method: Passed through to :func:`modelopt.onnx.quantization.quantize` (defaults
            to ``"entropy"`` for int8/fp8).
        calibration_eps: ONNXRuntime execution providers to use for both the reference and the
            per-target forward passes, and for calibration inside :func:`quantize`. Same schema as
            the ``--calibration_eps`` CLI flag.
        op_types_scope: Optional whitelist of op types to probe. If omitted, defaults to the
            intersection of ops present in ``onnx_path`` and the union of ORT's default
            quantizable set, activation ops, normalization ops, and fusible reduction ops
            (see :func:`_default_op_types_scope`). Graph plumbing (``Cast`` / ``Constant`` /
            ``Shape`` / ...) is skipped by default because it produces zero-drift probes.
            Ops that slip past the filter but that the underlying
            :func:`modelopt.onnx.quantization.quantize` still cannot quantize are reported
            with score ``0.0`` -- the CLI hides those from the pretty-printed table by
            default but they always appear in the JSON output.
        work_dir: Directory to place intermediate per-target quantized ONNX files. Defaults to a
            fresh temporary directory that is removed after the call returns.

    Returns:
        A dict with keys:

        * ``scores``: mapping of ``op_type`` (op-type granularity) or ``node_name`` (node
          granularity) to the summed metric across graph outputs.
        * ``calibration_source``: ``"real"`` if the caller supplied calibration data, ``"synthetic"``
          when the primitive fell back to random tensors.
        * ``num_calibration_samples``: number of samples used for the scoring pass.
        * ``metric``: the metric name as passed in.
        * ``granularity``: ``"op_type"`` or ``"node"``.
        * ``target_precision``: the requested quantization precision.
    """
    if metric not in _METRIC_FUNCS:
        raise ValueError(
            f"Unknown metric '{metric}'. Expected one of {list(_METRIC_FUNCS.keys())}."
        )
    if granularity not in (Granularity.OP_TYPE.value, Granularity.NODE.value):
        raise ValueError(f"Unknown granularity '{granularity}'. Expected 'op_type' or 'node'.")
    if target_precision not in ("int8", "fp8"):
        raise ValueError(
            f"Unsupported target_precision '{target_precision}'. Expected 'int8' or 'fp8'."
        )

    onnx_model = onnx.load(onnx_path)
    calib_dict, calibration_source = _resolve_calibration_data(
        onnx_model, calibration_data, num_synthetic_samples
    )
    num_samples = _num_samples(calib_dict)
    logger.info(
        f"Sensitivity scan on {onnx_path}: {calibration_source.value} calibration, "
        f"{num_samples} samples, granularity={granularity}, metric={metric}, "
        f"target_precision={target_precision}"
    )

    quantizable_ops = set(op_types_scope) if op_types_scope else _default_op_types_scope(onnx_model)
    if granularity == Granularity.OP_TYPE.value:
        targets = _enumerate_op_type_targets(onnx_model, quantizable_ops)
    else:
        targets = _enumerate_node_targets(onnx_model, quantizable_ops)
    if not targets:
        logger.warning("No quantizable targets found under the requested scope.")

    metric_fn = _METRIC_FUNCS[metric]
    calibration_eps_list = list(calibration_eps)
    ref_outputs = _run_inference(onnx_path, calib_dict, calibration_eps_list)

    scores: dict[str, float] = {}
    use_tempdir = work_dir is None
    tmp_ctx = tempfile.TemporaryDirectory() if use_tempdir else None
    target_dir = tmp_ctx.name if tmp_ctx is not None else work_dir
    assert target_dir is not None
    try:
        os.makedirs(target_dir, exist_ok=True)
        wall_start = time.monotonic()
        for idx, (target_name, quantize_kwargs) in enumerate(targets, start=1):
            probe_path = os.path.join(
                target_dir, f"probe_{_sanitize_filename(target_name)}.quant.onnx"
            )
            step_start = time.monotonic()
            try:
                quantize(
                    onnx_path=onnx_path,
                    quantize_mode=target_precision,
                    calibration_data=calib_dict,
                    calibration_method=calibration_method,
                    calibration_eps=calibration_eps_list,
                    output_path=probe_path,
                    # Keep non-quantized ops at fp32 to avoid I/O dtype drift between the reference
                    # and quantized graphs -- the metric then reflects pure Q/DQ distortion.
                    high_precision_dtype="fp32",
                    keep_intermediate_files=False,
                    **quantize_kwargs,
                )
            except Exception as e:
                logger.warning(
                    f"[{idx}/{len(targets)}] quantize() failed for target '{target_name}': {e}"
                )
                continue
            quant_outputs = _run_inference(probe_path, calib_dict, calibration_eps_list)
            scores[target_name] = _pair_metric(ref_outputs, quant_outputs, metric_fn)
            logger.info(
                f"[{idx}/{len(targets)}] scored '{target_name}' = {scores[target_name]:.6g} "
                f"(step {time.monotonic() - step_start:.1f}s, total {time.monotonic() - wall_start:.1f}s)"
            )
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    return {
        "scores": scores,
        "calibration_source": calibration_source.value,
        "num_calibration_samples": num_samples,
        "metric": metric,
        "granularity": granularity,
        "target_precision": target_precision,
    }


def _resolve_calibration_data(
    onnx_model: onnx.ModelProto,
    calibration_data: (
        Sequence[dict[str, np.ndarray]] | dict[str, np.ndarray] | np.ndarray | str | None
    ),
    num_synthetic_samples: int,
) -> tuple[dict[str, np.ndarray], CalibrationSource]:
    """Normalize any accepted calibration input into a batch-first ``dict[str, ndarray]``.

    Args:
        onnx_model: Loaded ONNX model, used to resolve input names and shapes when the caller
            passes an ``ndarray`` (single-input models) or ``None`` (synthetic fallback).
        calibration_data: One of the forms documented on :func:`score`.
        num_synthetic_samples: Number of synthetic samples to generate when ``calibration_data`` is
            ``None``.

    Returns:
        A tuple ``(calib_dict, source)`` where ``calib_dict`` has each input as a batch-first
        numpy array and ``source`` is either ``CalibrationSource.REAL`` or
        ``CalibrationSource.SYNTHETIC``.
    """
    input_names = get_input_names(onnx_model)
    if calibration_data is None:
        # np.random is used inside gen_random_inputs; reseed here so the fallback is deterministic
        # across invocations on the same model.
        np.random.seed(_SYNTHETIC_SEED)
        samples = [gen_random_inputs(onnx_model) for _ in range(num_synthetic_samples)]
        return _stack_sample_list(samples), CalibrationSource.SYNTHETIC
    if isinstance(calibration_data, str):
        return _load_calibration_from_path(calibration_data, input_names), CalibrationSource.REAL
    if isinstance(calibration_data, np.ndarray):
        assert len(input_names) == 1, (
            "ndarray calibration_data is only valid for single-input models."
        )
        return {input_names[0]: calibration_data}, CalibrationSource.REAL
    if isinstance(calibration_data, dict):
        return {k: np.asarray(v) for k, v in calibration_data.items()}, CalibrationSource.REAL
    # Sequence[dict]
    return _stack_sample_list(list(calibration_data)), CalibrationSource.REAL


def _load_calibration_from_path(path: str, input_names: list[str]) -> dict[str, np.ndarray]:
    """Load real calibration data from ``.npy``, ``.npz``, or a directory of ``.npz`` files.

    Args:
        path: Filesystem location.
        input_names: ONNX input names, used to attach ``.npy`` arrays to the sole input.

    Returns:
        Batch-first ``dict[str, ndarray]``.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.npz")))
        assert files, f"No .npz files found under directory {path}"
        parts: dict[str, list[np.ndarray]] = {}
        for f in files:
            payload = np.load(f, allow_pickle=False)
            for key in payload.files:
                parts.setdefault(key, []).append(payload[key])
        return {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    if path.endswith(".npz"):
        payload = np.load(path, allow_pickle=False)
        return {key: payload[key] for key in payload.files}
    if path.endswith(".npy"):
        arr = np.load(path, allow_pickle=False)
        assert len(input_names) == 1, (
            f"{path} is a single-tensor .npy but the model has {len(input_names)} inputs."
        )
        return {input_names[0]: arr}
    raise ValueError(f"Unsupported calibration_data path: {path}")


def _stack_sample_list(samples: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate a sequence of single-sample dicts into one batch-first dict."""
    assert samples, "Empty calibration sample sequence."
    keys = list(samples[0].keys())
    return {k: np.concatenate([np.asarray(s[k]) for s in samples], axis=0) for k in keys}


def _num_samples(calib_dict: dict[str, np.ndarray]) -> int:
    """Return the batch-axis length of the first array in ``calib_dict``."""
    first = next(iter(calib_dict.values()))
    return int(first.shape[0])


def _enumerate_op_type_targets(
    onnx_model: onnx.ModelProto, quantizable_ops: set[str]
) -> list[tuple[str, dict]]:
    """Return one probe per op type present in the model and in ``quantizable_ops``.

    Args:
        onnx_model: Loaded model to enumerate.
        quantizable_ops: Whitelist of op types considered quantizable.

    Returns:
        List of ``(op_type, quantize_kwargs)`` pairs where ``quantize_kwargs`` restricts
        :func:`quantize` to that op type only.
    """
    present = {node.op_type for node in onnx_model.graph.node}
    scoped = sorted(present & quantizable_ops)
    return [(op, {"op_types_to_quantize": [op]}) for op in scoped]


def _enumerate_node_targets(
    onnx_model: onnx.ModelProto, quantizable_ops: set[str]
) -> list[tuple[str, dict]]:
    """Return one probe per named quantizable node.

    Args:
        onnx_model: Loaded model to enumerate.
        quantizable_ops: Whitelist of op types considered quantizable.

    Returns:
        List of ``(node_name, quantize_kwargs)`` pairs where ``quantize_kwargs`` restricts
        :func:`quantize` to a regex matching that node only.
    """
    targets: list[tuple[str, dict]] = []
    for node in onnx_model.graph.node:
        if node.op_type not in quantizable_ops or not node.name:
            continue
        regex = f"^{re.escape(node.name)}$"
        targets.append((node.name, {"nodes_to_quantize": [regex]}))
    return targets


def _run_inference(
    onnx_path: str, calib_dict: dict[str, np.ndarray], calibration_eps: list[str]
) -> list[np.ndarray]:
    """Run every sample through ORT and stack outputs along the batch axis.

    Args:
        onnx_path: ONNX file to load into an ORT ``InferenceSession``.
        calib_dict: Batch-first input dict.
        calibration_eps: ORT execution providers, same schema as
            :func:`quantize`'s ``calibration_eps``.

    Returns:
        List of numpy arrays, one per graph output, each shaped ``(num_samples, ...)``.
    """
    session = create_inference_session(onnx_path, calibration_eps)
    num_output = len(session.get_outputs())
    num_samples = _num_samples(calib_dict)
    per_output: list[list[np.ndarray]] = [[] for _ in range(num_output)]
    for i in range(num_samples):
        feed = {name: arr[i : i + 1] for name, arr in calib_dict.items()}
        outputs = session.run(None, feed)
        for j, out in enumerate(outputs):
            per_output[j].append(np.asarray(out))
    return [np.concatenate(chunks, axis=0) for chunks in per_output]


def _pair_metric(
    ref_outputs: list[np.ndarray],
    quant_outputs: list[np.ndarray],
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
) -> float:
    """Sum the metric across matched graph outputs of the reference and quantized models."""
    return float(sum(metric_fn(ref, quant) for ref, quant in zip(ref_outputs, quant_outputs)))


def _sanitize_filename(name: str) -> str:
    """Turn an arbitrary op/node name into a filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80] or "unnamed"
