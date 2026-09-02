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

"""ONNX Q/DQ Autotuning Workflows.

This module provides high-level workflow functions for automated Q/DQ (Quantization/Dequantization)
optimization of ONNX models using pattern-based region analysis and TensorRT performance measurement.
"""

import dataclasses
import fnmatch
import hashlib
import json
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

import onnx

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.benchmark import TensorRTPyBenchmark, TrtExecBenchmark
from modelopt.onnx.quantization.autotune.common import (
    Config,
    PatternCache,
    SchemeAction,
    is_valid_latency,
)
from modelopt.onnx.quantization.ort_utils import _run_trtexec
from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

_benchmark_instance = None


def _meets_performance_threshold(
    reference_latency_ms: float, candidate_latency_ms: float, threshold: float
) -> bool:
    return (
        is_valid_latency(reference_latency_ms)
        and is_valid_latency(candidate_latency_ms)
        and reference_latency_ms / candidate_latency_ms >= threshold
    )


def _require_valid_latency(latency_ms: float, measurement: str) -> None:
    if not is_valid_latency(latency_ms):
        raise RuntimeError(f"Unable to measure a valid {measurement} latency")


def _get_model_artifact_info(model_path: Path) -> tuple[str, int]:
    model = onnx.load(model_path, load_external_data=True)
    model_sha256 = hashlib.sha256(model.SerializeToString(deterministic=True)).hexdigest()
    qdq_count = sum(node.op_type == "QuantizeLinear" for node in model.graph.node)
    return model_sha256, qdq_count


def _get_file_identity(path: str) -> dict[str, str | int | bool | None]:
    resolved_path = Path(path).resolve()
    identity: dict[str, str | int | bool | None] = {
        "name_sha256": hashlib.sha256(resolved_path.name.encode()).hexdigest(),
        "path_sha256": hashlib.sha256(str(resolved_path).encode()).hexdigest(),
        "exists": resolved_path.is_file(),
    }
    if not resolved_path.is_file():
        identity["sha256"] = None
        return identity
    digest = hashlib.sha256()
    with resolved_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    identity["size"] = resolved_path.stat().st_size
    identity["sha256"] = digest.hexdigest()
    return identity


def _get_path_identity(path: str) -> dict[str, str]:
    resolved_path = Path(path).resolve()
    return {
        "name_sha256": hashlib.sha256(resolved_path.name.encode()).hexdigest(),
        "path_sha256": hashlib.sha256(str(resolved_path).encode()).hexdigest(),
    }


def _normalize_fingerprint_value(value):
    if isinstance(value, dict):
        return {str(key): _normalize_fingerprint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_fingerprint_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _get_value_identity(value) -> dict[str, int | str]:
    normalized = _normalize_fingerprint_value(value)
    serialized = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    item_count = len(value) if isinstance(value, (dict, list, tuple)) else int(value is not None)
    return {"sha256": hashlib.sha256(serialized.encode()).hexdigest(), "item_count": item_count}


def _get_trtexec_identity() -> dict[str, object]:
    executable = shutil.which("trtexec")
    if executable is None:
        return {"available": False}
    executable = str(Path(executable).resolve())

    identity: dict[str, object] = {
        "available": True,
        "binary": _get_file_identity(executable),
    }
    try:
        result = _run_trtexec(timeout=10)
        version_output = "\n".join((result.stdout, result.stderr)).strip()
        match = re.search(r"TensorRT(?:\.trtexec)?[^\n]*?v([0-9][0-9.]*)", version_output)
        identity.update(
            {
                "version": match.group(1) if match else None,
                "version_output_sha256": hashlib.sha256(version_output.encode()).hexdigest(),
                "version_returncode": result.returncode,
            }
        )
    except Exception:
        identity["version"] = None
    return identity


def _get_benchmark_fingerprint() -> dict:
    if _benchmark_instance is None:
        return {"backend": None}
    benchmark_module = sys.modules.get(type(_benchmark_instance).__module__)
    trt_module = getattr(benchmark_module, "trt", None)
    torch_module = getattr(benchmark_module, "torch", None)
    cuda_device = None
    if torch_module is not None and torch_module.cuda.is_available():
        device_index = torch_module.cuda.current_device()
        properties = torch_module.cuda.get_device_properties(device_index)
        cuda_device = {
            "index": device_index,
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "uuid_sha256": hashlib.sha256(
                str(getattr(properties, "uuid", "")).encode()
            ).hexdigest(),
        }
    fingerprint = {
        "backend": type(_benchmark_instance).__name__,
        "tensorrt_version": getattr(trt_module, "__version__", None),
        "cuda_device": cuda_device,
        "timing_cache": _get_path_identity(_benchmark_instance.timing_cache_file),
        "warmup_runs": _benchmark_instance.warmup_runs,
        "timing_runs": _benchmark_instance.timing_runs,
        "plugin_libraries": [
            _get_file_identity(path) for path in _benchmark_instance.plugin_libraries
        ],
        "trtexec_args": _get_value_identity(getattr(_benchmark_instance, "trtexec_args", None)),
        "shape_configs": _get_value_identity(getattr(_benchmark_instance, "_shape_configs", None)),
    }
    if isinstance(_benchmark_instance, TrtExecBenchmark):
        fingerprint["trtexec"] = _get_trtexec_identity()
    return fingerprint


def _count_profile_measurements(autotuner: QDQAutotuner) -> int:
    pattern_schemes = list(autotuner.profiled_patterns)
    current_pattern = getattr(autotuner, "current_profile_pattern_schemes", None)
    if current_pattern is not None and all(current_pattern is not item for item in pattern_schemes):
        pattern_schemes.append(current_pattern)
    return sum(scheme.is_profiled for pattern in pattern_schemes for scheme in pattern.schemes)


def _reuse_proxy_decision(
    autotuner: QDQAutotuner,
    final_model_path: Path,
    baseline_latency: float,
    candidate_model_sha256: str,
    candidate_qdq_count: int,
    model_transform: Callable[[onnx.ModelProto], onnx.ModelProto] | None,
) -> bool:
    decision = autotuner.proxy_decision
    if not isinstance(decision, dict):
        return False

    proxy_selection = decision.get("proxy_selection")
    if (
        proxy_selection not in {SchemeAction.QDQ.value, SchemeAction.NO_QDQ.value}
        or decision.get("baseline_latency_ms") != baseline_latency
        or decision.get("candidate_model_sha256") != candidate_model_sha256
        or decision.get("candidate_quantization_site_count") != candidate_qdq_count
    ):
        logger.info("Saved Autotune proxy decision does not match the current candidate")
        return False

    keep_qdq = proxy_selection == SchemeAction.QDQ.value
    autotuner.set_force_no_qdq(not keep_qdq)
    autotuner.export_onnx(
        str(final_model_path),
        insert_qdq=keep_qdq,
        model_transform=model_transform,
    )
    selected_model_sha256, _ = _get_model_artifact_info(final_model_path)
    if decision.get("selected_model_sha256") == selected_model_sha256:
        logger.info("Reused the validated Autotune proxy decision from the checkpoint")
        return True

    logger.info("Saved Autotune proxy selection does not match the current artifact")
    autotuner.set_force_no_qdq(False)
    autotuner.export_onnx(str(final_model_path), insert_qdq=True, model_transform=model_transform)
    return False


def benchmark_onnx_model(
    model_path: str | bytes, log_file: str | None = None, flush_timing_cache: bool = False
) -> float:
    """Benchmark ONNX model inference latency using TensorRT Python API.

    Args:
        model_path: Path to ONNX model file, or bytes containing serialized model protobuf
        log_file: Optional path to save detailed TensorRT build and benchmark logs
                 (default: None, no logging)
        flush_timing_cache: If True, flushes TensorRT timing cache before building engine.
                           Useful for periodic cache refresh (default: False)

    Returns:
        Measured median inference latency in milliseconds.
        Returns float('inf') on failure (invalid model, build error, etc.)

    Raises:
        No exceptions raised - errors are caught and logged, returning float('inf')
    """
    global _benchmark_instance

    if _benchmark_instance is None:
        logger.error("Benchmark instance not initialized")
        return float("inf")

    try:
        latency = _benchmark_instance.run(
            model_path, log_file=log_file, flush_timing_cache=flush_timing_cache
        )

        if latency == float("inf"):
            if isinstance(model_path, bytes):
                logger.warning("Benchmark failed for model bytes")
            else:
                logger.warning(f"Benchmark failed: {model_path}")
            return float("inf")

        logger.debug(f"Benchmark result: {latency:.2f} ms")
        return latency

    except Exception as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        return float("inf")


def init_benchmark_instance(
    use_trtexec: bool = False,
    plugin_libraries: list[str] | None = None,
    timing_cache_file: str | None = None,
    warmup_runs: int = 5,
    timing_runs: int = 20,
    trtexec_args: list[str] | None = None,
):
    """Initialize global TensorRT benchmark instance for model performance measurement.

    Args:
        use_trtexec: Whether to use trtexec for benchmarking.
        plugin_libraries: List of paths to TensorRT plugin shared libraries (.so files).
                          These plugins will be loaded by trtexec or TensorRT Python API during engine building.
                          If None, no custom plugins are loaded.
        timing_cache_file: Path to TensorRT timing cache file for faster engine builds.
                          If None, uses default "trtexec_timing.cache" (default: None)
        warmup_runs: Number of warmup inference iterations before measurement.
                    Allows GPU to reach stable performance state (default: 5)
        timing_runs: Number of timed inference iterations for latency measurement.
                    Higher values give more stable median (default: 20)
        trtexec_args: Additional command-line arguments to pass to trtexec as a string (only used if use_trtexec=True).
                     Example: '--fp16 --workspace=4096 --verbose'
    """
    global _benchmark_instance
    try:
        if use_trtexec:
            _benchmark_instance = TrtExecBenchmark(
                timing_cache_file=timing_cache_file,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                plugin_libraries=plugin_libraries,
                trtexec_args=trtexec_args,
            )
            logger.info("Trtexec benchmark initialized")
        else:
            _benchmark_instance = TensorRTPyBenchmark(
                timing_cache_file=timing_cache_file,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                plugin_libraries=plugin_libraries,
            )
            logger.info("TensorRT Python API benchmark initialized")
        logger.debug(
            f"Settings: warmup={warmup_runs}, timing={timing_runs}, "
            f"cache={timing_cache_file or 'trtexec_timing.cache'}, plugin_libraries={plugin_libraries}"
        )
        return _benchmark_instance
    except Exception as e:
        logger.error(f"TensorRT initialization failed: {e}", exc_info=True)
        return None


def _region_matches_filter(region, graph, filter_patterns: list[str]) -> bool:
    """Check if any node in the region matches any of the filter patterns.

    Args:
        region: Region object to check
        graph: ONNX graph (graphsurgeon) containing node information
        filter_patterns: List of wildcard patterns to match against node names

    Returns:
        True if at least one node in the region matches any pattern, False otherwise
    """
    if not filter_patterns:
        return True

    node_indices = region.get_all_nodes_recursive()

    for node_idx in node_indices:
        if node_idx < len(graph.nodes):
            node_name = graph.nodes[node_idx].name
            for pattern in filter_patterns:
                if fnmatch.fnmatch(node_name, pattern):
                    return True

    return False


def region_pattern_autotuning_workflow(
    model_or_path: str | onnx.ModelProto,
    output_dir: Path | None = None,
    num_schemes_per_region: int = 30,
    pattern_cache_file: str | None = None,
    state_file: str | None = None,
    quant_type: str = "int8",
    default_dq_dtype: str = "float32",
    qdq_baseline_model: str | None = None,
    node_filter_list: list[str] | None = None,
    verbose: bool = False,
    model_transform: Callable[[onnx.ModelProto], onnx.ModelProto] | None = None,
    resume_fingerprint: dict | None = None,
) -> QDQAutotuner:
    """Run automated Q/DQ (Quantization/Dequantization) optimization on an ONNX model.

    This workflow uses pattern-based region optimization to efficiently find optimal
    Q/DQ insertion points. The key insight: regions with identical structural patterns
    can share the same Q/DQ scheme. When a best scheme is found for a pattern, it
    automatically applies to all regions matching that pattern, making optimization
    both efficient and consistent.

    Automatically discovers regions, generates and tests Q/DQ insertion schemes,
    and exports optimized model. Supports incremental state saving for crash recovery
    and pattern cache-based warm-start.

    **Workflow Steps:**
    1. Load model and initialize autotuner with automatic hierarchical region discovery
    2. Resume from checkpoint if state file exists (crash recovery)
    3. Load pattern cache if provided (warm-start with known-good schemes)
    4. Import Q/DQ patterns from baseline model if provided (transfer learning)
    5. Measure baseline performance without Q/DQ insertions
    6. For each discovered region pattern:
       a. Generate Q/DQ insertion schemes (pattern-relative)
       b. Build TensorRT engine and measure latency for each scheme
       c. Select best scheme for this pattern (applies to all matching regions)
       d. Save checkpoint and intermediate model
    7. Export final optimized model with best Q/DQ scheme for each pattern

    Args:
        model_or_path: Path to ONNX model file to optimize
        output_dir: Directory for output files (state, logs, models). Created if it doesn't exist.
        num_schemes_per_region: Number of Q/DQ insertion schemes to test per region pattern.
                               Higher values explore more configurations but take longer (default: 30)
        pattern_cache_file: Optional path to pattern cache YAML file containing known-good schemes
                           from previous runs. Enables warm-start optimization (default: None)
        state_file: Optional path to state file for checkpoint/resume. If None, automatically
                   uses <output_dir>/autotuner_state.yaml (default: None)
        quant_type: Quantization data type - "int8" for INT8 quantization (default),
                   "fp8" for FP8 quantization
        default_dq_dtype: Dtype for DequantizeLinear output; "float32" (default), "float16", or "bfloat16".
        qdq_baseline_model: Optional path to a pre-quantized ONNX model. If provided,
                           extracts Q/DQ insertion patterns and adds them to pattern cache
                           for warm-start (default: None)
        node_filter_list: Optional list of wildcard patterns to filter ONNX nodes. Regions
                         without any matching nodes are skipped during autotuning (default: None)
        verbose: Enable verbose logging in Config for detailed autotuner output (default: False)
        model_transform: Optional transformation applied to each benchmark model after Q/DQ
                         insertion and before FP8 conversion.
        resume_fingerprint: Additional runtime-precision and environment options used to validate
                            resumed measurements.

    Returns:
        QDQAutotuner instance after autotuning
    """
    if num_schemes_per_region < 1:
        raise ValueError("num_schemes_per_region must be at least 1")

    output_dir_is_temp = output_dir is None
    if not output_dir:
        output_dir = Path(tempfile.mkdtemp())

    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "region_models"
    models_dir.mkdir(exist_ok=True)

    if state_file is None:
        state_file = str(output_dir / "autotuner_state.yaml")
    state_path = Path(state_file)

    if isinstance(model_or_path, str):
        logger.info(f"Loading model: {model_or_path}")
        model = onnx.load(model_or_path)
    else:
        model = model_or_path

    pattern_cache = None
    if pattern_cache_file:
        pattern_cache_path = Path(pattern_cache_file)
        if pattern_cache_path.exists():
            pattern_cache = PatternCache.load(str(pattern_cache_path))
            logger.info(
                f"Loaded pattern cache: {pattern_cache.num_patterns} patterns, "
                f"{pattern_cache.total_schemes} schemes"
            )
        else:
            logger.warning(f"Pattern cache not found: {pattern_cache_file}")

    logger.info(
        f"Initializing autotuner (quant_type={quant_type}, default_dq_dtype={default_dq_dtype})"
    )
    config = Config(
        default_quant_type=quant_type,
        default_dq_dtype=default_dq_dtype,
        verbose=verbose,
    )

    autotuner = QDQAutotuner(model)
    autotuner.initialize(config, pattern_cache)

    fingerprint = {
        "config": _normalize_fingerprint_value(dataclasses.asdict(config)),
        "search": {
            "num_schemes_per_region": num_schemes_per_region,
            "node_filter_list": _get_value_identity(node_filter_list),
            "pattern_cache": (
                _get_file_identity(pattern_cache_file) if pattern_cache_file else None
            ),
            "qdq_baseline": (
                _get_file_identity(qdq_baseline_model) if qdq_baseline_model else None
            ),
        },
        "caller_options": _get_value_identity(resume_fingerprint),
        "benchmark": _get_benchmark_fingerprint(),
    }
    autotuner.set_resume_fingerprint(**fingerprint)

    if state_path.exists():
        logger.info(f"Resuming from checkpoint: {state_path}")
        autotuner.load_state(str(state_path))
    else:
        logger.info("Starting new autotuning session")

    if qdq_baseline_model:
        qdq_baseline_path = Path(qdq_baseline_model)
        if qdq_baseline_path.exists():
            logger.info(f"Importing patterns from QDQ baseline: {qdq_baseline_model}")
            qdq_model = onnx.load(str(qdq_baseline_path))
            quantized_tensors = get_quantized_tensors(qdq_model)
            logger.debug(f"Found {len(quantized_tensors)} quantized tensors in baseline")
            autotuner.import_insertion_points(quantized_tensors)
            logger.info("Pattern import complete")
        else:
            logger.warning(f"QDQ baseline not found: {qdq_baseline_model}")

    regions = autotuner.regions
    logger.info(f"Ready to profile {len(regions)} regions")

    if autotuner.baseline_latency_ms is None:
        logger.info("Measuring baseline (no Q/DQ)")
        baseline_path = output_dir / "baseline.onnx"
        autotuner.export_onnx(str(baseline_path), insert_qdq=False, model_transform=model_transform)
        baseline_log = logs_dir / "baseline.log"
        baseline_latency = benchmark_onnx_model(str(baseline_path), str(baseline_log))
        _require_valid_latency(baseline_latency, "baseline")
        autotuner.submit(baseline_latency)
        autotuner.save_state(str(state_path))
        logger.info(f"Baseline: {baseline_latency:.2f} ms")
    else:
        baseline_latency = autotuner.baseline_latency_ms
        _require_valid_latency(baseline_latency, "baseline")
        logger.info(f"Using baseline from checkpoint: {baseline_latency:.2f} ms")

    logger.info(
        f"Starting region profiling (incumbent plus {num_schemes_per_region} overrides per region)"
    )

    profile_measurement_count = _count_profile_measurements(autotuner)
    resumed_region_id = (
        autotuner.current_profile_region.id
        if autotuner.current_profile_region is not None
        else None
    )
    resumed_region_reached = resumed_region_id is None

    for region_idx, region in enumerate(regions):
        logger.info(
            f"Region {region_idx + 1}/{len(regions)} (ID={region.id}, level={region.level})"
        )

        if node_filter_list and not _region_matches_filter(
            region, autotuner.graph, node_filter_list
        ):
            logger.info("  Skipping (no nodes match filter patterns)")
            continue

        if not resumed_region_reached:
            if region.id != resumed_region_id:
                logger.info("  Skipping (completed before resumed region)")
                continue
            resumed_region_reached = True

        commit = region_idx > 0 and region.id != resumed_region_id
        autotuner.set_profile_region(region, commit=commit)

        if autotuner.current_profile_pattern_schemes is None:
            logger.info("  Skipping (already profiled)")
            continue

        inherit_scheme_idx = autotuner.begin_inherit_profile()
        if inherit_scheme_idx >= 0:
            inherit_model_bytes = autotuner.export_onnx(
                None, insert_qdq=True, model_transform=model_transform
            )
            inherit_log = logs_dir / f"region_{region.id}_inherit.log"
            profile_measurement_count += 1
            inherit_latency = benchmark_onnx_model(inherit_model_bytes, str(inherit_log))
            autotuner.submit_inherit(inherit_latency, success=is_valid_latency(inherit_latency))
            autotuner.save_state(str(state_path))

        ps = autotuner.current_profile_pattern_schemes
        remaining_override_budget = max(0, num_schemes_per_region - ps.profiled_override_count)
        schemes_tested = 0
        for scheme_num in range(remaining_override_budget):
            scheme_idx = autotuner.generate()

            if scheme_idx == -1:
                autotuner.save_state(str(state_path))
                logger.debug(f"  Stopping at scheme {scheme_num + 1} (no more unique schemes)")
                break

            schemes_tested += 1
            model_bytes = autotuner.export_onnx(
                None, insert_qdq=True, model_transform=model_transform
            )
            test_log = logs_dir / f"region_{region.id}_scheme_{scheme_idx}.log"
            profile_measurement_count += 1
            flush_timing_cache = (profile_measurement_count % 10) == 0
            latency = benchmark_onnx_model(
                model_bytes, str(test_log), flush_timing_cache=flush_timing_cache
            )

            autotuner.submit(latency, success=is_valid_latency(latency))
            autotuner.save_state(str(state_path))

        if ps is not None:
            ps.select_best(autotuner.config.performance_threshold)
        if ps and ps.schemes:
            best_scheme = ps.selected_scheme
            if best_scheme and best_scheme.latency_ms < float("inf") and baseline_latency > 0:
                speedup = baseline_latency / best_scheme.latency_ms
                logger.info(
                    f"  Tested {schemes_tested} overrides: "
                    f"best {best_scheme.latency_ms:.2f} ms ({speedup:.3f}x speedup)"
                )
            else:
                logger.info(f"  Tested {schemes_tested} overrides: no valid measurements")
        else:
            logger.info(f"  Tested {schemes_tested} overrides")

        region_model_path = models_dir / f"region_{region.id}_level_{region.level}.onnx"
        autotuner.export_onnx(
            str(region_model_path),
            insert_qdq=True,
            best=True,
            model_transform=model_transform,
        )
        logger.debug(f"  Saved best model: {region_model_path.name}")

        # Save state after each region (incremental, crash recovery)
        autotuner.save_state(str(state_path))
        logger.debug("  Checkpoint saved")

    # Commit final region
    autotuner.set_profile_region(None, commit=True)

    logger.info("Exporting final optimized model")
    final_model_path = output_dir / "optimized_final.onnx"
    autotuner.set_force_no_qdq(False)
    autotuner.export_onnx(str(final_model_path), insert_qdq=True, model_transform=model_transform)
    candidate_model_sha256, candidate_qdq_count = _get_model_artifact_info(final_model_path)

    reused_proxy_decision = _reuse_proxy_decision(
        autotuner,
        final_model_path,
        baseline_latency,
        candidate_model_sha256,
        candidate_qdq_count,
        model_transform,
    )
    if not reused_proxy_decision:
        final_log = logs_dir / "final.log"
        final_latency = benchmark_onnx_model(str(final_model_path), str(final_log))

        if candidate_qdq_count > 0 and _meets_performance_threshold(
            baseline_latency, final_latency, autotuner.config.performance_threshold
        ):
            proxy_selection = SchemeAction.QDQ.value
            speedup = baseline_latency / final_latency
            logger.info(
                f"Autotune proxy retained Q/DQ placement for calibrated evaluation: "
                f"{speedup:.3f}x speedup (required {autotuner.config.performance_threshold:.3f}x)"
            )
        else:
            autotuner.set_force_no_qdq()
            autotuner.export_onnx(
                str(final_model_path), insert_qdq=False, model_transform=model_transform
            )
            proxy_selection = SchemeAction.NO_QDQ.value
            candidate_outcome = (
                f"{baseline_latency / final_latency:.3f}x speedup"
                if is_valid_latency(final_latency)
                else "an invalid latency"
            )
            logger.warning(
                f"Autotune proxy rejected Q/DQ placement after measuring {candidate_outcome}; "
                f"the required speedup is {autotuner.config.performance_threshold:.3f}x. "
                "The calibrated candidate will use the high-precision no-Q/DQ placement."
            )

        selected_model_sha256, _ = _get_model_artifact_info(final_model_path)
        autotuner.record_proxy_decision(
            proxy_selection=proxy_selection,
            baseline_latency_ms=baseline_latency,
            candidate_latency_ms=final_latency if is_valid_latency(final_latency) else None,
            candidate_quantization_site_count=candidate_qdq_count,
            candidate_model_sha256=candidate_model_sha256,
            selected_model_sha256=selected_model_sha256,
        )
    autotuner.save_state(str(state_path))

    logger.info("Autotuning complete")
    logger.info(f"  Final model: {final_model_path}")
    logger.info(f"  State: {state_path}")
    logger.debug(f"  Logs: {logs_dir}")
    logger.debug(f"  Region models: {models_dir}")

    # Remove temporary folder
    if output_dir_is_temp and output_dir.exists():
        shutil.rmtree(output_dir)
        logger.info(f"Temporary directory {output_dir} was deleted!")

    return autotuner
