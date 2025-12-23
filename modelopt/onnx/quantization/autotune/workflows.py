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

"""ONNX Q/DQ Autotuning Workflows.

This module provides high-level workflow functions for automated Q/DQ (Quantization/Dequantization)
optimization of ONNX models using pattern-based region analysis and TensorRT performance measurement.

**Core Capabilities:**

1. **Automated Region Discovery**: Discovers hierarchical regions in the computation graph
   - LEAF regions: Contain actual graph nodes
   - COMPOSITE regions: Contain child regions with hierarchical structure

2. **Pattern-Based Optimization**: Groups regions by structural pattern
   - Regions with identical patterns share optimization schemes
   - One optimization applies to all matching regions simultaneously

3. **TensorRT Benchmarking**: Measures actual inference performance
   - Builds TensorRT engines for each Q/DQ configuration
   - Measures median latency across multiple runs
   - Caches timing data for faster iteration

4. **Incremental State Management**: Supports crash recovery and resume
   - Saves state after each region profiling
   - Resumes from last checkpoint automatically
   - Preserves baseline and all measurements

5. **Pattern Cache Warm-Start**: Leverages previous optimization results
   - Loads known-good schemes from cache
   - Reduces exploration time for similar models
   - Transfers learned patterns across runs

**Key Functions:**

- **benchmark_onnx_model()**: Benchmark ONNX model inference latency using TensorRT
- **init_benchmark_instance()**: Initialize global TensorRT benchmark instance
- **region_pattern_autotuning_workflow()**: Complete end-to-end Q/DQ optimization workflow

**Workflow Overview:**

1. Initialize autotuner with automatic region discovery
2. Measure baseline performance (no Q/DQ)
3. For each region pattern:
   - Generate Q/DQ insertion schemes
   - Benchmark each scheme with TensorRT
   - Select best scheme for pattern
   - Apply to all regions with matching pattern
4. Export final optimized model

**Performance Optimization:**

- Pattern-based approach reduces redundant evaluation
- TensorRT timing cache speeds up engine builds
- Incremental state saves enable long-running optimizations
- Pattern cache enables cross-model learning
"""

import fnmatch
import logging
from pathlib import Path

import onnx

from modelopt.onnx.quantization.autotune.autotuner import QDQAutotuner
from modelopt.onnx.quantization.autotune.benchmark import TensorRTPyBenchmark, TrtExecBenchmark
from modelopt.onnx.quantization.autotune.common import Config, PatternCache
from modelopt.onnx.quantization.qdq_utils import get_quantized_tensors

logger = logging.getLogger(__name__)

# Global benchmark instance - will be initialized with timing cache
_benchmark_instance = None


# =============================================================================
# Benchmarking
# =============================================================================


def benchmark_onnx_model(
    model_path: str | bytes, log_file: str | None = None, flush_timing_cache: bool = False
) -> float:
    """Benchmark ONNX model inference latency using TensorRT Python API.

    Uses the global TensorRTPyBenchmark instance to build a TensorRT engine
    and measure inference latency. The benchmark instance persists across calls
    for efficiency (reuses Builder, Runtime, Logger, and timing cache).

    **Process:**
    1. Loads ONNX model (from file path or bytes)
    2. Builds optimized TensorRT engine (uses timing cache for speed)
    3. Runs warmup iterations to stabilize performance
    4. Measures latency across multiple timing iterations
    5. Returns median latency

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

    Note:
        Requires _benchmark_instance to be initialized via init_benchmark_instance()
        before calling this function. Otherwise returns float('inf').

    Example:
        >>> init_benchmark_instance("timing.cache", warmup_runs=5, timing_runs=20)
        >>> latency = benchmark_onnx_model("model.onnx", log_file="build.log")
        >>> print(f"Latency: {latency:.2f} ms")
    """
    global _benchmark_instance

    if _benchmark_instance is None:
        logger.error("Benchmark instance not initialized")
        return float("inf")

    try:
        # Run TensorRT benchmark
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
):
    """Initialize global TensorRT benchmark instance for model performance measurement.

    Creates and configures a TensorRTPyBenchmark instance that persists across
    multiple benchmark_onnx_model() calls for efficiency. The instance reuses
    TensorRT Builder, Runtime, Logger, and timing cache.

    **Benefits of Persistent Instance:**
    - Avoids repeated initialization overhead
    - Reuses timing cache across multiple models
    - Maintains consistent benchmark configuration


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

    Returns:
        TensorRTPyBenchmark or TrtExecBenchmark instance if initialization succeeds, None on failure

    Example:
        >>> # Initialize with default settings using TensorRT Python API
        >>> benchmark = init_benchmark_instance(use_trtexec=False)
        >>> if benchmark:
        ...     latency = benchmark_onnx_model("model.onnx")
        ...     print(f"Latency: {latency:.2f} ms")

    See Also:
        benchmark_onnx_model(): Uses the initialized instance to benchmark models
    """
    global _benchmark_instance
    try:
        if use_trtexec:
            _benchmark_instance = TrtExecBenchmark(
                timing_cache_file=timing_cache_file,
                warmup_runs=warmup_runs,
                timing_runs=timing_runs,
                plugin_libraries=plugin_libraries,
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
        return True  # No filter means all regions pass

    # Get all node indices in this region (including children)
    node_indices = region.get_all_nodes_recursive()

    for node_idx in node_indices:
        if node_idx < len(graph.nodes):
            node_name = graph.nodes[node_idx].name
            for pattern in filter_patterns:
                if fnmatch.fnmatch(node_name, pattern):
                    return True

    return False


# =============================================================================
# Autotuning Workflow
# =============================================================================


def region_pattern_autotuning_workflow(
    model_path: str,
    output_dir: Path,
    num_schemes_per_region: int = 30,
    pattern_cache_file: str | None = None,
    state_file: str | None = None,
    quant_type: str = "int8",
    default_dq_dtype: str = "float32",
    qdq_baseline_model: str | None = None,
    node_filter_list: list[str] | None = None,
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

    **State Management (Crash Recovery):**
    - Automatically saves checkpoint after profiling each region
    - Resume from interruption by running same command (auto-detects state file)
    - State file contains:
      * Baseline latency measurement
      * All profiled pattern schemes and their latencies
      * Best scheme selection for each pattern
      * Region discovery results and pattern assignments
    - Enables long-running optimizations with fault tolerance
    - Safe for cluster environments with preemption

    **Pattern Cache (Warm-Start Optimization):**
    - Pattern cache files (YAML format) contain top-performing schemes indexed by pattern
    - Stores results from previous optimization runs for reuse
    - Used to seed scheme generation (warm-start vs cold-start)
    - Benefits:
      * Reduces exploration time by prioritizing known-good configurations
      * Transfers learned schemes across similar models or model versions
      * Accumulates knowledge across multiple optimization sessions
      * Particularly effective for models with recurring pattern structures
    - Cache is pattern-specific, not model-specific (generalizes well)

    **QDQ Baseline Model (Transfer Learning):**
    - If provided, extracts Q/DQ insertion points from a pre-quantized model
    - Identifies which tensors are quantized in the baseline model
    - Maps these quantization points to region patterns in the current model
    - Updates pattern cache with learned insertion strategies
    - Enables warm-start from:
      * Expert-tuned quantized models (manually optimized)
      * Previous autotuning runs (transfer across model versions)
      * Reference implementations (e.g., from framework exporters)

    Args:
        model_path: Path to ONNX model file to optimize
        output_dir: Directory for output files (state, logs, models). Created if doesn't exist.
        num_schemes_per_region: Number of Q/DQ insertion schemes to test per region pattern.
                               Higher values explore more configurations but take longer (default: 30)
        pattern_cache_file: Optional path to pattern cache YAML file containing known-good schemes
                           from previous runs. Enables warm-start optimization (default: None)
        state_file: Optional path to state file for checkpoint/resume. If None, automatically
                   uses <output_dir>/autotuner_state.yaml (default: None)
        quant_type: Quantization data type - "int8" for INT8 quantization (default),
                   "fp8" for FP8 quantization
        qdq_baseline_model: Optional path to a pre-quantized ONNX model. If provided,
                           extracts Q/DQ insertion patterns and adds them to pattern cache
                           for warm-start (default: None)

    Returns:
        Configured QDQAutotuner instance containing:
        - All discovered regions and their patterns
        - Profiled Q/DQ insertion schemes for each pattern
        - Best scheme selections and performance measurements
        - Complete optimization state (can be saved/loaded)

        The returned autotuner can be used for:
        - Exporting optimized models with best Q/DQ schemes
        - Analyzing per-pattern optimization results
        - Further refinement or experimentation
        - Pattern cache generation for future runs

    Example:
        >>> # Initial run
        >>> autotuner = region_pattern_autotuning_workflow("model.onnx", Path("./output"))
        >>> # Resume from interruption
        >>> autotuner = region_pattern_autotuning_workflow("model.onnx", Path("./output"))
        >>> # With pattern cache warm-start
        >>> autotuner = region_pattern_autotuning_workflow(
        ...     "model.onnx", Path("./output"), pattern_cache_file="./pattern_cache.yaml"
        ... )
        >>> # With QDQ baseline model for pattern import
        >>> autotuner = region_pattern_autotuning_workflow(
        ...     "model.onnx", Path("./output"), qdq_baseline_model="quantized_baseline.onnx"
        ... )
    """
    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    models_dir = output_dir / "region_models"
    models_dir.mkdir(exist_ok=True)

    # Determine state file path
    if state_file is None:
        state_file = str(output_dir / "autotuner_state.yaml")
    state_path = Path(state_file)

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = onnx.load(model_path)

    # Load pattern cache if provided
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

    # Initialize autotuner with config
    logger.info(
        f"Initializing autotuner (quant_type={quant_type}, default_dq_dtype={default_dq_dtype})"
    )
    config = Config(
        default_quant_type=quant_type,
        default_dq_dtype=default_dq_dtype,
        verbose=True,
    )

    autotuner = QDQAutotuner(model)
    autotuner.initialize(config, pattern_cache)

    # Load previous state if exists (resume capability)
    if state_path.exists():
        logger.info(f"Resuming from checkpoint: {state_path}")
        autotuner.load_state(str(state_path))
    else:
        logger.info("Starting new autotuning session")

    # Import quantization patterns from QDQ baseline model if provided
    if qdq_baseline_model:
        qdq_baseline_path = Path(qdq_baseline_model)
        if qdq_baseline_path.exists():
            logger.info(f"Importing patterns from QDQ baseline: {qdq_baseline_model}")
            qdq_model = onnx.load(str(qdq_baseline_path))

            # Extract quantized tensors from baseline model
            quantized_tensors = get_quantized_tensors(qdq_model)
            logger.debug(f"Found {len(quantized_tensors)} quantized tensors in baseline")

            # Import insertion points into pattern cache
            autotuner.import_insertion_points(quantized_tensors)
            logger.info("Pattern import complete")
        else:
            logger.warning(f"QDQ baseline not found: {qdq_baseline_model}")

    # Get discovered regions
    regions = autotuner.regions
    logger.info(f"Ready to profile {len(regions)} regions")

    # Measure baseline (no Q/DQ) if not already measured
    if autotuner.baseline_latency_ms is None:
        logger.info("Measuring baseline (no Q/DQ)")
        baseline_path = output_dir / "baseline.onnx"
        autotuner.export_onnx(str(baseline_path), insert_qdq=False)
        baseline_log = logs_dir / "baseline.log"
        baseline_latency = benchmark_onnx_model(str(baseline_path), str(baseline_log))
        autotuner.submit(baseline_latency)
        logger.info(f"Baseline: {baseline_latency:.2f} ms")
    else:
        baseline_latency = autotuner.baseline_latency_ms
        logger.info(f"Using baseline from checkpoint: {baseline_latency:.2f} ms")

    # Profile regions
    logger.info(f"Starting region profiling ({num_schemes_per_region} schemes per region)")

    iteration_count = 0

    for region_idx, region in enumerate(regions):
        logger.info(
            f"Region {region_idx + 1}/{len(regions)} (ID={region.id}, level={region.get_level()})"
        )

        # Check if region matches node filter list
        if node_filter_list and not _region_matches_filter(
            region, autotuner.graph, node_filter_list
        ):
            logger.info("  Skipping (no nodes match filter patterns)")
            continue

        # Set as current profile region
        # Commit previous region's results (except for first region)
        commit = region_idx > 0
        autotuner.set_profile_region(region, commit=commit)

        # Check if already profiled (from loaded state)
        if autotuner.current_profile_pattern_schemes is None:
            logger.info("  Skipping (already profiled)")
            continue

        # Generate and test schemes for this region
        schemes_tested = 0
        for scheme_num in range(num_schemes_per_region):
            iteration_count += 1

            # Generate new scheme
            scheme_idx = autotuner.generate()

            if scheme_idx == -1:
                logger.debug(f"  Stopping at scheme {scheme_num + 1} (no more unique schemes)")
                break

            schemes_tested += 1

            # Export model with this scheme + best from previous regions
            model_bytes = autotuner.export_onnx(None, insert_qdq=True)

            # Benchmark with TensorRT
            test_log = logs_dir / f"region_{region.id}_scheme_{scheme_idx}.log"
            flush_timing_cache = (iteration_count % 10) == 0
            latency = benchmark_onnx_model(
                model_bytes, str(test_log), flush_timing_cache=flush_timing_cache
            )

            # Record result
            autotuner.submit(latency, success=(latency != float("inf")))

        # Display region summary
        ps = autotuner.current_profile_pattern_schemes
        if ps and ps.schemes:
            best_scheme = ps.best_scheme
            if best_scheme and best_scheme.latency_ms < float("inf") and baseline_latency > 0:
                speedup = baseline_latency / best_scheme.latency_ms
                logger.info(
                    f"  Tested {schemes_tested} schemes: "
                    f"best {best_scheme.latency_ms:.2f} ms ({speedup:.3f}x speedup)"
                )
            else:
                logger.info(f"  Tested {schemes_tested} schemes: no valid measurements")
        else:
            logger.info(f"  Tested {schemes_tested} schemes")

        # Save best model for this region (before committing to next region)
        region_model_path = models_dir / f"region_{region.id}_level_{region.get_level()}.onnx"
        autotuner.export_onnx(str(region_model_path), insert_qdq=True, best=True)
        logger.debug(f"  Saved best model: {region_model_path.name}")

        # Save state after each region (incremental, crash recovery)
        autotuner.save_state(str(state_path))
        logger.debug("  Checkpoint saved")

    # Commit final region
    autotuner.set_profile_region(None, commit=True)

    # Export and measure final optimized model
    logger.info("Exporting final optimized model")
    final_model_path = output_dir / "optimized_final.onnx"
    autotuner.export_onnx(str(final_model_path), insert_qdq=True)
    final_log = logs_dir / "final.log"
    final_latency = benchmark_onnx_model(str(final_model_path), str(final_log))

    # Display results
    if final_latency > 0 and final_latency != float("inf"):
        speedup = baseline_latency / final_latency
        logger.info(
            f"Results: {baseline_latency:.2f} ms → {final_latency:.2f} ms ({speedup:.3f}x speedup)"
        )
    else:
        logger.info(f"Results: {baseline_latency:.2f} ms → failed (invalid measurement)")

    # Save final state
    autotuner.save_state(str(state_path))

    logger.info("Autotuning complete")
    logger.info(f"  Final model: {final_model_path}")
    logger.info(f"  State: {state_path}")
    logger.debug(f"  Logs: {logs_dir}")
    logger.debug(f"  Region models: {models_dir}")

    return autotuner
