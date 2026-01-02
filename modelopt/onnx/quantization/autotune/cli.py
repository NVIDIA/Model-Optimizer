#!/usr/bin/env python3
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

"""CLI argument parsing and execution for ONNX Q/DQ autotuning.

This module provides:
- `get_autotune_parser`: Creates the argument parser for the CLI
- `run_autotune`: Executes the autotuning workflow with parsed arguments

See `__main__.py` for usage examples and documentation.
"""

import argparse
import logging
import sys
from pathlib import Path

from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    region_pattern_autotuning_workflow,
)

logger = logging.getLogger(__name__)

# Default values for CLI arguments
DEFAULT_OUTPUT_DIR = "./autotuner_output"
DEFAULT_NUM_SCHEMES = 30
DEFAULT_QUANT_TYPE = "int8"
DEFAULT_DQ_DTYPE = "float32"
DEFAULT_TIMING_CACHE = "/tmp/trtexec_timing.cache"  # nosec B108
DEFAULT_WARMUP_RUNS = 5
DEFAULT_TIMING_RUNS = 20


# =============================================================================
# Helper Functions
# =============================================================================


def validate_file_path(path: str | None, description: str) -> Path | None:
    """Validate that a file path exists.

    Args:
        path: Path string to validate (can be None)
        description: Description of the file for error messages

    Returns:
        Path object if valid, None if path is None

    Raises:
        SystemExit: If path is provided but doesn't exist
    """
    if path is None:
        return None

    path_obj = Path(path)
    if not path_obj.exists():
        logger.error(f"{description} not found: {path_obj}")
        sys.exit(1)

    return path_obj


def log_benchmark_config(args):
    """Log TensorRT benchmark configuration for transparency.

    Logs timing cache path, warmup/timing run counts, and any custom
    plugin libraries that will be loaded.

    Args:
        args: Parsed command-line arguments with benchmark configuration
    """
    logger.info("Initializing TensorRT benchmark")
    logger.info(f"  Timing cache: {args.timing_cache}")
    logger.info(f"  Warmup runs: {args.warmup_runs}")
    logger.info(f"  Timing runs: {args.timing_runs}")
    if args.plugin_libraries:
        logger.info(f"  Plugin libraries: {', '.join(args.plugin_libraries)}")


# =============================================================================
# Command Handler
# =============================================================================


def run_autotune(args) -> int:
    """Execute the complete pattern-based Q/DQ autotuning workflow.

    This function orchestrates the entire optimization process:
    1. Validates input paths (model, baseline, output directory)
    2. Initializes TensorRT benchmark instance
    3. Runs pattern-based region autotuning workflow
    4. Handles interruptions gracefully with state preservation

    Args:
        args: Parsed command-line arguments containing:
            - model: Path to input ONNX model
            - output: Output directory path
            - num_schemes: Number of schemes to test per region
            - pattern_cache_file: Optional pattern cache for warm-start
            - state_file: Optional state file for resume capability
            - quant_type: Quantization type (int8 or fp8)
            - qdq_baseline: Optional baseline model for pattern import
            - timing_cache, warmup_runs, timing_runs: TensorRT config
            - verbose: Debug logging flag

    Returns:
        Exit code:
        - 0: Success
        - 1: Autotuning failed (exception occurred)
        - 130: Interrupted by user (Ctrl+C)
    """
    # Validate input paths
    model_path = validate_file_path(args.model, "Model file")
    validate_file_path(args.qdq_baseline, "QDQ baseline model")
    output_dir = Path(args.output)

    # Initialize TensorRT benchmark
    log_benchmark_config(args)
    init_benchmark_instance(
        use_trtexec=args.use_trtexec,
        plugin_libraries=args.plugin_libraries,
        timing_cache_file=args.timing_cache,
        warmup_runs=args.warmup_runs,
        timing_runs=args.timing_runs,
    )

    logger.info("Autotuning Mode: Pattern-Based")

    # Run autotuning workflow
    try:
        # Load node filter patterns from file if provided
        node_filter_list = None
        if args.node_filter_list:
            filter_file = validate_file_path(args.node_filter_list, "Node filter list file")
            if filter_file:
                with open(filter_file) as f:
                    node_filter_list = [
                        line.strip()
                        for line in f
                        if line.strip() and not line.strip().startswith("#")
                    ]
                logger.info(f"Loaded {len(node_filter_list)} filter patterns from {filter_file}")

        region_pattern_autotuning_workflow(
            model_path=str(model_path),
            output_dir=output_dir,
            num_schemes_per_region=args.num_schemes,
            pattern_cache_file=args.pattern_cache_file,
            state_file=args.state_file,
            quant_type=args.quant_type,
            default_dq_dtype=args.default_dq_dtype,
            qdq_baseline_model=args.qdq_baseline,
            node_filter_list=node_filter_list,
        )

        # Success message
        logger.info("\n" + "=" * 70)
        logger.info("✓ Autotuning completed successfully!")
        logger.info(f"✓ Results: {output_dir}")
        logger.info("=" * 70)
        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        state_file = args.state_file or output_dir / "autotuner_state.yaml"
        logger.info(f"Progress saved to: {state_file}")
        return 130

    except Exception as e:
        logger.error(f"\nAutotuning failed: {e}", exc_info=args.verbose)
        return 1


# =============================================================================
# Argument Parser
# =============================================================================


def get_autotune_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser.

    Sets up argument groups for:
    - Model and Output: Input model and output directory
    - Autotuning Strategy: Scheme count, pattern cache, baseline import, state file
    - Quantization: Data type selection (int8/fp8)
    - TensorRT Benchmark: Timing cache, warmup/timing runs, plugins
    - Logging: Verbose debug mode

    Returns:
        Configured ArgumentParser instance with all CLI options
    """
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune",
        description="ONNX Q/DQ Autotuning with TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m modelopt.onnx.quantization.autotune --model model.onnx

  # Import patterns from QDQ baseline model
  python -m modelopt.onnx.quantization.autotune \\
      --model model.onnx --qdq-baseline baseline.onnx

  # Use pattern cache for warm-start
  python -m modelopt.onnx.quantization.autotune --model model.onnx --pattern-cache cache.yaml

  # Full example with all options
  python -m modelopt.onnx.quantization.autotune \\
      --model model.onnx --schemes-per-region 50 \\
      --pattern-cache cache.yaml --qdq-baseline baseline.onnx \\
      --quant-type int8 --verbose
        """,
    )

    # Model and Output
    io_group = parser.add_argument_group("Model and Output")
    io_group.add_argument("--model", "-m", type=str, required=True, help="Path to ONNX model file")
    io_group.add_argument(
        "--output",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Autotuning Strategy
    strategy_group = parser.add_argument_group("Autotuning Strategy")
    strategy_group.add_argument(
        "--schemes-per-region",
        "-s",
        type=int,
        default=DEFAULT_NUM_SCHEMES,
        dest="num_schemes",
        help=f"Number of schemes to test per region (default: {DEFAULT_NUM_SCHEMES})",
    )
    strategy_group.add_argument(
        "--pattern-cache",
        type=str,
        default=None,
        dest="pattern_cache_file",
        help="Path to pattern cache YAML for warm-start (optional)",
    )
    strategy_group.add_argument(
        "--qdq-baseline",
        type=str,
        default=None,
        help="Path to QDQ baseline ONNX model to import quantization patterns (optional)",
    )
    strategy_group.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="State file path for resume capability (default: <output>/autotuner_state.yaml)",
    )
    strategy_group.add_argument(
        "--node-filter-list",
        type=str,
        default=None,
        help="Path to a file containing wildcard patterns to filter ONNX nodes (one pattern per line). "
        "Regions without any matching nodes are skipped during autotuning.",
    )

    # Quantization
    quant_group = parser.add_argument_group("Quantization")
    quant_group.add_argument(
        "--quant-type",
        type=str,
        default=DEFAULT_QUANT_TYPE,
        choices=["int8", "fp8"],
        help=f"Quantization data type (default: {DEFAULT_QUANT_TYPE})",
    )
    quant_group.add_argument(
        "--default-dq-dtype",
        type=str,
        default=DEFAULT_DQ_DTYPE,
        choices=["float16", "float32", "bfloat16"],
        help="Default DQ output dtype if cannot be deduced (optional)",
    )

    # TensorRT Benchmark
    trt_group = parser.add_argument_group("TensorRT Benchmark")
    trt_group.add_argument(
        "--use-trtexec",
        action="store_true",
        help="Use trtexec for benchmarking (default: False)",
        default=False,
    )
    trt_group.add_argument(
        "--timing-cache",
        type=str,
        default=DEFAULT_TIMING_CACHE,
        help=f"TensorRT timing cache file (default: {DEFAULT_TIMING_CACHE})",
    )
    trt_group.add_argument(
        "--warmup-runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP_RUNS})",
    )
    trt_group.add_argument(
        "--timing-runs",
        type=int,
        default=DEFAULT_TIMING_RUNS,
        help=f"Number of timing runs (default: {DEFAULT_TIMING_RUNS})",
    )
    trt_group.add_argument(
        "--plugin-libraries",
        "--plugins",
        type=str,
        nargs="+",
        default=None,
        dest="plugin_libraries",
        help="TensorRT plugin libraries (.so files) to load (optional, space-separated)",
    )

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    return parser