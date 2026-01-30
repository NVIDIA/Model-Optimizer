#!/usr/bin/env python3
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

"""CLI argument parsing and execution for ONNX Q/DQ autotuning.

This module provides `run_autotune` which handles both argument parsing and
workflow execution. See `__main__.py` for usage examples.
"""

import argparse
import sys
from pathlib import Path

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    region_pattern_autotuning_workflow,
)

DEFAULT_OUTPUT_DIR = "./autotuner_output"
DEFAULT_NUM_SCHEMES = 30
DEFAULT_QUANT_TYPE = "int8"
DEFAULT_DQ_DTYPE = "float32"
DEFAULT_TIMING_CACHE = "/tmp/trtexec_timing.cache"  # nosec B108
DEFAULT_WARMUP_RUNS = 5
DEFAULT_TIMING_RUNS = 20


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


def run_autotune(args=None) -> int:
    """Execute the complete pattern-based Q/DQ autotuning workflow.

    This function orchestrates the entire optimization process:
    1. Parses command-line arguments (if not provided)
    2. Validates input paths (model, baseline, output directory)
    3. Initializes TensorRT benchmark instance
    4. Runs pattern-based region autotuning workflow
    5. Handles interruptions gracefully with state preservation

    Args:
        args: Optional parsed command-line arguments. If None, parses sys.argv.

    Returns:
        Exit code:
        - 0: Success
        - 1: Autotuning failed (exception occurred)
        - 130: Interrupted by user (Ctrl+C)
    """
    if args is None:
        args = _get_autotune_parser().parse_args()

    model_path = validate_file_path(args.onnx_path, "Model file")
    validate_file_path(args.qdq_baseline, "QDQ baseline model")
    output_dir = Path(args.output)

    log_benchmark_config(args)
    init_benchmark_instance(
        use_trtexec=args.use_trtexec,
        plugin_libraries=args.plugin_libraries,
        timing_cache_file=args.timing_cache,
        warmup_runs=args.warmup_runs,
        timing_runs=args.timing_runs,
    )

    logger.info("Autotuning Mode: Pattern-Based")

    try:
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


def _get_autotune_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune",
        description="ONNX Q/DQ Autotuning with TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx

  # Import patterns from QDQ baseline model
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --qdq_baseline baseline.onnx

  # Use pattern cache for warm-start
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx --pattern_cache cache.yaml

  # Full example with all options
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --schemes_per_region 50 \\
      --pattern_cache cache.yaml --qdq_baseline baseline.onnx \\
      --quant_type int8 --verbose
        """,
    )

    # Model and Output
    io_group = parser.add_argument_group("Model and Output")
    io_group.add_argument(
        "--onnx_path", "-m", type=str, required=True, help="Path to ONNX model file"
    )
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
        "--schemes_per_region",
        "-s",
        type=int,
        default=DEFAULT_NUM_SCHEMES,
        dest="num_schemes",
        help=f"Number of schemes to test per region (default: {DEFAULT_NUM_SCHEMES})",
    )
    strategy_group.add_argument(
        "--pattern_cache",
        type=str,
        default=None,
        dest="pattern_cache_file",
        help="Path to pattern cache YAML for warm-start (optional)",
    )
    strategy_group.add_argument(
        "--qdq_baseline",
        type=str,
        default=None,
        help="Path to QDQ baseline ONNX model to import quantization patterns (optional)",
    )
    strategy_group.add_argument(
        "--state_file",
        type=str,
        default=None,
        help="State file path for resume capability (default: <output>/autotuner_state.yaml)",
    )
    strategy_group.add_argument(
        "--node_filter_list",
        type=str,
        default=None,
        help="Path to a file containing wildcard patterns to filter ONNX nodes (one pattern per line). "
        "Regions without any matching nodes are skipped during autotuning.",
    )

    # Quantization
    quant_group = parser.add_argument_group("Quantization")
    quant_group.add_argument(
        "--quant_type",
        type=str,
        default=DEFAULT_QUANT_TYPE,
        choices=["int8", "fp8"],
        help=f"Quantization data type (default: {DEFAULT_QUANT_TYPE})",
    )
    quant_group.add_argument(
        "--default_dq_dtype",
        type=str,
        default=DEFAULT_DQ_DTYPE,
        choices=["float16", "float32", "bfloat16"],
        help="Default DQ output dtype if cannot be deduced (optional)",
    )

    # TensorRT Benchmark
    trt_group = parser.add_argument_group("TensorRT Benchmark")
    trt_group.add_argument(
        "--use_trtexec",
        action="store_true",
        help="Use trtexec for benchmarking (default: False)",
        default=False,
    )
    trt_group.add_argument(
        "--timing_cache",
        type=str,
        default=DEFAULT_TIMING_CACHE,
        help=f"TensorRT timing cache file (default: {DEFAULT_TIMING_CACHE})",
    )
    trt_group.add_argument(
        "--warmup_runs",
        type=int,
        default=DEFAULT_WARMUP_RUNS,
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP_RUNS})",
    )
    trt_group.add_argument(
        "--timing_runs",
        type=int,
        default=DEFAULT_TIMING_RUNS,
        help=f"Number of timing runs (default: {DEFAULT_TIMING_RUNS})",
    )
    trt_group.add_argument(
        "--plugin_libraries",
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
