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

"""Command-line interface for ONNX Q/DQ autotuning."""

import argparse
import sys
import tempfile
from pathlib import Path

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.autotune.utils import (
    StoreWithExplicitFlag,
    get_node_filter_list,
    validate_file_path,
)
from modelopt.onnx.quantization.autotune.subgraph_workflow import (
    subgraph_autotuning_workflow,
)
from modelopt.onnx.quantization.autotune.workflows import (
    init_benchmark_instance,
    region_pattern_autotuning_workflow,
)

DEFAULT_OUTPUT_DIR = "./autotuner_output"
DEFAULT_NUM_SCHEMES = 50
DEFAULT_QUANT_TYPE = "int8"
DEFAULT_DQ_DTYPE = "float32"
DEFAULT_TIMING_CACHE = str(Path(tempfile.gettempdir()) / "trtexec_timing.cache")
DEFAULT_WARMUP_RUNS = 50
DEFAULT_TIMING_RUNS = 100
MODE_PRESETS = {
    "quick": {"schemes_per_region": 30, "warmup_runs": 10, "timing_runs": 50},
    "default": {
        "schemes_per_region": DEFAULT_NUM_SCHEMES,
        "warmup_runs": DEFAULT_WARMUP_RUNS,
        "timing_runs": DEFAULT_TIMING_RUNS,
    },
    "extensive": {"schemes_per_region": 200, "warmup_runs": 50, "timing_runs": 200},
}


def _load_calibration_data(path: str | None):
    """Load calibration data from .npy or .npz file, or return None."""
    if path is None:
        return None
    import numpy as np

    path_str = str(path)
    if path_str.endswith(".npz"):
        data = np.load(path_str)
        return dict(data)
    elif path_str.endswith(".npy"):
        return np.load(path_str)
    else:
        logger.warning(
            f"Unsupported calibration data format: {path_str}. "
            "Expected .npy or .npz. Using random calibration."
        )
        return None


def apply_mode_presets(args) -> None:
    """Apply --mode preset to schemes_per_region, warmup_runs, timing_runs.

    Only applies preset for an option when that option was not explicitly set on the
    command line (explicit flags override the preset even when the value equals the default).
    """
    if args.mode not in MODE_PRESETS:
        return
    preset = MODE_PRESETS[args.mode]
    if not getattr(args, "_explicit_num_schemes", False):
        args.num_schemes = preset["schemes_per_region"]
    if not getattr(args, "_explicit_warmup_runs", False):
        args.warmup_runs = preset["warmup_runs"]
    if not getattr(args, "_explicit_timing_runs", False):
        args.timing_runs = preset["timing_runs"]


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
    if hasattr(args, "trtexec_benchmark_args") and args.trtexec_benchmark_args:
        logger.info(f"  Trtexec args: {args.trtexec_benchmark_args}")


def run_autotune() -> int:
    """Execute the Q/DQ autotuning workflow (region or subgraph).

    Parses command-line arguments, then:
    1. Validates input paths (model, baseline, output directory)
    2. Initializes TensorRT benchmark instance
    3. Runs the selected autotuning workflow
    4. Handles interruptions gracefully with state preservation

    Returns:
        Exit code:
        - 0: Success
        - 1: Autotuning failed (exception occurred)
        - 130: Interrupted by user (Ctrl+C)
    """
    args = get_parser().parse_args()
    apply_mode_presets(args)
    model_path = validate_file_path(args.onnx_path, "Model file")
    validate_file_path(args.qdq_baseline, "QDQ baseline model")
    output_dir = Path(args.output_dir)

    workflow = getattr(args, "workflow", "region")
    use_trtexec = workflow == "subgraph"

    log_benchmark_config(args)
    trtexec_args = getattr(args, "trtexec_benchmark_args", None)
    if trtexec_args and isinstance(trtexec_args, str):
        trtexec_args = trtexec_args.split()
    benchmark_instance = init_benchmark_instance(
        use_trtexec=use_trtexec,
        plugin_libraries=args.plugin_libraries,
        timing_cache_file=args.timing_cache,
        warmup_runs=args.warmup_runs,
        timing_runs=args.timing_runs,
        trtexec_args=trtexec_args,
    )

    if benchmark_instance is None:
        logger.error("Failed to initialize TensorRT benchmark")
        return 1

    logger.info(f"Autotuning workflow: {workflow}")

    custom_quantizable_ops = getattr(args, "quantizable_ops", None)
    if custom_quantizable_ops:
        custom_quantizable_ops = set(custom_quantizable_ops)
        logger.info(f"Custom quantizable ops: {custom_quantizable_ops}")

    calibration_data = _load_calibration_data(
        getattr(args, "calibration_data_path", None)
    )

    try:
        if workflow == "subgraph":
            graph_json = getattr(args, "graph_json", None)
            if graph_json:
                validate_file_path(graph_json, "graph.json file")

            subgraph_autotuning_workflow(
                model_path=str(model_path),
                output_dir=output_dir,
                graph_json_path=graph_json,
                quant_type=args.quant_type,
                plugin_libraries=args.plugin_libraries,
                schemes_per_group=args.num_schemes,
                strongly_typed=args.strongly_typed,
                extra_trtexec_args=trtexec_args,
                incremental_validation=args.incremental_validation,
                quantizable_ops=custom_quantizable_ops,
                calibration_data=calibration_data,
                calibration_method=args.calibration_method,
                calibration_shapes=args.calibration_shapes,
                calibration_eps=args.calibration_eps,
                high_precision_dtype=args.high_precision_dtype,
                use_external_data_format=args.use_external_data_format,
                skip_calibration=args.skip_calibration,
            )
        else:
            node_filter_list = get_node_filter_list(args.node_filter_list)
            region_pattern_autotuning_workflow(
                model_or_path=str(model_path),
                output_dir=output_dir,
                num_schemes_per_region=args.num_schemes,
                pattern_cache_file=args.pattern_cache_file,
                state_file=args.state_file,
                quant_type=args.quant_type,
                default_dq_dtype=args.default_dq_dtype,
                qdq_baseline_model=args.qdq_baseline,
                node_filter_list=node_filter_list,
                verbose=args.verbose,
            )

        logger.info("\n" + "=" * 70)
        logger.info("Autotuning completed successfully!")
        logger.info(f"Results: {output_dir}")
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


def get_parser() -> argparse.ArgumentParser:
    """Create and return the autotune CLI argument parser.

    Intended for Sphinx documentation and programmatic use (e.g. subparsers).
    """
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.autotune",
        description="ONNX Q/DQ Autotuning with TensorRT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx

  # Quick mode (fewer schemes and benchmark runs for fast iteration)
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx --mode quick

  # Extensive mode (more schemes and runs for thorough tuning)
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx --mode extensive

  # Import patterns from QDQ baseline model
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --qdq_baseline baseline.onnx

  # Use pattern cache for warm-start
  python -m modelopt.onnx.quantization.autotune --onnx_path model.onnx --pattern_cache cache.yaml

  # Full example with all options (region workflow)
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --schemes_per_region 50 \\
      --pattern_cache cache.yaml --qdq_baseline baseline.onnx \\
      --quant_type int8 --verbose

  # Subgraph workflow (faster, fusion-aware)
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --workflow subgraph

  # Subgraph workflow with pre-generated graph.json
  python -m modelopt.onnx.quantization.autotune \\
      --onnx_path model.onnx --workflow subgraph --graph_json model.fp16.graph.json
        """,
    )

    # Model and Output
    io_group = parser.add_argument_group("Model and Output")
    io_group.add_argument(
        "--onnx_path", "-m", type=str, required=True, help="Path to ONNX model file"
    )
    io_group.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        dest="output_dir",
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR})",
    )

    # Workflow Selection
    workflow_group = parser.add_argument_group("Workflow Selection")
    workflow_group.add_argument(
        "--workflow",
        type=str,
        default="region",
        choices=["region", "subgraph"],
        help="Autotuning workflow: 'region' (pattern-based, default) or 'subgraph' (fusion-aware, faster)",
    )
    workflow_group.add_argument(
        "--graph_json",
        type=str,
        default=None,
        help="Path to TRT graph.json for subgraph workflow. If omitted, one is generated via trtexec FP16.",
    )
    workflow_group.add_argument(
        "--incremental_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="incremental_validation",
        help="Enable/disable per-group incremental validation in subgraph workflow "
        "(default: enabled). Use --no-incremental_validation to disable.",
    )

    # Autotuning Strategy
    strategy_group = parser.add_argument_group("Autotuning Strategy")
    strategy_group.add_argument(
        "--mode",
        type=str,
        default="default",
        choices=["quick", "default", "extensive"],
        help="Preset for schemes_per_region, warmup_runs, and timing_runs. "
        "'quick': fewer schemes/runs for fast iteration; "
        "'default': balanced; "
        "'extensive': more schemes/runs for thorough tuning. "
        "Explicit --schemes_per_region, --warmup_runs, --timing_runs override the preset.",
    )
    strategy_group.add_argument(
        "--schemes_per_region",
        "-s",
        type=int,
        default=DEFAULT_NUM_SCHEMES,
        dest="num_schemes",
        action=StoreWithExplicitFlag,
        explicit_attr="_explicit_num_schemes",
        help=f"Schemes per region (default: {DEFAULT_NUM_SCHEMES}; preset from --mode if not set)",
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
        help="State file path for resume capability (default: <output_dir>/autotuner_state.yaml)",
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
    quant_group.add_argument(
        "--quantizable_ops",
        type=str,
        nargs="+",
        default=None,
        help="ONNX op types to treat as quantizable (default: Conv ConvTranspose MatMul Gemm). "
        "Example: --quantizable_ops Conv MatMul Einsum",
    )

    # Calibration (Phase 4)
    calib_group = parser.add_argument_group("Calibration (Final Export)")
    calib_group.add_argument(
        "--calibration_data_path",
        type=str,
        default=None,
        help="Path to calibration data (.npy or .npz file). "
        "If omitted, random calibration data is used.",
    )
    calib_group.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        choices=["entropy", "minmax", "percentile"],
        help="ORT calibration method (default: entropy)",
    )
    calib_group.add_argument(
        "--calibration_shapes",
        type=str,
        default=None,
        help="Input shape spec for calibration, e.g. 'input:1x3x224x224'",
    )
    calib_group.add_argument(
        "--calibration_eps",
        type=str,
        nargs="+",
        default=None,
        help="Execution providers for calibration (default: cpu cuda:0 trt)",
    )
    calib_group.add_argument(
        "--high_precision_dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32"],
        help="Non-quantized precision for the final model (default: fp16)",
    )
    calib_group.add_argument(
        "--skip_calibration",
        action="store_true",
        default=False,
        help="Skip Phase 4 calibration; output only placeholder Q/DQ model "
        "(for latency-only analysis)",
    )

    # Model Options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--strongly_typed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use --stronglyTyped for trtexec graph.json generation in subgraph workflow "
        "(default: enabled). Use --no-strongly_typed to disable.",
    )
    model_group.add_argument(
        "--use_external_data_format",
        action="store_true",
        default=False,
        help="Use ONNX external data format for large models (>2GB)",
    )

    # TensorRT Benchmark
    trt_group = parser.add_argument_group("TensorRT Benchmark")
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
        action=StoreWithExplicitFlag,
        explicit_attr="_explicit_warmup_runs",
        help=f"Number of warmup runs (default: {DEFAULT_WARMUP_RUNS}; preset from --mode applies if not set)",
    )
    trt_group.add_argument(
        "--timing_runs",
        type=int,
        default=DEFAULT_TIMING_RUNS,
        action=StoreWithExplicitFlag,
        explicit_attr="_explicit_timing_runs",
        help=f"Number of timing runs (default: {DEFAULT_TIMING_RUNS}; preset from --mode applies if not set)",
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
    trt_group.add_argument(
        "--trtexec_benchmark_args",
        type=str,
        default=None,
        help="Additional command-line arguments to pass to trtexec as a single quoted string. "
        "Example: --trtexec_benchmark_args '--fp16 --workspace=4096 --verbose'",
    )

    # Logging
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose DEBUG logging")

    return parser


if __name__ == "__main__":
    sys.exit(run_autotune())
