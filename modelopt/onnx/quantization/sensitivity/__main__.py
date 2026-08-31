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

"""Command-line entrypoint for the ONNX quantization sensitivity scan.

Runs :func:`modelopt.onnx.quantization.sensitivity.score` and renders the ranked results to stderr
and to a JSON file.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from modelopt.onnx.logging_config import logger
from modelopt.onnx.quantization.__main__ import validate_file_size
from modelopt.onnx.quantization.sensitivity.score import Granularity, Metric, score

# 2 GiB matches the ``--onnx_path`` guard in ``modelopt.onnx.quantization.__main__``.
_ONNX_MAX_SIZE_BYTES = 2 * (1024**3)
# 4 GiB accommodates ImageNet-scale calibration NPZ files.
_CALIB_MAX_SIZE_BYTES = 4 * (1024**3)
# 16 GiB aggregate cap for a directory of .npz shards.
_CALIB_DIR_MAX_TOTAL_BYTES = 16 * (1024**3)


def _validate_calibration_dir(path: str) -> None:
    """Enforce per-file and aggregate size limits on a directory of ``.npz`` calibration shards.

    The directory loader in :func:`score` concatenates every ``.npz`` in the directory without
    bounds, so a directory containing many large shards can exhaust process memory during load.
    Cap each shard at ``_CALIB_MAX_SIZE_BYTES`` and the aggregate at
    ``_CALIB_DIR_MAX_TOTAL_BYTES``.

    Args:
        path: Directory expected to contain one or more ``.npz`` calibration shards.

    Raises:
        FileNotFoundError: If ``path`` contains no ``.npz`` files.
        ValueError: If any shard or the aggregate exceeds the limit.
    """
    files = sorted(glob.glob(os.path.join(path, "*.npz")))
    if not files:
        raise FileNotFoundError(f"No .npz files found under calibration directory: {path}")
    total = 0
    for f in files:
        validate_file_size(f, _CALIB_MAX_SIZE_BYTES)
        total += os.path.getsize(f)
    if total > _CALIB_DIR_MAX_TOTAL_BYTES:
        raise ValueError(
            f"Aggregate calibration directory size {total} bytes exceeds "
            f"{_CALIB_DIR_MAX_TOTAL_BYTES} bytes ({len(files)} shards under {path})."
        )


def _default_output_json(onnx_path: str) -> str:
    """Derive the default ``--output_json`` path next to the input ONNX file."""
    stem, _ = os.path.splitext(os.path.basename(onnx_path))
    return os.path.join(os.path.dirname(os.path.abspath(onnx_path)), f"{stem}.sensitivity.json")


def _render_ranked_table(result: dict, show_zero_scores: bool = False) -> str:
    """Format a sensitivity result as a two-column, high-to-low ranked table.

    Args:
        result: The return value of :func:`score`.
        show_zero_scores: If False (default), hide targets whose drift score is exactly ``0.0``.

    Returns:
        A newline-joined string with a header, one row per non-hidden target, and highest / lowest
        markers. A trailing footer notes the count of hidden zero-score rows and, when applicable,
        the number of unprobed / failed targets.
    """
    scores = result["scores"]
    failed = result.get("failed", [])
    header = (
        f"Sensitivity scan ({result['target_precision']} / "
        f"{result['metric']} / {result['granularity']}):"
    )
    if not scores:
        if failed:
            return (
                header + f"\n  (no scores produced; {len(failed)} target(s) failed to probe -- "
                f"see calibration_source / failed in the JSON)"
            )
        return header + "\n  (no quantizable targets found)"

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    hidden = 0
    if not show_zero_scores:
        visible = [(n, v) for n, v in ranked if v != 0.0]
        hidden = len(ranked) - len(visible)
        ranked = visible

    if not ranked:
        footer = f"\n  (all {hidden} target(s) scored 0.0 -- pass --show_zero_scores to see them)"
        if failed:
            footer += f"\n  ({len(failed)} additional target(s) failed to probe)"
        return header + footer

    name_width = max(len(name) for name, _ in ranked)
    lines = [header]
    for i, (name, value) in enumerate(ranked):
        marker = ""
        if i == 0:
            marker = "  <-- highest impact"
        elif i == len(ranked) - 1:
            marker = "  <-- lowest impact"
        lines.append(f"  {name:<{name_width}}  {value:.3f}{marker}")
    if hidden:
        lines.append(
            f"  ({hidden} target(s) with score 0.0 hidden; pass --show_zero_scores or read the JSON)"
        )
    if failed:
        lines.append(f"  ({len(failed)} target(s) failed to probe -- see failed in the JSON)")
    return "\n".join(lines)


def get_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for the sensitivity CLI."""
    parser = argparse.ArgumentParser(
        prog="modelopt.onnx.quantization.sensitivity",
        description=(
            "Rank ONNX quantization targets (op types or individual nodes) by their impact on "
            "model output. Emits a ranked table to stderr and a JSON file for downstream tooling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--onnx_path", required=True, type=str, help="Path to the input ONNX model."
    )
    parser.add_argument(
        "--calibration_data_path",
        type=str,
        default=None,
        help=(
            "Real calibration data (.npy, .npz, or a directory of .npz files). If omitted, "
            "falls back to synthetic random tensors and produces directional-only rankings."
        ),
    )
    parser.add_argument(
        "--num_calib_samples",
        type=int,
        default=100,
        help="Number of synthetic samples generated when --calibration_data_path is omitted.",
    )
    parser.add_argument(
        "--granularity",
        type=str,
        default=Granularity.OP_TYPE.value,
        choices=[g.value for g in Granularity],
        help="Scan granularity: 'op_type' (fast, one probe per type) or 'node' (per-instance).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=Metric.KL_DIV.value,
        choices=[m.value for m in Metric],
        help="Proxy metric between FP-reference and quantized graph outputs.",
    )
    parser.add_argument(
        "--target_precision",
        type=str,
        default="int8",
        choices=["int8", "fp8"],
        help="Precision to probe per target.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="entropy",
        choices=["entropy", "max"],
        help="Calibration method threaded through to quantize().",
    )
    parser.add_argument(
        "--calibration_eps",
        type=str,
        nargs="+",
        default=["cpu", "cuda:0", "trt"],
        help="ORT execution providers, in priority order.",
    )
    parser.add_argument(
        "--op_types_scope",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional whitelist of op types to probe. Defaults to every unique op type actually "
            "present in the ONNX graph."
        ),
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Where to write the sensitivity JSON. Defaults to <onnx_stem>.sensitivity.json.",
    )
    parser.add_argument(
        "--show_zero_scores",
        action="store_true",
        help="Include zero-score targets in the stderr ranked table.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``). Provided for programmatic use
            from tests and other callers.

    Returns:
        Process exit code: 0 on success, non-zero if :func:`score` raises.
    """
    args = get_parser().parse_args(argv)

    # Boundary validation on user-supplied paths -- mirrors modelopt.onnx.quantization.__main__.
    validate_file_size(args.onnx_path, _ONNX_MAX_SIZE_BYTES)
    if args.calibration_data_path is not None:
        if os.path.isdir(args.calibration_data_path):
            _validate_calibration_dir(args.calibration_data_path)
        else:
            validate_file_size(args.calibration_data_path, _CALIB_MAX_SIZE_BYTES)

    if args.calibration_data_path is None:
        logger.warning(
            "Synthetic random calibration -- scores are directional-only; do not pair with "
            "absolute thresholds. See calibration_source in the output JSON."
        )

    result = score(
        onnx_path=args.onnx_path,
        calibration_data=args.calibration_data_path,  # path -> score() delegates to its loader
        num_synthetic_samples=args.num_calib_samples,
        target_precision=args.target_precision,
        granularity=args.granularity,
        metric=args.metric,
        calibration_method=args.calibration_method,
        calibration_eps=args.calibration_eps,
        op_types_scope=args.op_types_scope,
    )
    payload = {"onnx_path": os.path.abspath(args.onnx_path), **result}
    output_json = args.output_json or _default_output_json(args.onnx_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_json)) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(_render_ranked_table(result, show_zero_scores=args.show_zero_scores), file=sys.stderr)
    print(f"Wrote {output_json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
