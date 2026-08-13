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

"""Day-0 post-quantization checkpoint gate.

Mirrors the required checks in ptq/references/checkpoint-validation.md:
  1. Output smaller than source (size ratio < 1 for a compression recipe).
  2. Quantized-weight coverage matches the requested recipe (no intended layer
     group left unquantized).
  3. No unexpected metadata diffs vs the source.

Pure decision logic in ``evaluate_checkpoint`` (unit-tested without real
checkpoints); ``main`` reads a validation-summary JSON produced from the
exported checkpoint (e.g. from hf_ptq.py's quant summary + a size scan) and
prints the verdict.

Validation summary shape:
    {
      "source_bytes": int,
      "output_bytes": int,
      "recipe": "nvfp4" | "fp8" | "nvfp4_mlp_only" | ...,
      "layer_precision_counts": {
          "NVFP4": int, "FP8": int, "INT4": int,
          "BF16_or_excluded": int,
          "unexpected_unquantized": int,
          "declaration_mismatch": int
      },
      "metadata_diffs": [str, ...]   # unexpected diffs only; [] if clean
    }
"""

from __future__ import annotations

import argparse
import json
import sys

# Which precision bucket each recipe is expected to populate with a nonzero count.
_RECIPE_EXPECTED_PRECISION = {
    "nvfp4": "NVFP4",
    "nvfp4_mlp_only": "NVFP4",
    "nvfp4_experts_only": "NVFP4",
    "nvfp4_omlp_only": "NVFP4",
    "fp8": "FP8",
    "int4_awq": "INT4",
}


def evaluate_checkpoint(summary):
    """Validate an exported quantized checkpoint summary.

    Returns dict ``{pass, failure_class, detail, checks}``.
    """
    if not summary:
        return {
            "pass": False,
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "empty validation summary",
            "checks": {},
        }

    src = summary.get("source_bytes")
    out = summary.get("output_bytes")
    recipe = (summary.get("recipe") or "").lower()
    counts = summary.get("layer_precision_counts") or {}
    metadata_diffs = summary.get("metadata_diffs") or []

    checks = {}
    failures = []  # (failure_class, detail)

    # Check 1 — size.
    if not isinstance(src, (int, float)) or not isinstance(out, (int, float)) or src <= 0:
        checks["size"] = "missing/invalid source or output bytes"
        failures.append(("USER_CONFIG_ERROR", "missing source/output sizes"))
    else:
        ratio = out / src
        checks["size"] = f"{out}/{src} = {ratio:.3f}x"
        if ratio >= 1.0:
            failures.append(
                ("QUANT_COVERAGE_FAILURE", f"output not smaller than source (ratio {ratio:.3f})")
            )

    # Check 2 — coverage.
    expected_bucket = _RECIPE_EXPECTED_PRECISION.get(recipe)
    if expected_bucket is None:
        checks["coverage"] = f"unknown recipe {recipe!r}; cannot verify coverage"
        failures.append(("USER_CONFIG_ERROR", f"unknown recipe {recipe!r}"))
    else:
        covered = counts.get(expected_bucket, 0)
        unexpected = counts.get("unexpected_unquantized", 0)
        mismatch = counts.get("declaration_mismatch", 0)
        checks["coverage"] = (
            f"{expected_bucket}={covered}, "
            f"unexpected_unquantized={unexpected}, "
            f"declaration_mismatch={mismatch}"
        )
        if covered == 0:
            failures.append(
                (
                    "MODEL_UNSUPPORTED",
                    f"recipe {recipe} targets {expected_bucket} but 0 layers covered "
                    "(wildcard likely missed the module names)",
                )
            )
        if unexpected > 0:
            failures.append(
                ("QUANT_COVERAGE_FAILURE", f"{unexpected} layer(s) unexpectedly unquantized")
            )
        if mismatch > 0:
            failures.append(
                (
                    "QUANT_COVERAGE_FAILURE",
                    f"{mismatch} layer(s) with precision/declaration mismatch",
                )
            )

    # Check 3 — metadata.
    checks["metadata"] = "clean" if not metadata_diffs else f"{len(metadata_diffs)} diff(s)"
    if metadata_diffs:
        failures.append(("QUANT_COVERAGE_FAILURE", f"unexpected metadata diffs: {metadata_diffs}"))

    if not failures:
        return {
            "pass": True,
            "failure_class": None,
            "detail": "size, coverage, and metadata all pass",
            "checks": checks,
        }

    # Surface the most actionable failure_class first: MODEL_UNSUPPORTED >
    # QUANT_COVERAGE_FAILURE > USER_CONFIG_ERROR.
    order = ["MODEL_UNSUPPORTED", "QUANT_COVERAGE_FAILURE", "USER_CONFIG_ERROR"]
    failures.sort(key=lambda f: order.index(f[0]) if f[0] in order else len(order))
    return {
        "pass": False,
        "failure_class": failures[0][0],
        "detail": "; ".join(d for _, d in failures),
        "checks": checks,
    }


def main(argv=None):
    """CLI entry point: read a validation-summary JSON and print the verdict."""
    p = argparse.ArgumentParser(description="Day-0 post-quantization checkpoint gate")
    p.add_argument("--summary", help="validation-summary JSON (see module docstring)")
    p.add_argument("--recipe", help="qformat; overrides the recipe recorded in the summary")
    args = p.parse_args(argv)

    if not args.summary:
        print(
            json.dumps(
                {
                    "pass": False,
                    "failure_class": "USER_CONFIG_ERROR",
                    "detail": "v1 requires --summary <validation-summary.json>; "
                    "produce it from the exported checkpoint (size scan + hf_ptq quant summary)",
                }
            )
        )
        return 2

    try:
        with open(args.summary) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"pass": False, "failure_class": "USER_CONFIG_ERROR", "detail": str(e)}))
        return 2

    if args.recipe:
        summary["recipe"] = args.recipe

    result = evaluate_checkpoint(summary)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
