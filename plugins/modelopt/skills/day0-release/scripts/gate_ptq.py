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
  1. Output smaller than source. Growth blocks unless the summary declares an
     already-sub-8-bit source, which cannot shrink further under a 4-bit recipe.
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
      "metadata_diffs": [str, ...],  # unexpected diffs only; [] if clean

      # Optional. Precision of the SOURCE checkpoint's weights. Required to waive the
      # size check: a source already at 4 bits cannot shrink further under a 4-bit
      # recipe. Matched against a CLOSED vocabulary (_SUB8_SOURCE_PRECISIONS) -- any
      # other value, including a free-form description, blocks like an absent field.
      # Mixed-precision sources: record the precision of the DOMINANT weight mass, since
      # that is what decides whether the checkpoint can shrink (e.g. a model with MXFP4
      # experts at ~96% of bytes and BF16 attention is "mxfp4", not "mixed").
      "source_precision": str,
      # Optional, last resort. Must be the literal boolean true. Waives the size check
      # unconditionally (no growth bound) and records no reason; prefer source_precision,
      # whose claim we can actually check.
      "accept_size_growth": bool
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


# Largest source->output growth an already-4-bit source can explain. NVFP4 over MXFP4
# keeps the E2M1 nibbles but swaps an E8M0 scale per 32 elements for an E4M3 per 16, so
# scale bytes double; published checkpoints land near 1.06x. Beyond this, suspect a real
# problem rather than an inherent one.
_INHERENT_GROWTH_MAX = 1.10

# Source precisions that cannot shrink further under a 4-bit recipe. Growth is only
# excused when the summary declares one of these (or sets accept_size_growth).
_SUB8_SOURCE_PRECISIONS = frozenset({"mxfp4", "nvfp4", "fp4", "int4", "w4a16", "awq", "4bit"})


def evaluate_checkpoint(summary):
    """Validate an exported quantized checkpoint summary.

    Returns dict ``{pass, failure_class, detail, checks, notes}``, where ``notes``
    holds non-blocking observations and is present on every path.
    """
    if not summary:
        return {
            "pass": False,
            "notes": [],
            "failure_class": "USER_CONFIG_ERROR",
            "detail": "empty validation summary",
            "checks": {},
        }

    src = summary.get("source_bytes")
    out = summary.get("output_bytes")
    recipe = (summary.get("recipe") or "").lower()
    source_precision = str(summary.get("source_precision") or "").strip().lower()
    # Exact membership, not substring: "not_mxfp4" must not match. And require a real
    # boolean, since a JSON string "false" is truthy and would silently waive the gate.
    accept_growth = summary.get("accept_size_growth") is True
    source_is_sub8 = source_precision in _SUB8_SOURCE_PRECISIONS
    counts = summary.get("layer_precision_counts") or {}
    metadata_diffs = summary.get("metadata_diffs") or []

    checks = {}
    failures = []
    notes = []  # non-blocking observations

    # Check 1 — size.
    if not isinstance(src, (int, float)) or not isinstance(out, (int, float)) or src <= 0:
        checks["size"] = "missing/invalid source or output bytes"
        failures.append(("USER_CONFIG_ERROR", "missing source/output sizes"))
    else:
        ratio = out / src
        checks["size"] = f"{out}/{src} = {ratio:.3f}x"
        if ratio >= 1.0:
            # Blocking by default (ptq/references/checkpoint-validation.md: a ratio >= 1.0 for
            # a compression recipe blocks "unless the user explicitly accepts the explanation").
            # Two distinct waivers, deliberately not conflated:
            #   source_precision -- we can check the claim, so it is bounded by the growth an
            #     already-4-bit source explains (NVFP4 over MXFP4 keeps the E2M1 nibbles but
            #     swaps an E8M0 scale per 32 for an E4M3 per 16, so scale bytes double).
            #   accept_size_growth -- an explicit human override. We cannot check the reason,
            #     and the reference states it without a bound, so neither do we.
            if accept_growth:
                notes.append(
                    f"SIZE_NOT_REDUCED waived: {ratio:.3f}x growth accepted explicitly via "
                    "accept_size_growth (no source precision declared)"
                )
            elif source_is_sub8 and ratio <= _INHERENT_GROWTH_MAX:
                notes.append(
                    f"SIZE_NOT_REDUCED waived: {ratio:.3f}x growth is inherent for the declared "
                    f"{source_precision!r} source; judge reduction against BF16"
                )
            else:
                why = (
                    f"declared {source_precision!r} source explains at most {_INHERENT_GROWTH_MAX}x"
                    if source_is_sub8
                    else (
                        f"source_precision={source_precision or None!r} is not one of "
                        f"{sorted(_SUB8_SOURCE_PRECISIONS)}, so growth is not explained"
                    )
                )
                failures.append(("SIZE_NOT_REDUCED", f"output {ratio:.3f}x source, {why}"))

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
            "notes": notes,
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
        "notes": notes,
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
                    "checks": {},
                    "notes": [],
                }
            )
        )
        return 2

    try:
        with open(args.summary) as f:
            summary = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(
            json.dumps(
                {
                    "pass": False,
                    "failure_class": "USER_CONFIG_ERROR",
                    "detail": str(e),
                    "checks": {},
                    "notes": [],
                }
            )
        )
        return 2

    if args.recipe:
        summary["recipe"] = args.recipe

    result = evaluate_checkpoint(summary)
    print(json.dumps(result, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
