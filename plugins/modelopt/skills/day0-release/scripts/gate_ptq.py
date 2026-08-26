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
  1. Output smaller than source. Growth blocks unless either the declared
     ``source_precision`` is already at or below the recipe's target bits (waived up to
     ``_INHERENT_GROWTH_MAX``, since scale bytes can still grow), or
     ``accept_size_growth: true`` overrides it (unbounded, and records no reason).
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
      # size check: a recipe can only shrink a source wider than its target, so an fp8
      # source under an fp8 recipe cannot shrink either. Matched against a CLOSED
      # vocabulary (_PRECISION_BITS) -- any other value, including a free-form
      # description, blocks like an absent field.
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
# Which precision bucket each recipe is expected to populate, and its target bits per
# weight. Both live here so adding a recipe cannot silently omit the bits: deriving them
# by substring-scanning the recipe name is unsafe, since a name that mentions a
# KV-cache or excluded-layer precision (e.g. "fp8_bf16_kv") would resolve to that
# instead -- and a too-wide target makes the size waiver fire on a real failed
# compression.
_RECIPE_EXPECTED_PRECISION = {
    "nvfp4": ("NVFP4", 4),
    "nvfp4_mlp_only": ("NVFP4", 4),
    "nvfp4_experts_only": ("NVFP4", 4),
    "nvfp4_omlp_only": ("NVFP4", 4),
    "fp8": ("FP8", 8),
    "int4_awq": ("INT4", 4),
}


# Largest growth we are willing to call inherent when the recipe cannot shrink the
# source. Calibrated on the best-understood case -- NVFP4 over MXFP4 keeps the E2M1
# nibbles but swaps an E8M0 scale per 32 elements for an E4M3 per 16, so scale bytes
# double and published checkpoints land near 1.06x. It is applied to 8-bit sources as a
# conservative ceiling rather than a derived figure: those paths add scale/metadata
# overhead too, but nobody has characterised it, so this bounds rather than blesses them.
_INHERENT_GROWTH_MAX = 1.10

# Bits per weight, used to decide whether a recipe can shrink a given source at all.
_PRECISION_BITS = {
    "fp32": 32,
    "bf16": 16,
    "fp16": 16,
    "fp8": 8,
    "int8": 8,
    "mxfp8": 8,
    "mxfp4": 4,
    "nvfp4": 4,
    "fp4": 4,
    "int4": 4,
    "w4a16": 4,
    "awq": 4,
    "4bit": 4,
}


def _recipe_bits(recipe):
    """Target bits per weight for a recipe, or None if the recipe is not recognised.

    Exact lookup, not a substring scan: see the note on _RECIPE_EXPECTED_PRECISION.
    """
    entry = _RECIPE_EXPECTED_PRECISION.get(recipe)
    return entry[1] if entry else None


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
    # A recipe can only shrink a source that is wider than its target. Comparing bits
    # rather than testing a fixed 4-bit line means an fp8 source under an fp8 recipe is
    # correctly treated as unable to shrink, instead of being told it "should compress".
    src_bits = _PRECISION_BITS.get(source_precision)
    tgt_bits = _recipe_bits(recipe)
    bits_known = src_bits is not None and tgt_bits is not None
    source_cannot_shrink = bits_known and src_bits <= tgt_bits
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
        if ratio >= 1.0 and tgt_bits is None and recipe:
            # Growth genuinely cannot be assessed without a target precision, and the
            # unknown-recipe USER_CONFIG_ERROR below is the actionable failure. Emitting
            # SIZE_NOT_REDUCED here would outrank it and hand the operator a triage row
            # about compression when the real problem is the recipe name.
            checks["size"] += " (not assessed: unknown recipe)"
        elif ratio >= 1.0:
            # Blocking by default (ptq/references/checkpoint-validation.md: a ratio >= 1.0 for
            # a compression recipe blocks "unless the user explicitly accepts the explanation").
            # Two distinct waivers, deliberately not conflated:
            #   source_precision -- we can check the claim, so it is bounded by the growth an
            #     already-4-bit source explains (NVFP4 over MXFP4 keeps the E2M1 nibbles but
            #     swaps an E8M0 scale per 32 for an E4M3 per 16, so scale bytes double).
            #   accept_size_growth -- an explicit human override. We cannot check the reason,
            #     and the reference states it without a bound, so neither do we.
            if accept_growth:
                # Both fields can be set. Say which one won, and that the bounded,
                # checkable waiver was discarded in favour of the unbounded override.
                extra = (
                    f"; source_precision={source_precision!r} was also declared, but the "
                    "explicit override takes precedence (unbounded)"
                    if source_precision
                    else " (no source precision declared)"
                )
                notes.append(
                    f"SIZE_NOT_REDUCED waived: {ratio:.3f}x growth accepted explicitly via "
                    f"accept_size_growth{extra}"
                )
            elif source_cannot_shrink and ratio <= _INHERENT_GROWTH_MAX:
                notes.append(
                    f"SIZE_NOT_REDUCED waived: {ratio:.3f}x growth is inherent for the declared "
                    f"{source_precision!r} source; judge reduction against BF16"
                )
            else:
                if source_cannot_shrink:
                    why = (
                        f"declared {source_precision!r} source explains at most "
                        f"{_INHERENT_GROWTH_MAX}x"
                    )
                elif bits_known:
                    why = (
                        f"declared {source_precision!r} source should compress under this "
                        "recipe, so growth is not explained"
                    )
                elif tgt_bits is not None:
                    why = (
                        f"source_precision={source_precision or None!r} is not a recognised "
                        "precision token"
                    )
                else:
                    why = (
                        f"recipe {recipe!r} has no known target precision, so growth "
                        "cannot be assessed"
                    )
                # Every branch points at the reference: the bounded-claim branch quotes a
                # bare number, which is the one an operator is most likely to question.
                why += " (see ptq/references/checkpoint-validation.md)"
                failures.append(("SIZE_NOT_REDUCED", f"output {ratio:.3f}x source, {why}"))

    # Check 2 — coverage.
    entry = _RECIPE_EXPECTED_PRECISION.get(recipe)
    expected_bucket = entry[0] if entry else None
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
    # QUANT_COVERAGE_FAILURE > SIZE_NOT_REDUCED > USER_CONFIG_ERROR. Anything unranked
    # sorts last via the len(order) fallback, so new classes must be added here.
    order = [
        "MODEL_UNSUPPORTED",
        "QUANT_COVERAGE_FAILURE",
        "SIZE_NOT_REDUCED",
        "USER_CONFIG_ERROR",
    ]
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
