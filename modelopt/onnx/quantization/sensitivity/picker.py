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

"""Exclusion picker for the sensitivity primitive.

Turns a per-node or per-op-type score dictionary produced by :func:`sensitivity.score`
into an actionable ``--nodes_to_exclude`` or ``--op_types_to_exclude`` list (depending on
granularity) for :func:`modelopt.onnx.quantization.quantize`. Supports two policy modes:

* **Coverage mode** (default): pick the largest target set whose cumulative
  sensitivity score stays at or below ``coverage * total_mass``. Portable
  across architectures because the target is a fraction, not an absolute
  number.
* **Threshold mode**: exclude every target whose individual sensitivity
  score exceeds an absolute cutoff. Simpler and more predictable when the
  operator already knows what per-target sensitivity score magnitude they
  consider "too sensitive to quantize" for a given model.
"""

from __future__ import annotations

from collections.abc import Mapping

from modelopt.onnx.logging_config import logger


def suggest_exclusion(
    scores: Mapping[str, float],
    coverage: float = 0.90,
    *,
    threshold: float | None = None,
    max_nodes: int | None = None,
    min_score_floor: float = 0.0,
    near_tie_ratio: float | None = 0.99,
) -> list[str]:
    """Return an exclusion list from a per-target sensitivity score dictionary.

    Two policy modes are supported:

    * **Coverage mode** (the default): return the largest target set whose
      cumulative sensitivity score stays at or below ``coverage * total_mass``.
      Used when ``threshold`` is ``None``. The actual coverage will be less
      than or equal to the requested value -- adding the next target in the
      ranking would exceed the requested value, so the picker stops before
      crossing it.
    * **Threshold mode**: return every target whose sensitivity score
      exceeds ``threshold``. Used when ``threshold`` is a float;
      ``coverage`` is ignored in this mode.

    Coverage mode is architecture-portable (the target is a fraction of the
    model's total mass, so the same ``coverage`` value produces
    proportionally-sized exclusion sets on different models). Threshold mode
    is simpler and more predictable when the operator already knows the
    sensitivity score magnitude they consider "too sensitive to quantize"
    for the specific model.

    Args:
        scores: Per-target (node or op-type) sensitivity scores from
            :func:`sensitivity.score` output.
        coverage: Fraction of total sensitivity score mass to leave unquantized (coverage
            mode only). Guidance:

            * ``0.85 - 0.90`` (default): balanced exploration. Recovers the
              majority of the accuracy gap between default QDQ and the FP16
              reference while keeping the exclusion set small enough to
              preserve most of the INT8 latency benefit. For architectures
              with concentrated sensitivity distributions (e.g.,
              ResNet-family with sensitivity clustered in the first
              bottleneck), ``0.80 - 0.85`` may produce equivalent accuracy
              with a smaller exclusion set.
            * ``0.95 - 0.99``: accuracy-critical deployments. Larger
              exclusion set, approaches the FP16 accuracy ceiling, at the
              cost of more Cast boundaries and reduced INT8 latency benefit.
            * ``0.70 - 0.80``: performance-critical deployments. Smaller
              exclusion set, maximizes INT8 coverage for latency at the
              cost of a wider accuracy gap versus the FP16 reference.

        threshold: Absolute sensitivity score cutoff (threshold mode). When
            set, every target with individual sensitivity score strictly
            greater than ``threshold`` is excluded from quantization;
            ``coverage`` is ignored. Set to ``None`` (default) to use
            coverage mode. Guidance is model-dependent because per-target
            sensitivity score magnitudes scale with model complexity: on
            ResNet-50 a value of ``0.005 - 0.02`` picks up
            the load-bearing targets; on CoAtNet-0 or larger models
            ``0.05 - 0.5`` is a similar magnitude in relative terms. Use
            coverage mode if you need portability across models.
        max_nodes: Optional cap on the exclusion set size. Prevents
            long-tail-heavy distributions from producing very large
            exclusion sets that fragment the graph and hurt latency.
            Applied in both modes; whichever limit triggers first stops
            the accumulation.
        min_score_floor: Targets with individual score below this value are
            never included, even if the coverage target has not been
            reached (coverage mode) or the target exceeds ``threshold``
            (threshold mode -- a defensive check).
        near_tie_ratio: If the first-excluded target's sensitivity score is
            at least this fraction of the last-included target's sensitivity
            score, a warning is emitted via ``logger.warning`` recommending
            the operator consider a slightly larger coverage / smaller
            threshold to avoid intra-group precision fragmentation. Default
            0.99 (warn when the first-excluded target's sensitivity score is
            within 1% of the last-included's). Set to ``None`` to disable
            the warning entirely.

    Returns:
        List of target names (from ``scores`` keys), sorted from highest to
        lowest sensitivity score. Pass to
        ``modelopt.onnx.quantization.quantize(..., nodes_to_exclude=...)`` if
        ``scores`` came from per-node granularity, or to
        ``modelopt.onnx.quantization.quantize(..., op_types_to_exclude=...)``
        if it came from per-op-type granularity.
    """
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return []

    if threshold is not None:
        # Threshold mode: pick every target whose sensitivity score strictly
        # exceeds ``threshold``. Iteration order is highest-to-lowest score.
        excluded: list[str] = []
        for name, score in ranked:
            if score <= threshold or score < min_score_floor:
                break
            excluded.append(name)
            if max_nodes is not None and len(excluded) >= max_nodes:
                break
        _warn_near_tie(ranked, excluded, near_tie_ratio, mode="threshold")
        return excluded

    # Coverage mode: pick the largest target set whose cumulative sensitivity
    # score stays at or below ``coverage * total_mass``. Stops BEFORE crossing
    # the requested value, so the actual coverage is <= requested. Guarantees
    # the operator never gets more exclusion than they asked for.
    total = sum(scores.values())
    if total <= 0.0:
        return []
    target = coverage * total

    cumulative = 0.0
    excluded = []
    for name, score in ranked:
        if score < min_score_floor:
            break
        if cumulative + score > target:
            # Adding this target would exceed the requested coverage; stop.
            break
        excluded.append(name)
        cumulative += score
        if max_nodes is not None and len(excluded) >= max_nodes:
            break

    _warn_near_tie(ranked, excluded, near_tie_ratio, mode="coverage")
    return excluded


def _warn_near_tie(
    ranked: list[tuple[str, float]],
    excluded: list[str],
    near_tie_ratio: float | None,
    mode: str,
) -> None:
    """Emit a logger warning if the cut-off between included and excluded is a near-tie.

    A near-tie means the first-excluded target's sensitivity score is at
    least ``near_tie_ratio`` of the last-included target's sensitivity score.
    In that case, the two targets carry nearly equivalent sensitivity signal
    but end up in different precisions (one FP16, one INT8), which can
    produce intra-group fragmentation and unnecessary Cast overhead. The
    operator can widen the coverage or lower the threshold to bring the
    near-tied target into the exclusion set.
    """
    if near_tie_ratio is None:
        return
    if not excluded or len(excluded) >= len(ranked):
        return
    last_included_kl = ranked[len(excluded) - 1][1]
    if last_included_kl <= 0.0:
        return
    first_excluded_name, first_excluded_kl = ranked[len(excluded)]
    ratio = first_excluded_kl / last_included_kl
    if ratio < near_tie_ratio:
        return
    last_included_name = ranked[len(excluded) - 1][0]
    logger.warning(
        f"suggest_exclusion (mode={mode}): near-tie at the exclusion cut-off. "
        f"Last included target '{last_included_name}' has score={last_included_kl:.5f}, "
        f"first excluded target '{first_excluded_name}' has score={first_excluded_kl:.5f} "
        f"({100.0 * ratio:.2f}% of last-included). "
        f"Consider a slightly larger coverage / smaller threshold to include the "
        f"near-tied target and avoid intra-group precision fragmentation."
    )


def summarize_exclusion(
    scores: Mapping[str, float],
    excluded: list[str],
) -> dict:
    """Return a summary dictionary describing an exclusion set.

    Useful for logging or reporting the effect of :func:`suggest_exclusion`
    before feeding the result into ``modelopt.onnx.quantization.quantize``.

    Args:
        scores: The full per-target (node or op-type) sensitivity scores.
        excluded: The list of target names that will be excluded from
            quantization.

    Returns:
        Dict with:

        * ``coverage_pct``: Percentage of total sensitivity score mass
          captured by the exclusion set.
        * ``num_excluded``: Number of targets to exclude from quantization.
        * ``num_previously_quantized``: Total number of quantizable targets
          the primitive probed (i.e., what would have been quantized
          without the exclusion set).
        * ``num_remaining_quantized``: How many targets will still be
          quantized after the exclusion set is applied.
        * ``excluded_mass``: Absolute cumulative sensitivity score
          captured by the exclusion set.
        * ``total_mass``: Sum of sensitivity scores across every probed target.
    """
    total_mass = sum(scores.values())
    excluded_mass = sum(float(scores.get(name, 0.0)) for name in excluded)
    coverage_pct = 100.0 * excluded_mass / total_mass if total_mass > 0.0 else 0.0
    return {
        "coverage_pct": coverage_pct,
        "num_excluded": len(excluded),
        "num_previously_quantized": len(scores),
        "num_remaining_quantized": len(scores) - len(excluded),
        "excluded_mass": excluded_mass,
        "total_mass": total_mass,
    }
