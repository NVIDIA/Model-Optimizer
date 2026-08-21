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

The picker also supports **block-aware grouping** via the ``blocks`` argument:
per-node scores can be aggregated into user-defined architectural groups
(transformer blocks, residual blocks, MBConv stages, ...) and the picker's
coverage / threshold semantics apply to the group ranking rather than
individual nodes. This is useful for transformer / attention-heavy
architectures where per-node picking leaves precision boundaries scrambled
inside the affected blocks and softmax numerics degrade.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Literal

from modelopt.onnx.logging_config import logger


def suggest_exclusion(
    scores: Mapping[str, float],
    coverage: float = 0.90,
    *,
    threshold: float | None = None,
    blocks: Mapping[str, Sequence[str | re.Pattern]] | None = None,
    block_agg: Literal["sum", "max", "mean"] = "sum",
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
        blocks: Optional mapping from group name to a list of regex patterns
            (either compiled ``re.Pattern`` objects or plain regex strings)
            that match node paths. When provided, the picker ranks *groups*
            rather than individual nodes: each node in ``scores`` is
            assigned to at most one group (first-match wins across the
            ``blocks`` dict); nodes matching no pattern become their own
            singleton group named after themselves. Group scores are
            computed via ``block_agg``, and coverage / threshold /
            ``max_nodes`` / ``near_tie_ratio`` apply identically to the
            group ranking. The returned exclusion list is the union of
            member node names across the selected groups. Default ``None``
            -- every node is its own singleton group, equivalent to
            per-node picking.
        block_agg: Aggregation function used to compute a group's score
            from its members' individual scores when ``blocks`` is set.
            One of ``"sum"``, ``"max"``, ``"mean"``. Natural pairings with
            the two policy modes:

            * ``block_agg="sum"`` with **coverage** (recommended default):
              identical "fraction of total KL mass" semantic as per-node
              coverage, because summing group sums equals summing all node
              scores. Portable across granularity choices.
            * ``block_agg="max"`` with **threshold**: same units as
              per-node threshold (excludes any group whose peak-node score
              exceeds the cutoff). Preserves operator intuition when
              transferring per-node threshold guidance to the block level.
            * Other combinations are valid but change what ``coverage``
              and ``threshold`` mean in units. Under ``block_agg="max"``
              coverage counts fraction-of-total-group-max-scores (not
              fraction-of-total-KL-mass). Under ``block_agg="sum"``
              threshold operates in summed-KL units per group, so
              per-node threshold values must be scaled up to be
              meaningful. Ignored when ``blocks`` is ``None``.
        max_nodes: Optional cap on the exclusion set size. When ``blocks``
            is set, this caps the number of *groups* included in the
            aggregate ranking before expansion; when ``blocks`` is ``None``,
            it caps the number of individual targets. Prevents long-tail-
            heavy distributions from producing very large exclusion sets
            that fragment the graph and hurt latency. Applied in both
            modes; whichever limit triggers first stops the accumulation.
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
        if it came from per-op-type granularity. When ``blocks`` is set the
        returned list is always suitable for ``nodes_to_exclude=`` because
        it is the union of member node names across the selected groups.
    """
    if blocks is not None:
        if block_agg not in {"sum", "max", "mean"}:
            raise ValueError(
                f"block_agg must be 'sum', 'max', or 'mean' (got {block_agg!r})"
            )
        groups = _assign_groups(scores, blocks)
        group_scores = _aggregate_group_scores(scores, groups, block_agg)
        selected_groups = _pick_from_scores(
            group_scores,
            coverage=coverage,
            threshold=threshold,
            max_nodes=max_nodes,
            min_score_floor=min_score_floor,
            near_tie_ratio=near_tie_ratio,
        )
        return [n for g in selected_groups for n in groups[g]]

    return _pick_from_scores(
        scores,
        coverage=coverage,
        threshold=threshold,
        max_nodes=max_nodes,
        min_score_floor=min_score_floor,
        near_tie_ratio=near_tie_ratio,
    )


def _pick_from_scores(
    scores: Mapping[str, float],
    *,
    coverage: float,
    threshold: float | None,
    max_nodes: int | None,
    min_score_floor: float,
    near_tie_ratio: float | None,
) -> list[str]:
    """Core picker: coverage / threshold selection on any ``{name: score}`` dict.

    Called for per-node picking (from :func:`suggest_exclusion` with
    ``blocks=None``) and for per-group picking (from :func:`suggest_exclusion`
    with ``blocks`` set, after aggregating per-node scores into per-group
    scores). Extracted so both paths share identical coverage / threshold /
    near-tie / ``max_nodes`` / ``min_score_floor`` semantics.
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


def _assign_groups(
    scores: Mapping[str, float],
    blocks: Mapping[str, Sequence[str | re.Pattern]],
) -> dict[str, list[str]]:
    """Assign each node in ``scores`` to at most one group.

    Rules:

    * A node matching any regex in ``blocks[name]`` joins group ``name``.
    * First-match wins across the iteration order of ``blocks`` -- callers
      that need mixed-depth grouping should list more-specific groups
      earlier.
    * Nodes matching no pattern become their own singleton group named
      after themselves, so architecturally-important standalone nodes
      compete for exclusion on equal footing with multi-node blocks.
    """
    compiled = {
        gname: [re.compile(p) if isinstance(p, str) else p for p in patterns]
        for gname, patterns in blocks.items()
    }
    groups: dict[str, list[str]] = {}
    for node_name in scores:
        matched: str | None = None
        for gname, pats in compiled.items():
            if any(pat.match(node_name) for pat in pats):
                matched = gname
                break
        key = matched if matched is not None else node_name
        groups.setdefault(key, []).append(node_name)
    return groups


def _aggregate_group_scores(
    scores: Mapping[str, float],
    groups: Mapping[str, Sequence[str]],
    block_agg: Literal["sum", "max", "mean"],
) -> dict[str, float]:
    """Aggregate per-node scores into per-group scores using ``block_agg``."""
    if block_agg == "sum":
        return {g: sum(scores[n] for n in members) for g, members in groups.items()}
    if block_agg == "max":
        return {g: max(scores[n] for n in members) for g, members in groups.items()}
    # mean
    return {
        g: (sum(scores[n] for n in members) / len(members)) if members else 0.0
        for g, members in groups.items()
    }


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
