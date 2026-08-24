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

"""Turn a sensitivity score dictionary into an exclusion list, with optional block-level aggregation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from modelopt.onnx.logging_config import logger

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


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

    Coverage mode (default) picks the largest target set whose cumulative score stays at or
    below ``coverage * total_mass``. Threshold mode (when ``threshold`` is set; ``coverage``
    is then ignored) picks every target whose individual score exceeds ``threshold``.

    Args:
        scores: Per-target sensitivity scores from :func:`sensitivity.score`.
        coverage: Fraction of total sensitivity score mass to leave unquantized. Portable
            across models. ``0.85-0.90`` (default) balances accuracy and INT8 latency benefit;
            ``0.95-0.99`` favors accuracy; ``0.70-0.80`` favors latency.
        threshold: Absolute score cutoff. Model-dependent.
        blocks: Optional ``{group_name: [regex, ...]}``. When set, ranks *groups* rather than
            individual nodes: each node joins at most one group (first-match wins), unmatched
            nodes become singleton groups, and all selection semantics apply to the group
            ranking. The returned exclusion list is the union of member nodes across selected
            groups.
        block_agg: Aggregation for group scores when ``blocks`` is set: ``"sum"`` (default;
            natural with ``coverage``), ``"max"`` (natural with ``threshold``; preserves
            per-node units), or ``"mean"``. Off-diagonal combinations change what ``coverage``
            and ``threshold`` mean in units.
        max_nodes: Optional cap on the number of selected items. Prevents long-tail
            distributions from producing large exclusion sets that fragment the graph.
        min_score_floor: Targets below this score are never included.
        near_tie_ratio: Emit a warning when the first-excluded score is at least this fraction
            of the last-included score (default 0.99). ``None`` disables it.

    Returns:
        Target names sorted highest-to-lowest score. Pass to ``nodes_to_exclude=`` for
        per-node scores (or when ``blocks`` is set) and to ``op_types_to_exclude=`` for
        per-op-type scores.
    """
    if blocks is not None:
        if block_agg not in {"sum", "max", "mean"}:
            raise ValueError(f"block_agg must be 'sum', 'max', or 'mean' (got {block_agg!r})")
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
    """Coverage / threshold selection on any ``{name: score}`` dict.

    Shared between per-node picking and per-group picking (which aggregates per-node scores into
    per-group scores first) so both paths use identical selection semantics.
    """
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    if not ranked:
        return []

    if threshold is not None:
        excluded: list[str] = []
        for name, score in ranked:
            if score <= threshold or score < min_score_floor:
                break
            excluded.append(name)
            if max_nodes is not None and len(excluded) >= max_nodes:
                break
        _warn_near_tie(ranked, excluded, near_tie_ratio, mode="threshold")
        return excluded

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
    - A node matching any regex in ``blocks[name]`` joins group ``name``.
    - First-match wins across the iteration order of ``blocks``, so callers that mix depths list
    more-specific groups earlier.
    - Nodes matching no pattern become their own singleton group named after themselves so
    architecturally-important standalone nodes compete on equal footing with multi-node blocks.
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
    """Aggregate per-node scores into per-group scores."""
    if block_agg == "sum":
        return {g: sum(scores[n] for n in members) for g, members in groups.items()}
    if block_agg == "max":
        return {g: max(scores[n] for n in members) for g, members in groups.items()}
    # mean
    return {g: sum(scores[n] for n in members) / len(members) for g, members in groups.items()}


def _warn_near_tie(
    ranked: list[tuple[str, float]],
    excluded: list[str],
    near_tie_ratio: float | None,
    mode: str,
) -> None:
    """Warn if the last-included and first-excluded scores are within ``near_tie_ratio``.

    When the two boundary targets carry nearly equivalent sensitivity but land in different
    precisions (one FP16, one INT8), the resulting Cast boundary produces intra-group
    fragmentation. The warning prompts widening ``coverage`` or narrowing ``threshold``.
    """
    if near_tie_ratio is None:
        return
    if not excluded or len(excluded) >= len(ranked):
        return
    last_included_score = ranked[len(excluded) - 1][1]
    if last_included_score <= 0.0:
        return
    first_excluded_name, first_excluded_score = ranked[len(excluded)]
    ratio = first_excluded_score / last_included_score
    if ratio < near_tie_ratio:
        return
    last_included_name = ranked[len(excluded) - 1][0]
    logger.warning(
        f"suggest_exclusion (mode={mode}): near-tie at the exclusion cut-off. "
        f"Last included target '{last_included_name}' has score={last_included_score:.5f}, "
        f"first excluded target '{first_excluded_name}' has score={first_excluded_score:.5f} "
        f"({100.0 * ratio:.2f}% of last-included). "
        f"Consider a slightly larger coverage / smaller threshold to include the "
        f"near-tied target and avoid intra-group precision fragmentation."
    )


def summarize_exclusion(
    scores: Mapping[str, float],
    excluded: list[str],
) -> dict:
    """Return a summary dict describing an exclusion set.

    Useful for logging the effect of :func:`suggest_exclusion` before feeding the result into
    :func:`modelopt.onnx.quantization.quantize`.

    Args:
        scores: The full per-target sensitivity scores.
        excluded: The list of target names that will be excluded from quantization.

    Returns:
        Dict with ``coverage_pct`` (percentage of total mass captured by the exclusion set),
        ``num_excluded``, ``num_previously_quantized``, ``num_remaining_quantized``,
        ``excluded_mass`` (absolute cumulative score), and ``total_mass`` (sum across all
        probed targets).
    """
    total_mass = sum(scores.values())
    excluded_mass = sum(scores.get(name, 0.0) for name in excluded)
    coverage_pct = 100.0 * excluded_mass / total_mass if total_mass > 0.0 else 0.0
    return {
        "coverage_pct": coverage_pct,
        "num_excluded": len(excluded),
        "num_previously_quantized": len(scores),
        "num_remaining_quantized": len(scores) - len(excluded),
        "excluded_mass": excluded_mass,
        "total_mass": total_mass,
    }
