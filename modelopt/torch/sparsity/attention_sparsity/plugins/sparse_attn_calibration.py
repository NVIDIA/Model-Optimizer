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

"""vLLM-free helpers for skip-softmax calibration through a serving engine.

The serving adapters (``plugins/vllm.py``) record **raw per-threshold tile
counts** per scheduled request. These helpers merge those counts — across the
layers of one rank and across tensor-parallel ranks — then fit the exponential
threshold model and build the canonical ``sparse_attention_config`` block.

Counts are additive, so aggregation is a plain sum: every layer of a rank and
every TP rank observes the same launches in the same order (TP ranks each see
their head shard), which makes records align by index within a phase. Fitting
happens once, per phase, on the globally merged counts — never per rank
(head-sharded counts are incomplete) and never by averaging independently
fitted coefficients (the fit is nonlinear).

Everything here operates on plain Python data and is unit-testable without
vLLM installed.
"""

from typing import Any

import modelopt

__all__ = [
    "DEFAULT_THRESHOLD_TRIALS",
    "build_sparse_attention_config",
    "fit_from_counts",
    "merge_count_records",
    "merge_phase_counts",
    "split_records_by_phase",
    "stats_from_counts",
]

# Default threshold sweep — should span sparsities from ~10% to ~95%.
DEFAULT_THRESHOLD_TRIALS = [
    1e-4,
    1e-3,
    5e-3,
    1e-2,
    3e-2,
    5e-2,
    1e-1,
    2e-1,
    3e-1,
    5e-1,
    7e-1,
    9e-1,
]

_PHASES = ("prefill", "decode")


def split_records_by_phase(records: list[dict]) -> dict[str, list[dict]]:
    """Group one impl's ordered calibration records by phase, preserving order."""
    per_phase: dict[str, list[dict]] = {phase: [] for phase in _PHASES}
    for record in records:
        per_phase.setdefault(record["phase"], []).append(record)
    return per_phase


def merge_count_records(sources: list[list[dict]]) -> list[dict]:
    """Element-wise sum aligned raw-count records from multiple sources.

    ``sources`` is a list over sources — the layers of one rank, or the
    already layer-merged records of each TP rank — where each source is an
    ordered list of ``{"sample_length", "total_tiles", "skipped_tiles"}``
    records for one phase. All sources observe the same launches in the same
    order, so records align by index; tile counts are additive across both
    layers and head-sharded TP ranks.
    """
    sources = [source for source in sources if source]
    if not sources:
        return []
    num_samples = min(len(source) for source in sources)
    merged = []
    for i in range(num_samples):
        base = sources[0][i]
        total = [0] * len(base["total_tiles"])
        skipped = [0] * len(base["skipped_tiles"])
        for source in sources:
            record = source[i]
            if record["sample_length"] != base["sample_length"]:
                raise ValueError(
                    "Misaligned calibration records: sample lengths differ across "
                    f"sources at index {i} ({record['sample_length']} vs "
                    f"{base['sample_length']})"
                )
            total = [a + b for a, b in zip(total, record["total_tiles"])]
            skipped = [a + b for a, b in zip(skipped, record["skipped_tiles"])]
        merged.append(
            {
                "sample_length": base["sample_length"],
                "total_tiles": total,
                "skipped_tiles": skipped,
            }
        )
    return merged


def merge_phase_counts(rank_counts: list[dict[str, list[dict]]]) -> dict[str, list[dict]]:
    """Merge per-phase raw-count records collected from every TP rank.

    ``rank_counts`` is the list of per-rank results (one
    ``{"prefill": [...], "decode": [...]}`` dict per rank, as returned by
    ``collect_calibration_counts``). Use ALL ranks: with tensor parallelism
    each rank only measures its attention-head shard, so any single rank's
    counts are incomplete.
    """
    phases = {phase for rank in rank_counts for phase in rank}
    return {
        phase: merge_count_records([rank.get(phase, []) for rank in rank_counts])
        for phase in phases
    }


def stats_from_counts(count_records: list[dict]) -> list[dict]:
    """Convert merged raw-count records into per-sample sparsity-ratio stats.

    Returns ``{"sample_length", "sparsity"}`` records in the shape
    :meth:`DynamicThresholdCalibrator.calibrate_from_stats` consumes.
    """
    stats = []
    for record in count_records:
        sparsity = [
            (skipped / total if total else 0.0)
            for skipped, total in zip(record["skipped_tiles"], record["total_tiles"])
        ]
        stats.append({"sample_length": record["sample_length"], "sparsity": sparsity})
    return stats


def fit_from_counts(
    per_phase_counts: dict[str, list[dict]],
    threshold_trials: list[float],
    *,
    fit_logspace: bool = False,
) -> dict[str, dict[str, float]]:
    """Fit the exponential skip-softmax model from globally merged counts.

    Reuses :class:`DynamicThresholdCalibrator` so vLLM-calibrated ``(a, b)``
    are identical in form to the HF path and export unchanged via
    ``threshold_scale_factor``. One fit per phase, on counts already merged
    across all TP ranks and layers.

    Returns:
        ``{phase: {"a", "b", "min_observed_sparsity", "max_observed_sparsity"}}``
        for each phase that produced a valid fit.
    """
    from ..calibration.calibrator import DynamicThresholdCalibrator

    calibration_params: dict[str, dict[str, float]] = {}
    for phase, records in per_phase_counts.items():
        if not records:
            continue
        calibrator = DynamicThresholdCalibrator(
            threshold_trials=list(threshold_trials), fit_logspace=fit_logspace
        )
        result = calibrator.calibrate_from_stats(stats_from_counts(records), phase=phase)
        if "a" in result and "b" in result:
            params = {"a": result["a"], "b": result["b"]}
            for key in ("min_observed_sparsity", "max_observed_sparsity"):
                if key in result:
                    params[key] = result[key]
            calibration_params[phase] = params
    return calibration_params


def _normalize_target_sparsity(target_sparsity: dict[str, float] | float) -> dict[str, float]:
    if isinstance(target_sparsity, (int, float)):
        return {phase: float(target_sparsity) for phase in _PHASES}
    return {phase: float(target_sparsity.get(phase, 0.5)) for phase in _PHASES}


def build_sparse_attention_config(
    calibration_params: dict[str, dict[str, float]],
    target_sparsity: dict[str, float] | float = 0.5,
    *,
    existing_config: dict | None = None,
) -> dict[str, Any]:
    """Build the canonical ``sparse_attention_config`` block for a checkpoint.

    Emits the same schema as
    ``modelopt.torch.sparsity.attention_sparsity.conversion.export_sparse_attention_config``
    — a ``config_groups`` entry with ``algorithm: skip_softmax`` holding the
    group-local ``threshold_scale_factor`` and ``target_sparsity`` — so
    ``load_from_checkpoint_metadata`` (the serving loader) round-trips it
    without changes.

    Non-skip groups from ``existing_config`` (e.g. exported N:M
    ``sparse_softmax`` metadata) are preserved after the skip group; an
    existing ``skip_softmax`` group is replaced by the new calibration.
    """
    threshold_scale_factor: dict[str, Any] = {"formula": "a * exp(b * target_sparsity)"}
    for phase in _PHASES:
        if phase in calibration_params:
            threshold_scale_factor[phase] = {
                "a": float(calibration_params[phase]["a"]),
                "b": float(calibration_params[phase]["b"]),
            }

    skip_group: dict[str, Any] = {
        "algorithm": "skip_softmax",
        "targets": ["Attention"],
        "threshold_scale_factor": threshold_scale_factor,
        "target_sparsity": _normalize_target_sparsity(target_sparsity),
    }

    config_groups: dict[str, Any] = {"group_0": skip_group}
    existing_groups = (existing_config or {}).get("config_groups")
    if isinstance(existing_groups, dict):
        preserved = [
            group
            for group in existing_groups.values()
            if isinstance(group, dict) and group.get("algorithm") != "skip_softmax"
        ]
        for idx, group in enumerate(preserved, start=1):
            config_groups[f"group_{idx}"] = group

    return {
        "config_groups": config_groups,
        "producer": {"name": "modelopt", "version": modelopt.__version__},
    }
