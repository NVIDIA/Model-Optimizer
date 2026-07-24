# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime application helpers for named vLLM measurement contracts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from ..orchestration.vllm_measurements import (
    VllmMeasurement,
    normalize_vllm_measurements,
)

__all__ = [
    "VllmMeasurement",
    "apply_vllm_measurement",
    "measurement_workload",
    "normalize_vllm_measurements",
]


def apply_vllm_measurement(
    config: Mapping[str, Any],
    measurement: VllmMeasurement | str,
) -> dict[str, Any]:
    """Return an isolated runtime config for exactly one measurement."""

    selected = (
        normalize_vllm_measurements(config)[measurement]
        if isinstance(measurement, str)
        else measurement
    )
    rendered = deepcopy(dict(config))
    vllm = rendered.setdefault("vllm_stats", {})
    vllm["batch_sizes"] = [selected.batch_size]
    vllm["prefill_seq_len"] = selected.prefill_seq_len
    vllm["generation_seq_len"] = selected.generation_seq_len
    vllm["runtime_stats"] = deepcopy(dict(selected.runtime_stats))
    vllm["runtime_stats"]["max_num_seqs"] = selected.max_num_seqs
    vllm["runtime_stats"]["granularity"] = selected.granularity
    if selected.model_hidden_sizes:
        vllm["model_hidden_sizes"] = list(selected.model_hidden_sizes)
    vllm["subblock_stats_filename"] = str(selected.relative_stats_path)
    vllm["active_measurement"] = {
        "id": selected.measurement_id,
        "identity": selected.identity,
    }
    mip = rendered.get("mip")
    if isinstance(mip, dict):
        mip["subblock_stats_args"] = measurement_workload(selected)
    return rendered


def measurement_workload(measurement: VllmMeasurement) -> dict[str, int | str]:
    """Project one measurement to the exact MIP selector fields."""

    return {
        "workload_id": measurement.measurement_id,
        "batch_size": measurement.batch_size,
        "prefill_seq_len": measurement.prefill_seq_len,
        "generation_seq_len": measurement.generation_seq_len,
    }
