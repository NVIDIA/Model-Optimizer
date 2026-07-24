# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guided CRUD model for named vLLM measurements."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence

if TYPE_CHECKING:
    from modelopt.torch.puzzletron.subblock_stats.measurements import VllmMeasurement

__all__ = ["VllmMeasurementEditor"]


class VllmMeasurementEditor:
    """Ordered named-measurement editor with reference-safe deletion."""

    def __init__(
        self,
        measurements: Optional[Mapping[str, "VllmMeasurement"]] = None,
    ) -> None:
        self._measurements: OrderedDict[str, VllmMeasurement] = OrderedDict(
            measurements or {}
        )

    def measurements(self) -> Mapping[str, "VllmMeasurement"]:
        return OrderedDict(self._measurements)

    def add(self, measurement: "VllmMeasurement") -> None:
        if measurement.measurement_id in self._measurements:
            raise ValueError(f"duplicate vLLM measurement {measurement.measurement_id!r}")
        duplicate = next(
            (
                name
                for name, existing in self._measurements.items()
                if existing.to_dict() == measurement.to_dict()
            ),
            None,
        )
        if duplicate is not None:
            raise ValueError(
                f"vLLM measurement {measurement.measurement_id!r} duplicates {duplicate!r}"
            )
        self._measurements[measurement.measurement_id] = measurement

    def clone(self, source_id: str, target_id: str) -> "VllmMeasurement":
        clone = replace(self._measurements[source_id], measurement_id=target_id)
        self.add(clone)
        return clone

    def edit(self, measurement_id: str, **changes: Any) -> "VllmMeasurement":
        updated = replace(self._measurements[measurement_id], **changes)
        for name, existing in self._measurements.items():
            if name != measurement_id and existing.to_dict() == updated.to_dict():
                raise ValueError(f"vLLM measurement duplicates {name!r}")
        self._measurements[measurement_id] = updated
        return updated

    def delete(self, measurement_id: str, referenced_by: Sequence[str] = ()) -> None:
        if referenced_by:
            raise ValueError(
                f"measurement {measurement_id!r} is referenced by {sorted(referenced_by)}"
            )
        del self._measurements[measurement_id]

    def workloads(self) -> Mapping[str, Mapping[str, int | str]]:
        return OrderedDict(
            (
                name,
                {
                    "workload_id": name,
                    "batch_size": measurement.batch_size,
                    "prefill_seq_len": measurement.prefill_seq_len,
                    "generation_seq_len": measurement.generation_seq_len,
                },
            )
            for name, measurement in self._measurements.items()
        )

    def work_estimate(self, candidate_count: int) -> int:
        return int(candidate_count) * len(self._measurements)

    def to_config(self) -> dict[str, Any]:
        return {
            "measurements": OrderedDict(
                (name, measurement.to_dict())
                for name, measurement in self._measurements.items()
            )
        }
