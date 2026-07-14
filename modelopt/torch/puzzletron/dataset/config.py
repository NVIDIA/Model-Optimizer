# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict public data configuration for native Puzzletron stages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .batch import DataLayout, Modality

__all__ = ["PackingSpec", "PuzzletronDataSpec"]


@dataclass(frozen=True)
class PackingSpec:
    pack_size: int
    packing_ratio: float = 1.0
    drop_long_samples: bool = True

    def __post_init__(self) -> None:
        if self.pack_size <= 0:
            raise ValueError(f"data.packing.pack_size must be positive, got {self.pack_size}")
        if not 0.0 < self.packing_ratio <= 1.0:
            raise ValueError(
                "data.packing.packing_ratio must be in (0, 1], "
                f"got {self.packing_ratio}"
            )


@dataclass(frozen=True)
class PuzzletronDataSpec:
    modality: Modality
    layout: DataLayout
    max_sample_length: int
    packing: PackingSpec | None = None

    def __post_init__(self) -> None:
        if self.max_sample_length <= 0:
            raise ValueError(
                f"data.max_sample_length must be positive, got {self.max_sample_length}"
            )
        if self.layout is DataLayout.PACKED_VARLEN and self.packing is None:
            raise ValueError("layout=packed_varlen requires a data.packing mapping")
        if self.layout is not DataLayout.PACKED_VARLEN and self.packing is not None:
            raise ValueError("data.packing is only valid with layout=packed_varlen")
        if self.packing is not None and self.max_sample_length > self.packing.pack_size:
            raise ValueError(
                "data.max_sample_length cannot exceed data.packing.pack_size when long samples "
                "are packed"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "PuzzletronDataSpec":
        raw = dict(raw)
        if "varlen" in raw:
            raise ValueError(
                "data.varlen was removed because it did not preserve attention boundaries. "
                "Use data.layout=fixed, padded_varlen, or packed_varlen; for the old true "
                "behavior use data.layout=packed_varlen and configure data.packing."
            )
        try:
            modality = Modality(raw["modality"])
            layout = DataLayout(raw["layout"])
            max_sample_length = int(raw["max_sample_length"])
        except KeyError as exc:
            raise ValueError(f"canonical data config is missing data.{exc.args[0]}") from exc
        packing_raw = raw.get("packing")
        packing = None
        if packing_raw is not None:
            packing_raw = dict(packing_raw)
            if "pack_size" not in packing_raw:
                raise ValueError("data.packing requires pack_size")
            packing = PackingSpec(
                pack_size=int(packing_raw["pack_size"]),
                packing_ratio=float(packing_raw.get("packing_ratio", 1.0)),
                drop_long_samples=bool(packing_raw.get("drop_long_samples", True)),
            )
        return cls(
            modality=modality,
            layout=layout,
            max_sample_length=max_sample_length,
            packing=packing,
        )

    @property
    def sequence_length(self) -> int:
        return self.packing.pack_size if self.packing is not None else self.max_sample_length

    @property
    def legacy_varlen(self) -> bool:
        return self.layout is DataLayout.PACKED_VARLEN
