# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from .identity import stable_hash

__all__ = ["AxisSearchConfig", "SearchSpace", "load_search_space"]


@dataclass(frozen=True, kw_only=True)
class AxisSearchConfig:
    axis_id: str
    enabled: bool = False
    values: tuple[Any, ...] = ()
    min_value: Any | None = None
    max_value: Any | None = None
    step: Any | None = None
    budget_grid: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, axis_id: str, data: dict[str, Any] | None) -> "AxisSearchConfig":
        data = dict(data or {})
        reserved = {"enabled", "values", "min", "max", "step", "budget_grid"}
        return cls(
            axis_id=axis_id,
            enabled=bool(data.get("enabled", False)),
            values=tuple(data.get("values") or ()),
            min_value=data.get("min"),
            max_value=data.get("max"),
            step=data.get("step"),
            budget_grid=tuple(data.get("budget_grid") or ()),
            metadata={key: value for key, value in data.items() if key not in reserved},
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, kw_only=True)
class SearchSpace:
    axes: dict[str, AxisSearchConfig] = field(default_factory=dict)
    layer_filter: dict[str, Any] = field(default_factory=dict)
    subblock_filter: dict[str, Any] = field(default_factory=dict)

    @property
    def identity(self) -> str:
        return stable_hash(self.to_dict(), prefix="search_space")

    @property
    def enabled_axes(self) -> tuple[str, ...]:
        return tuple(axis_id for axis_id, cfg in self.axes.items() if cfg.enabled)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axes": {axis_id: axis.to_dict() for axis_id, axis in sorted(self.axes.items())},
            "layer_filter": self.layer_filter,
            "subblock_filter": self.subblock_filter,
            "identity": self.identity,
        }


def load_search_space(config: dict[str, Any]) -> SearchSpace:
    raw = dict(config.get("search_space") or {})
    axes = {
        axis_id: AxisSearchConfig.from_mapping(axis_id, axis_cfg)
        for axis_id, axis_cfg in sorted(dict(raw.get("axes") or {}).items())
    }
    return SearchSpace(
        axes=axes,
        layer_filter=dict(raw.get("layer_filter") or {}),
        subblock_filter=dict(raw.get("subblock_filter") or {}),
    )
