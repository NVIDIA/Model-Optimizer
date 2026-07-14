# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Width/depth scenario identity and artifact layout."""

from dataclasses import dataclass
from pathlib import Path

from .identity import stable_hash

__all__ = ["ScenarioKey"]


@dataclass(frozen=True)
class ScenarioKey:
    hidden_width: int
    removed_sublayers: int

    def __post_init__(self) -> None:
        if self.hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if self.removed_sublayers < 0:
            raise ValueError("removed_sublayers cannot be negative")

    @property
    def identity(self) -> str:
        return stable_hash(
            {
                "hidden_width": self.hidden_width,
                "removed_sublayers": self.removed_sublayers,
            },
            prefix="scenario",
        )

    @property
    def relative_path(self) -> Path:
        return (
            Path("scenarios")
            / f"width-{self.hidden_width:04d}"
            / f"depth-{self.removed_sublayers:02d}"
        )

    def path_under(self, root: str | Path) -> Path:
        return Path(root) / self.relative_path
