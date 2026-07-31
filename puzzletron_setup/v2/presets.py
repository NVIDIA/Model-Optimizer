# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intent-based nested defaults for the guided setup-v2 flow."""

from __future__ import annotations

from collections.abc import Mapping  # noqa: TC003 - runtime type-hint introspection
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from puzzletron_setup import SetupError

__all__ = ["QUICK_SETUP_PRESETS", "SetupPreset", "get_setup_preset"]


@dataclass(frozen=True)
class SetupPreset:
    """One guided setup profile and its nested algorithm defaults."""

    name: str
    title: str
    guidance: str
    defaults: Mapping[str, Any]

    @property
    def choice_title(self) -> str:
        """Render the preset name and its selection guidance."""
        return f"{self.title}: {self.guidance}"

    def resolved_defaults(self) -> dict[str, Any]:
        """Return an isolated mutable copy of the nested defaults."""
        return deepcopy(dict(self.defaults))


QUICK_SETUP_PRESETS = (
    SetupPreset(
        name="smoke",
        title="Quick smoke",
        guidance="fastest; verifies the campaign shape with minimal scoring",
        defaults={
            "pruning": {
                "depth_remove": 1,
                "depth_importance_samples": 32,
                "width_importance_samples": 512,
                "sort_sanity": False,
                "width_sanity": False,
                "slicing_sanity": False,
                "replacement_samples": 32,
                "bypass": {
                    "enabled": False,
                    "samples": 64,
                },
            },
            "mip": {
                "goal_value": "90%",
                "num_solutions": 2,
            },
        },
    ),
    SetupPreset(
        name="balanced",
        title="Balanced pruning (recommended)",
        guidance="best first real campaign; useful coverage at moderate cost",
        defaults={
            "pruning": {
                "depth_remove": 4,
                "depth_importance_samples": 128,
                "width_importance_samples": 32768,
                "sort_sanity": False,
                "width_sanity": False,
                "slicing_sanity": False,
                "replacement_samples": 128,
                "bypass": {
                    "enabled": True,
                    "samples": 4096,
                },
            },
            "mip": {
                "goal_value": "75%",
                "num_solutions": 8,
            },
        },
    ),
    SetupPreset(
        name="high-confidence",
        title="High-confidence search",
        guidance="more checks and scoring; choose when extra runtime is acceptable",
        defaults={
            "pruning": {
                "depth_remove": 6,
                "depth_importance_samples": 512,
                "width_importance_samples": 65536,
                "sort_sanity": True,
                "sort_sanity_samples": 512,
                "width_sanity": True,
                "width_sanity_samples": 512,
                "slicing_sanity": True,
                "replacement_samples": 512,
                "bypass": {
                    "enabled": True,
                    "samples": 8192,
                },
            },
            "mip": {
                "goal_value": "70%",
                "num_solutions": 16,
            },
        },
    ),
)


def get_setup_preset(name: str) -> SetupPreset:
    """Return a known setup preset or fail with an actionable error."""
    for preset in QUICK_SETUP_PRESETS:
        if preset.name == name:
            return preset
    choices = ", ".join(preset.name for preset in QUICK_SETUP_PRESETS)
    raise SetupError(f"Unknown setup preset {name!r}; choose one of: {choices}.")
