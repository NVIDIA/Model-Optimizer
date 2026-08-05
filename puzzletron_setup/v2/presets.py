# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Intent metadata and model-family defaults for guided setup v2."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from puzzletron_setup import SetupError

from .defaults import validate_defaults

__all__ = ["QUICK_SETUP_PRESETS", "SetupPreset", "get_setup_preset"]

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_FAMILY_DEFAULTS_FILENAME = "setup_v2_defaults.yaml"


@dataclass(frozen=True)
class SetupPreset:
    """One guided setup profile whose tuning is owned by each model family."""

    name: str
    title: str
    guidance: str

    @property
    def choice_title(self) -> str:
        """Render the preset name and its selection guidance."""
        return f"{self.title}: {self.guidance}"

    def resolved_defaults(self, family_config: str | Path) -> dict[str, Any]:
        """Load an isolated copy of this profile's model-family defaults."""
        family_path = Path(family_config)
        if not family_path.is_absolute():
            family_path = _REPOSITORY_ROOT / family_path
        defaults_path = family_path.with_name(_FAMILY_DEFAULTS_FILENAME)
        try:
            payload = yaml.safe_load(defaults_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise SetupError(
                f"Cannot read guided setup defaults for model family at {defaults_path}: {error}"
            ) from error
        if not isinstance(payload, Mapping):
            raise SetupError(f"Guided setup defaults must contain a YAML mapping: {defaults_path}")
        unknown_fields = set(payload) - {"schema_version", "profiles"}
        if unknown_fields:
            fields = ", ".join(sorted(str(field) for field in unknown_fields))
            raise SetupError(f"Unknown guided setup defaults fields in {defaults_path}: {fields}")
        if payload.get("schema_version") != 1:
            raise SetupError(
                f"Unsupported guided setup defaults schema "
                f"{payload.get('schema_version')!r} in {defaults_path}; expected 1."
            )
        profiles = payload.get("profiles")
        if not isinstance(profiles, Mapping):
            raise SetupError(f"Guided setup defaults profiles must be a mapping: {defaults_path}")
        defaults = profiles.get(self.name)
        if not isinstance(defaults, Mapping):
            raise SetupError(
                f"Guided setup profile {self.name!r} is not configured in {defaults_path}."
            )
        try:
            validated = validate_defaults({"schema_version": 1, **dict(defaults)})
        except SetupError as error:
            raise SetupError(
                f"Invalid guided setup profile {self.name!r} in {defaults_path}: {error}"
            ) from error
        validated.pop("schema_version")
        return deepcopy(validated)


QUICK_SETUP_PRESETS = (
    SetupPreset(
        name="smoke",
        title="Quick smoke",
        guidance="fastest; verifies the campaign shape with minimal scoring",
    ),
    SetupPreset(
        name="balanced",
        title="Balanced pruning (recommended)",
        guidance="best first real campaign; useful coverage at moderate cost",
    ),
    SetupPreset(
        name="high-confidence",
        title="High-confidence search",
        guidance="more checks and scoring; choose when extra runtime is acceptable",
    ),
)


def get_setup_preset(name: str) -> SetupPreset:
    """Return a known setup preset or fail with an actionable error."""
    for preset in QUICK_SETUP_PRESETS:
        if preset.name == name:
            return preset
    choices = ", ".join(preset.name for preset in QUICK_SETUP_PRESETS)
    raise SetupError(f"Unknown setup preset {name!r}; choose one of: {choices}.")
