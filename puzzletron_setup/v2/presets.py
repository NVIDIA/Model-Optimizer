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
_SUPPORTED_SCHEMA_VERSIONS = {1, 2}
_MODEL_OVERRIDE_FIELDS = {"match", "profiles"}
_MODEL_MATCH_FIELDS = {"facts", "moe", "num_layers", "num_sublayers"}


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recursive copy of ``base`` with ``overlay`` applied."""
    merged = deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _matches_inventory(expected: Mapping[str, Any], inventory: Any) -> bool:
    """Return whether an inventory contains every value in a declarative selector."""
    for key, expected_value in expected.items():
        if isinstance(inventory, Mapping):
            if key not in inventory:
                return False
            actual_value = inventory[key]
        else:
            if not hasattr(inventory, key):
                return False
            actual_value = getattr(inventory, key)
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping) or not _matches_inventory(
                expected_value, actual_value
            ):
                return False
        elif actual_value != expected_value:
            return False
    return True


def _validated_profile(defaults: Any, *, profile: str, path: Path) -> dict[str, Any]:
    if not isinstance(defaults, Mapping):
        raise SetupError(f"Guided setup profile {profile!r} must be a mapping in {path}.")
    try:
        validated = validate_defaults({"schema_version": 1, **dict(defaults)})
    except SetupError as error:
        raise SetupError(f"Invalid guided setup profile {profile!r} in {path}: {error}") from error
    validated.pop("schema_version")
    return validated


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

    def resolved_defaults(
        self,
        family_config: str | Path,
        model_inventory: Any | None = None,
    ) -> dict[str, Any]:
        """Load family defaults and apply a matching model-specific overlay."""
        family_defaults, model_defaults = self.resolved_default_layers(
            family_config,
            model_inventory,
        )
        return _deep_merge(family_defaults, model_defaults)

    def resolved_default_layers(
        self,
        family_config: str | Path,
        model_inventory: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load separate family and matching model layers for provenance."""
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
        schema_version = payload.get("schema_version")
        unknown_fields = set(payload) - {"schema_version", "profiles", "model_overrides"}
        if unknown_fields:
            fields = ", ".join(sorted(str(field) for field in unknown_fields))
            raise SetupError(f"Unknown guided setup defaults fields in {defaults_path}: {fields}")
        if schema_version not in _SUPPORTED_SCHEMA_VERSIONS:
            raise SetupError(
                f"Unsupported guided setup defaults schema "
                f"{schema_version!r} in {defaults_path}; expected 1 or 2."
            )
        if schema_version == 1 and "model_overrides" in payload:
            raise SetupError(f"Guided setup model overrides require schema 2 in {defaults_path}.")
        profiles = payload.get("profiles")
        if not isinstance(profiles, Mapping):
            raise SetupError(f"Guided setup defaults profiles must be a mapping: {defaults_path}")
        defaults = profiles.get(self.name)
        if not isinstance(defaults, Mapping):
            raise SetupError(
                f"Guided setup profile {self.name!r} is not configured in {defaults_path}."
            )
        resolved = _validated_profile(defaults, profile=self.name, path=defaults_path)

        model_overrides = payload.get("model_overrides", {})
        if not isinstance(model_overrides, Mapping):
            raise SetupError(f"Guided setup model overrides must be a mapping: {defaults_path}")
        matches = []
        for model_name, model_override in model_overrides.items():
            if not isinstance(model_override, Mapping):
                raise SetupError(
                    f"Guided setup model override {model_name!r} must be a mapping "
                    f"in {defaults_path}."
                )
            unknown_model_fields = set(model_override) - _MODEL_OVERRIDE_FIELDS
            if unknown_model_fields:
                fields = ", ".join(sorted(str(field) for field in unknown_model_fields))
                raise SetupError(
                    f"Unknown fields for guided setup model override {model_name!r} "
                    f"in {defaults_path}: {fields}"
                )
            selector = model_override.get("match")
            if not isinstance(selector, Mapping) or not selector:
                raise SetupError(
                    f"Guided setup model override {model_name!r} requires a non-empty "
                    f"match mapping in {defaults_path}."
                )
            unknown_match_fields = set(selector) - _MODEL_MATCH_FIELDS
            if unknown_match_fields:
                fields = ", ".join(sorted(str(field) for field in unknown_match_fields))
                raise SetupError(
                    f"Unknown match fields for guided setup model override "
                    f"{model_name!r} in {defaults_path}: {fields}"
                )
            override_profiles = model_override.get("profiles")
            if not isinstance(override_profiles, Mapping):
                raise SetupError(
                    f"Guided setup model override {model_name!r} profiles must be a "
                    f"mapping in {defaults_path}."
                )
            known_profiles = {preset.name for preset in QUICK_SETUP_PRESETS}
            unknown_profiles = set(override_profiles) - known_profiles
            if unknown_profiles:
                profiles_list = ", ".join(sorted(str(profile) for profile in unknown_profiles))
                raise SetupError(
                    f"Unknown profiles for guided setup model override {model_name!r} "
                    f"in {defaults_path}: {profiles_list}"
                )
            validated_overrides = {
                profile: _validated_profile(profile_defaults, profile=profile, path=defaults_path)
                for profile, profile_defaults in override_profiles.items()
            }
            if model_inventory is not None and _matches_inventory(selector, model_inventory):
                matches.append((str(model_name), validated_overrides))

        if len(matches) > 1:
            names = ", ".join(name for name, _ in matches)
            raise SetupError(
                f"Model inventory matches multiple guided setup overrides in "
                f"{defaults_path}: {names}."
            )
        model_defaults = matches[0][1].get(self.name, {}) if matches else {}
        return resolved, model_defaults


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
