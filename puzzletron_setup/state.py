# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Versioned, interruption-safe answer persistence for Puzzletron setup."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import yaml

from . import SetupError

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["SECTIONS", "AnswerState"]

SCHEMA_VERSION = 1
WIZARD_VERSION = "1"
SECTIONS = (
    "model",
    "data",
    "pruning",
    "runtime",
    "mip",
    "post_mip",
    "infrastructure",
    "output",
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _state_path(path: Path) -> Path:
    path = path.expanduser().resolve()
    return path / "answers.yaml" if path.is_dir() or path.suffix == "" else path


@dataclass
class AnswerState:
    """Mutable answer state whose every change is written atomically."""

    path: Path
    payload: dict[str, Any]

    @classmethod
    def start(cls, campaign_dir: Path, *, detailed: bool) -> AnswerState:
        """Create fresh answer state in a new or empty campaign directory."""
        campaign_dir = campaign_dir.expanduser().resolve()
        if campaign_dir.exists() and any(campaign_dir.iterdir()):
            raise SetupError(
                f"Campaign directory is not empty: {campaign_dir}. "
                "Choose a new directory or use --resume."
            )
        campaign_dir.mkdir(parents=True, exist_ok=True)
        state = cls(
            path=campaign_dir / "answers.yaml",
            payload={
                "schema_version": SCHEMA_VERSION,
                "wizard_version": WIZARD_VERSION,
                "detailed": bool(detailed),
                "completed_section": None,
                "model": {},
                "inventory": {},
                "answers": {},
                "updated_at": _timestamp(),
            },
        )
        state.save()
        return state

    @classmethod
    def resume(cls, path: Path) -> AnswerState:
        """Load explicit resume state from a campaign directory or YAML path."""
        state_path = _state_path(path)
        if not state_path.is_file():
            raise SetupError(f"Resume state does not exist: {state_path}")
        try:
            payload = yaml.safe_load(state_path.read_text())
        except (OSError, yaml.YAMLError) as error:
            raise SetupError(f"Cannot read resume state {state_path}: {error}") from error
        if not isinstance(payload, dict):
            raise SetupError(f"Resume state must be a YAML mapping: {state_path}")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SetupError(
                f"Unsupported answer schema {payload.get('schema_version')!r}; "
                f"this wizard expects {SCHEMA_VERSION}."
            )
        payload.setdefault("answers", {})
        payload.setdefault("model", {})
        payload.setdefault("inventory", {})
        return cls(path=state_path, payload=payload)

    @property
    def detailed(self) -> bool:
        """Whether this setup session exposes advanced questions."""
        return bool(self.payload.get("detailed", False))

    def section(self, name: str) -> dict[str, Any]:
        """Return a copy of one normalized answer section."""
        value = self.payload.get("answers", {}).get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def record(self, section: str, key: str, value: Any) -> None:
        """Record one answer and atomically persist it immediately."""
        if section not in SECTIONS:
            raise ValueError(f"Unknown answer section: {section}")
        answers = self.payload.setdefault("answers", {})
        section_payload = answers.setdefault(section, {})
        section_payload[key] = value
        self.payload["completed_section"] = section
        self.save()

    def record_many(self, section: str, values: Mapping[str, Any]) -> None:
        """Record a completed group while still using one atomic replacement."""
        if section not in SECTIONS:
            raise ValueError(f"Unknown answer section: {section}")
        self.payload.setdefault("answers", {}).setdefault(section, {}).update(values)
        self.payload["completed_section"] = section
        self.save()

    def set_model(self, model: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
        """Persist model identity and normalized inventory."""
        self.payload["model"] = dict(model)
        self.payload["inventory"] = dict(inventory)
        self.payload["completed_section"] = "model"
        self.save()

    def invalidate_after(self, section: str) -> None:
        """Discard answers derived from sections after the named boundary."""
        if section not in SECTIONS:
            raise ValueError(f"Unknown answer section: {section}")
        boundary = SECTIONS.index(section)
        answers = self.payload.setdefault("answers", {})
        for later in SECTIONS[boundary + 1 :]:
            answers.pop(later, None)
        self.payload["completed_section"] = section
        self.save()

    def save(self) -> None:
        """Flush state to a sibling temporary file and atomically replace it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload["updated_at"] = _timestamp()
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w") as stream:
                yaml.safe_dump(_plain(self.payload), stream, sort_keys=False, width=100)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            raise SetupError(f"Cannot save setup answers to {self.path}: {error}") from error
