# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Atomic authoring state and dependency-aware invalidation for setup v2."""

from __future__ import annotations

import os
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from puzzletron_setup import SetupError

__all__ = ["FieldRecord", "PromptFrame", "WizardState"]

SCHEMA_VERSION = 1
WIZARD_VERSION = 2


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


@dataclass
class FieldRecord:
    """One authored value with provenance and dependency metadata."""

    value: Any
    source: str = "user"
    dependencies: tuple[str, ...] = ()
    stale: bool = False
    requested: Any = None
    effective: Any = None
    error: str | None = None

    def __post_init__(self) -> None:
        self.dependencies = tuple(str(item) for item in self.dependencies)
        if self.effective is None:
            self.effective = self.value

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FieldRecord:
        """Restore a field record from serialized state."""
        return cls(
            value=payload.get("value"),
            source=str(payload.get("source", "user")),
            dependencies=tuple(payload.get("dependencies") or ()),
            stale=bool(payload.get("stale", False)),
            requested=payload.get("requested"),
            effective=payload.get("effective"),
            error=payload.get("error"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the field record to plain Python values."""
        return _plain(asdict(self))


@dataclass(frozen=True)
class PromptFrame:
    """One resumable prompt location."""

    section: str
    prompt_id: str
    collection: str | None = None
    item_id: str | None = None
    cursor: int | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PromptFrame:
        """Restore a prompt frame from serialized state."""
        return cls(
            section=str(payload["section"]),
            prompt_id=str(payload["prompt_id"]),
            collection=payload.get("collection"),
            item_id=payload.get("item_id"),
            cursor=payload.get("cursor"),
        )


@dataclass
class WizardState:
    """Mutable v2 setup state whose mutations are persisted atomically."""

    path: Path
    payload: dict[str, Any]
    _fields: dict[str, FieldRecord] = field(default_factory=dict)

    @classmethod
    def start(
        cls,
        campaign_dir: Path,
        *,
        defaults_path: Path | None,
        setup_mode: str = "full",
        preset: str | None = None,
    ) -> WizardState:
        """Create and persist a new setup-v2 campaign state."""
        campaign_dir = Path(campaign_dir).expanduser().resolve()
        if campaign_dir.exists() and any(campaign_dir.iterdir()):
            raise SetupError(
                f"Campaign directory is not empty: {campaign_dir}. "
                "Choose a new directory or use --resume."
            )
        campaign_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "wizard_version": WIZARD_VERSION,
            "defaults_path": (
                str(Path(defaults_path).expanduser().resolve())
                if defaults_path is not None
                else None
            ),
            "setup": {
                "mode": str(setup_mode),
                "preset": str(preset) if preset is not None else None,
            },
            "fields": {},
            "navigation": {"frames": [], "cursor": None},
            "collections": {},
            "model": {},
            "inventory": {},
            "updated_at": _timestamp(),
        }
        state = cls(campaign_dir / "answers_v2.yaml", payload)
        state.save()
        return state

    @classmethod
    def resume(cls, path: Path) -> WizardState:
        """Load a compatible setup-v2 campaign state."""
        candidate = Path(path).expanduser().resolve()
        state_path = candidate / "answers_v2.yaml" if candidate.is_dir() else candidate
        if not state_path.is_file():
            raise SetupError(f"V2 setup state does not exist: {state_path}")
        try:
            payload = yaml.safe_load(state_path.read_text()) or {}
        except (OSError, yaml.YAMLError) as error:
            raise SetupError(f"Cannot read setup state {state_path}: {error}") from error
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise SetupError(
                f"Unsupported v2 setup schema {payload.get('schema_version')!r}; "
                f"expected {SCHEMA_VERSION}."
            )
        if payload.get("wizard_version") != WIZARD_VERSION:
            raise SetupError(
                f"Unsupported wizard version {payload.get('wizard_version')!r}; "
                f"expected {WIZARD_VERSION}."
            )
        fields = {
            str(name): FieldRecord.from_dict(record)
            for name, record in dict(payload.get("fields") or {}).items()
        }
        return cls(state_path, dict(payload), fields)

    @property
    def campaign_dir(self) -> Path:
        """Return the campaign directory containing this state."""
        return self.path.parent

    @property
    def defaults_path(self) -> Path | None:
        """Return the persisted defaults-file path, if configured."""
        value = self.payload.get("defaults_path")
        return Path(str(value)) if value else None

    @property
    def setup_mode(self) -> str:
        """Return the persisted guided or full interaction mode."""
        setup = self.payload.get("setup")
        if not isinstance(setup, Mapping):
            return "full"
        return str(setup.get("mode", "full"))

    @property
    def preset(self) -> str | None:
        """Return the persisted guided preset name, if any."""
        setup = self.payload.get("setup")
        if not isinstance(setup, Mapping):
            return None
        value = setup.get("preset")
        return str(value) if value else None

    def set_setup_mode(self, mode: str) -> None:
        """Persist an explicit guided or full mode transition."""
        self.payload.setdefault("setup", {})["mode"] = str(mode)
        self.save()

    def set_preset(self, preset: str) -> None:
        """Persist a replacement guided profile selection."""
        self.payload.setdefault("setup", {})["preset"] = str(preset)
        self.save()

    def set_defaults_path(self, path: Path) -> None:
        """Persist an explicitly accepted replacement defaults file."""
        self.payload["defaults_path"] = str(Path(path).expanduser().resolve())
        self.save()

    def field(self, path: str) -> FieldRecord:
        """Return one required authored field record."""
        try:
            return self._fields[path]
        except KeyError as error:
            raise KeyError(f"Unknown setup field: {path}") from error

    def get_field(self, path: str, default: Any = None) -> Any:
        """Return one effective field value or a fallback."""
        record = self._fields.get(path)
        return default if record is None else record.effective

    def records(self) -> Mapping[str, FieldRecord]:
        """Return a copy of all authored field records."""
        return dict(self._fields)

    def set_field(
        self,
        path: str,
        value: Any,
        *,
        source: str = "user",
        dependencies: Sequence[str] = (),
        requested: Any = None,
        effective: Any = None,
    ) -> FieldRecord:
        """Persist a field value and invalidate downstream dependents."""
        previous = self._fields.get(path)
        resolved_effective = value if effective is None else effective
        changed = previous is None or previous.effective != resolved_effective
        record = FieldRecord(
            value=value,
            source=source,
            dependencies=tuple(dependencies),
            stale=False,
            requested=requested,
            effective=resolved_effective,
            error=None,
        )
        self._fields[path] = record
        if changed:
            self.mark_dependents_stale(path, save=False)
        self.save()
        return record

    def mark_dependents_stale(self, changed_path: str, *, save: bool = True) -> tuple[str, ...]:
        """Mark transitive dependents of a changed field as stale."""
        reverse: dict[str, set[str]] = {}
        for field_path, record in self._fields.items():
            for dependency in record.dependencies:
                reverse.setdefault(dependency, set()).add(field_path)
        marked: list[str] = []
        queue = deque([changed_path])
        visited = {changed_path}
        while queue:
            dependency = queue.popleft()
            for field_path in sorted(reverse.get(dependency, ())):
                if field_path in visited:
                    continue
                visited.add(field_path)
                record = self._fields[field_path]
                record.stale = True
                record.error = "Requires revalidation after an upstream change."
                marked.append(field_path)
                queue.append(field_path)
        if save:
            self.save()
        return tuple(marked)

    def revalidate(
        self,
        validators: Mapping[str, Callable[[Any, WizardState], str | None]],
    ) -> Mapping[str, str]:
        """Revalidate stale fields and return unresolved issues."""
        issues: dict[str, str] = {}
        for path, record in self._fields.items():
            if not record.stale:
                continue
            validator = validators.get(path)
            error = (
                validator(record.effective, self)
                if validator is not None
                else "No validator is registered for this stale field."
            )
            record.error = error
            record.stale = error is not None
            if error is not None:
                issues[path] = error
        self.save()
        return issues

    def collection(self, path: str) -> Any:
        """Return a named collection from persisted state."""
        return self.payload.setdefault("collections", {}).get(path)

    def set_collection(self, path: str, value: Any) -> None:
        """Persist a named collection."""
        self.payload.setdefault("collections", {})[path] = _plain(value)
        self.save()

    @property
    def frames(self) -> tuple[PromptFrame, ...]:
        """Return the persisted prompt-navigation stack."""
        return tuple(
            PromptFrame.from_dict(item)
            for item in self.payload.setdefault("navigation", {}).get("frames", ())
        )

    @property
    def answered_frames(self) -> tuple[tuple[PromptFrame, Any, int], ...]:
        """Return persisted prompt answers with their navigation-stack positions."""
        return tuple(
            (PromptFrame.from_dict(item), item["answer"], index)
            for index, item in enumerate(
                self.payload.setdefault("navigation", {}).get("frames", ())
            )
            if "answer" in item
        )

    def push_frame(self, frame: PromptFrame) -> None:
        """Push a prompt frame onto the navigation stack."""
        frames = self.payload.setdefault("navigation", {}).setdefault("frames", [])
        if not frames or frames[-1] != asdict(frame):
            frames.append(_plain(asdict(frame)))
        self.payload["navigation"]["cursor"] = frame.prompt_id
        self.save()

    def answer_frame(self, frame: PromptFrame, value: Any) -> None:
        """Persist the accepted answer for the current prompt frame."""
        frames = self.payload.setdefault("navigation", {}).setdefault("frames", [])
        if not frames or PromptFrame.from_dict(frames[-1]) != frame:
            raise RuntimeError(f"Cannot answer inactive prompt frame {frame.prompt_id!r}.")
        frames[-1]["answer"] = _plain(value)
        self.save()

    def pop_frame(self) -> PromptFrame | None:
        """Pop the active frame and return the new active frame."""
        navigation = self.payload.setdefault("navigation", {})
        frames = navigation.setdefault("frames", [])
        if frames:
            frames.pop()
        navigation["cursor"] = frames[-1]["prompt_id"] if frames else None
        self.save()
        return PromptFrame.from_dict(frames[-1]) if frames else None

    def truncate_frames(self, count: int) -> None:
        """Discard navigation frames at and after ``count`` without rewriting answers."""
        navigation = self.payload.setdefault("navigation", {})
        frames = navigation.setdefault("frames", [])
        del frames[max(0, int(count)) :]
        navigation["cursor"] = frames[-1]["prompt_id"] if frames else None
        self.save()

    def replace_frames(self, frames: Sequence[PromptFrame]) -> None:
        """Replace the complete prompt-navigation stack."""
        rendered = [_plain(asdict(frame)) for frame in frames]
        self.payload["navigation"] = {
            "frames": rendered,
            "cursor": rendered[-1]["prompt_id"] if rendered else None,
        }
        self.save()

    def set_model(self, model: Mapping[str, Any], inventory: Mapping[str, Any]) -> None:
        """Persist inspected model metadata and inventory."""
        self.payload["model"] = _plain(model)
        self.payload["inventory"] = _plain(inventory)
        self.save()

    def save(self) -> None:
        """Atomically write the current state to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.payload["fields"] = {
            path: record.to_dict() for path, record in sorted(self._fields.items())
        }
        self.payload["updated_at"] = _timestamp()
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        try:
            with temporary.open("w") as stream:
                yaml.safe_dump(_plain(self.payload), stream, sort_keys=False, width=100)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except OSError as error:
            raise SetupError(f"Cannot save v2 setup state {self.path}: {error}") from error
