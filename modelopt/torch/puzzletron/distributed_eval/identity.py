"""Canonical, content-addressed identities for distributed evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from datetime import date, datetime, time, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def canonicalize(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation of ``value``."""
    if isinstance(value, BaseModel):
        return canonicalize(value.model_dump(mode="python", exclude_none=False))
    if dataclasses.is_dataclass(value):
        return canonicalize(dataclasses.asdict(value))
    if isinstance(value, Enum):
        return canonicalize(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, (date, time)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"__float__": repr(value)}
        return value
    if isinstance(value, dict):
        return {
            str(key): canonicalize(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonicalize(item) for item in value]
    if isinstance(value, set):
        normalized = [canonicalize(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return canonicalize(value.to_dict())
    raise TypeError(f"Cannot canonicalize {type(value).__module__}.{type(value).__qualname__}")


def canonical_json(value: Any) -> str:
    """Serialize ``value`` with stable ordering and no insignificant whitespace."""
    return json.dumps(
        canonicalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def content_id(kind: str, value: Any) -> str:
    """Return a full SHA-256 content identity prefixed by ``kind``."""
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return f"{kind}_{digest}"


def prefix_cache_id(
    *,
    campaign_id: str,
    model: Any,
    data: Any,
    prefix_signature: Any,
    prefix_length: int,
    parallelism: Any,
) -> str:
    """Stable identity reserved for a future topology-specific prefix cache."""
    return content_id(
        "prefix",
        {
            "campaign_id": campaign_id,
            "model": model,
            "data": data,
            "prefix_signature": prefix_signature,
            "prefix_length": prefix_length,
            "parallelism": parallelism,
        },
    )
