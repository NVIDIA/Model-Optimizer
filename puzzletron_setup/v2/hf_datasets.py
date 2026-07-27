# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamic Hugging Face dataset configuration metadata for setup v2."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, get_token

from puzzletron_setup import SetupError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

__all__ = [
    "HfSubsetCatalog",
    "HfSubsetInfo",
    "discover_hf_subset_catalog",
    "format_subset_choice",
    "proportional_subset_weights",
]

_DATASET_VIEWER_SIZE_URL = "https://datasets-server.huggingface.co/size"


@dataclass(frozen=True)
class HfSubsetInfo:
    """One dynamically discovered Hugging Face dataset configuration."""

    name: str
    num_rows: int | None
    num_bytes_original_files: int | None
    selectable: bool = True
    disabled_reason: str | None = None
    num_media_shards: int | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("Hugging Face subset name cannot be empty")
        object.__setattr__(self, "name", name)
        for field_name in ("num_rows", "num_bytes_original_files"):
            value = getattr(self, field_name)
            if value is not None:
                object.__setattr__(self, field_name, int(value))
        if self.num_media_shards is not None:
            num_media_shards = int(self.num_media_shards)
            if num_media_shards < 0:
                raise ValueError("Hugging Face media-shard count cannot be negative")
            object.__setattr__(self, "num_media_shards", num_media_shards)
        if self.selectable and self.disabled_reason:
            raise ValueError("a selectable subset cannot have a disabled reason")
        if not self.selectable and not self.disabled_reason:
            raise ValueError("a disabled subset requires a reason")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HfSubsetInfo:
        """Restore one YAML-safe subset record."""
        return cls(
            name=str(payload["name"]),
            num_rows=payload.get("num_rows"),
            num_bytes_original_files=payload.get("num_bytes_original_files"),
            num_media_shards=payload.get("num_media_shards"),
            selectable=bool(payload.get("selectable", True)),
            disabled_reason=payload.get("disabled_reason"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe subset record."""
        return asdict(self)


@dataclass(frozen=True)
class HfSubsetCatalog:
    """Revision-locked Hugging Face subset metadata."""

    source: str
    revision: str
    default_subset: str | None
    subsets: tuple[HfSubsetInfo, ...]

    def __post_init__(self) -> None:
        source = str(self.source).strip()
        revision = str(self.revision).strip()
        if not source or not revision:
            raise ValueError("Hugging Face catalog requires source and immutable revision")
        if not self.subsets:
            raise ValueError("Hugging Face catalog must contain at least one subset")
        names = [item.name for item in self.subsets]
        if len(names) != len(set(names)):
            raise ValueError("Hugging Face catalog subset names must be unique")
        if self.default_subset is not None and self.default_subset not in set(names):
            raise ValueError(
                f"default subset {self.default_subset!r} is absent from the catalog"
            )
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "revision", revision)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> HfSubsetCatalog:
        """Restore a cached catalog without contacting Hugging Face."""
        return cls(
            source=str(payload["source"]),
            revision=str(payload["revision"]),
            default_subset=payload.get("default_subset"),
            subsets=tuple(
                HfSubsetInfo.from_dict(item) for item in payload.get("subsets") or ()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-safe cache payload."""
        return {
            "source": self.source,
            "revision": self.revision,
            "default_subset": self.default_subset,
            "subsets": [item.to_dict() for item in self.subsets],
        }


def _get_dataset_config_names(source: str, *, revision: str) -> list[str]:
    try:
        from datasets import get_dataset_config_names
    except ImportError as error:
        raise SetupError(
            "Hugging Face subset discovery requires the `datasets` package. "
            "Install examples/puzzletron/requirements-setup.txt."
        ) from error
    return list(get_dataset_config_names(source, revision=revision))


def _load_size_payload(source: str) -> Mapping[str, Any]:
    headers = {"Accept": "application/json"}
    token = get_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(
        f"{_DATASET_VIEWER_SIZE_URL}?{urlencode({'dataset': source})}",
        headers=headers,
    )
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
    except (OSError, ValueError) as error:
        raise SetupError(
            f"Cannot fetch Hugging Face size metadata for {source}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise SetupError(f"Hugging Face returned malformed size metadata for {source}")
    return payload


def _default_subset(card_data: Any, config_names: Sequence[str]) -> str | None:
    if isinstance(card_data, dict) or hasattr(card_data, "get"):
        configs = card_data.get("configs") or ()
    else:
        configs = ()
    for item in configs:
        if isinstance(item, dict) and bool(item.get("default", False)):
            name = item.get("config_name", item.get("name"))
            if name in config_names:
                return str(name)
    return config_names[0] if len(config_names) == 1 else None


def _size_records(payload: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    size = payload.get("size")
    configs = size.get("configs") if isinstance(size, dict) else None
    records = {}
    for item in configs or ():
        if not isinstance(item, dict) or "config" not in item or "split" in item:
            continue
        records[str(item["config"])] = item
    return records


def _sibling_paths(info: Any) -> tuple[str, ...]:
    return tuple(
        str(getattr(item, "rfilename", getattr(item, "path", "")))
        for item in (getattr(info, "siblings", None) or ())
    )


def _repository_subset_bytes(info: Any, subset: str) -> int | None:
    sizes = []
    prefix = f"{subset}/"
    for item in getattr(info, "siblings", None) or ():
        path = str(getattr(item, "rfilename", getattr(item, "path", "")))
        size = getattr(item, "size", None)
        if (
            path.startswith(prefix)
            and isinstance(size, int)
            and not isinstance(size, bool)
            and size >= 0
        ):
            sizes.append(size)
    return sum(sizes) if sizes else None


def discover_hf_subset_catalog(
    source: str,
    revision: str | None = None,
    *,
    require_hosted_media: bool = False,
    api: HfApi | None = None,
    config_names_loader: Callable[..., list[str]] | None = None,
    size_payload_loader: Callable[[str], Mapping[str, Any]] | None = None,
) -> HfSubsetCatalog:
    """Discover ordered subset names and sizes without downloading dataset rows."""
    source = str(source).strip()
    if not source:
        raise SetupError("Hugging Face dataset source cannot be empty.")
    api = api or HfApi()
    try:
        info = api.dataset_info(source, revision=revision, files_metadata=True)
    except Exception as error:
        raise SetupError(f"Cannot inspect Hugging Face dataset {source}: {error}") from error
    resolved_revision = str(getattr(info, "sha", "") or "").strip()
    if not resolved_revision:
        raise SetupError(
            f"Hugging Face did not return an immutable revision for {source}"
        )
    config_names_loader = config_names_loader or _get_dataset_config_names
    try:
        names = [
            str(name).strip()
            for name in config_names_loader(source, revision=resolved_revision)
            if str(name).strip()
        ]
    except Exception as error:
        raise SetupError(
            f"Cannot enumerate Hugging Face subsets for {source}: {error}"
        ) from error
    if not names:
        raise SetupError(f"Hugging Face dataset {source} exposes no subsets")
    if len(names) != len(set(names)):
        raise SetupError(f"Hugging Face dataset {source} returned duplicate subsets")

    size_payload_loader = size_payload_loader or _load_size_payload
    try:
        records = _size_records(size_payload_loader(source))
    except SetupError:
        raise
    except Exception as error:
        raise SetupError(
            f"Cannot fetch Hugging Face size metadata for {source}: {error}"
        ) from error

    sibling_paths = _sibling_paths(info)
    subsets = []
    for name in names:
        record = records.get(name)
        num_media_shards = sum(
            1
            for path in sibling_paths
            if path.startswith(f"{name}/media/") and path.lower().endswith(".tar")
        )
        num_rows = record.get("num_rows") if record is not None else None
        num_bytes = (
            record.get("num_bytes_original_files") if record is not None else None
        )
        if num_bytes is None:
            num_bytes = _repository_subset_bytes(info, name)
        if num_bytes is None and record is not None:
            num_bytes = record.get("num_bytes_parquet_files")
        disabled_reason = None
        if num_rows is None:
            disabled_reason = "row count unavailable"
        elif int(num_rows) <= 0:
            disabled_reason = "subset has no rows"
        elif num_bytes is None:
            disabled_reason = "size unavailable"
        elif require_hosted_media and num_media_shards == 0:
            disabled_reason = "external media required"
        subsets.append(
            HfSubsetInfo(
                name=name,
                num_rows=num_rows,
                num_bytes_original_files=num_bytes,
                num_media_shards=num_media_shards,
                selectable=disabled_reason is None,
                disabled_reason=disabled_reason,
            )
        )

    return HfSubsetCatalog(
        source=source,
        revision=resolved_revision,
        default_subset=_default_subset(getattr(info, "card_data", None), names),
        subsets=tuple(subsets),
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown size"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:,.2f} {unit}"
        amount /= 1024.0
    raise AssertionError("unreachable")


def format_subset_choice(item: HfSubsetInfo) -> str:
    """Render one compact checkbox label."""
    rows = "unknown rows" if item.num_rows is None else f"{item.num_rows:,} rows"
    label = f"{item.name} — {rows} — {_format_bytes(item.num_bytes_original_files)}"
    return f"{label} — {item.disabled_reason}" if item.disabled_reason else label


def proportional_subset_weights(
    catalog: HfSubsetCatalog,
    selected_names: Sequence[str],
) -> dict[str, float]:
    """Return row-proportional weights for an ordered checkbox selection."""
    selected = [str(name) for name in selected_names]
    if not selected:
        raise SetupError("Choose at least one dataset subset.")
    if len(selected) != len(set(selected)):
        raise SetupError("Selected dataset subsets must be unique.")
    by_name = {item.name: item for item in catalog.subsets}
    unknown = [name for name in selected if name not in by_name]
    if unknown:
        raise SetupError(f"Selected dataset subset is unknown: {unknown[0]}")
    unavailable = [by_name[name] for name in selected if not by_name[name].selectable]
    if unavailable:
        item = unavailable[0]
        raise SetupError(
            f"Selected dataset subset {item.name!r} is unavailable: "
            f"{item.disabled_reason}."
        )
    row_counts = [by_name[name].num_rows for name in selected]
    if any(value is None or value <= 0 for value in row_counts):
        raise SetupError("Selected dataset subsets require positive row counts.")
    total = math.fsum(int(value) for value in row_counts if value is not None)
    weights = {}
    for index, (name, rows) in enumerate(zip(selected, row_counts)):
        if index == len(selected) - 1:
            weight = 1.0 - math.fsum(weights.values())
        else:
            weight = int(rows) / total
        weights[name] = weight
    return weights
