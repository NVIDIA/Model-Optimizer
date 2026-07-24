# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration-only model and dataset inspection for Puzzletron setup."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import yaml
from huggingface_hub import HfApi
from transformers import AutoConfig, PretrainedConfig

from . import SetupError
from .profiles import ModelInventory, resolve_profile

__all__ = [
    "InspectedModel",
    "ModalityFinding",
    "infer_dataset_modality",
    "inspect_model",
    "normalize_dataset_source",
    "normalize_model_source",
]

_MEDIA_KEYS = {"audio", "image", "images", "pixel_values", "video", "videos", "vision"}
_HUGGING_FACE_HOSTS = {
    "huggingface.co",
    "www.huggingface.co",
    "huggingface.com",
    "www.huggingface.com",
}


def _normalize_source(source: str, *, kind: str, url_prefix: str | None = None) -> str:
    """Normalize an existing local path or Hugging Face web URL."""

    source = source.strip()
    if not source:
        raise SetupError(f"Enter a {kind} path or Hugging Face URL.")

    expanded = Path(source).expanduser()
    if expanded.exists():
        return str(expanded.resolve())

    url_source = source
    first_component = source.split("/", 1)[0].lower()
    if first_component in _HUGGING_FACE_HOSTS:
        url_source = f"https://{source}"
    parsed = urlparse(url_source)
    if parsed.scheme not in {"http", "https"}:
        if parsed.scheme or parsed.netloc:
            raise SetupError(
                f"Unsupported {kind} source {source!r}; use an existing local path "
                "or a Hugging Face HTTP(S) URL."
            )
        repository_parts = [part for part in source.split("/") if part]
        if not source.startswith((".", "/")) and len(repository_parts) == 2:
            return "/".join(repository_parts)
        raise SetupError(f"Local {kind} path does not exist: {expanded}")
    if parsed.netloc.lower() not in _HUGGING_FACE_HOSTS:
        raise SetupError(
            f"{kind.capitalize()} URLs must point to huggingface.co or huggingface.com; "
            "otherwise enter an existing local path."
        )
    parts = [unquote(part) for part in parsed.path.split("/") if part]
    if url_prefix is not None and parts[:1] == [url_prefix]:
        parts = parts[1:]
    if len(parts) != 2:
        raise SetupError(
            f"Enter a Hugging Face {kind} URL containing an owner and repository name."
        )
    return "/".join(parts)


def normalize_model_source(source: str) -> str:
    """Normalize an existing model path or Hugging Face model web URL."""

    return _normalize_source(source, kind="model")


def normalize_dataset_source(source: str) -> str:
    """Normalize an existing dataset path or Hugging Face dataset web URL."""

    return _normalize_source(source, kind="dataset", url_prefix="datasets")


@dataclass(frozen=True)
class InspectedModel:
    """A resolved source config and its normalized Puzzletron inventory."""

    source: str
    requested_revision: str | None
    resolved_revision: str | None
    is_local: bool
    config: Mapping[str, Any]
    inventory: ModelInventory

    def to_dict(self) -> dict[str, Any]:
        """Convert the inspection result to YAML-safe built-in values."""
        return asdict(self)


@dataclass(frozen=True)
class ModalityFinding:
    """Best-effort dataset modality and the evidence used to infer it."""

    modality: str
    evidence: str


def _load_config_dict(source: str, *, revision: str | None, local: bool) -> dict[str, Any]:
    path = Path(source).expanduser()
    if local and path.is_file():
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise SetupError(f"Cannot read model config file {path}: {error}") from error
        if not isinstance(payload, dict):
            raise SetupError(f"Model config file must contain a JSON object: {path}")
        return payload

    kwargs = {
        "revision": revision,
        "local_files_only": local,
        "trust_remote_code": False,
    }
    try:
        return dict(AutoConfig.from_pretrained(source, **kwargs).to_dict())
    except (OSError, ValueError, KeyError):
        try:
            config, _unused = PretrainedConfig.get_config_dict(
                source,
                revision=revision,
                local_files_only=local,
            )
        except Exception as error:
            raise SetupError(f"Cannot load Hugging Face config for {source!r}: {error}") from error
        return dict(config)


def inspect_model(source: str, revision: str | None = None) -> InspectedModel:
    """Inspect a local path or Hugging Face URL without loading model weights."""

    source = normalize_model_source(source)
    expanded = Path(source).expanduser()
    is_local = expanded.exists()
    resolved_revision = None
    config_source = str(expanded.resolve()) if is_local else source
    effective_source = (
        str(expanded.parent.resolve())
        if is_local and expanded.is_file() and expanded.name == "config.json"
        else config_source
    )
    if not is_local:
        try:
            resolved_revision = HfApi().model_info(config_source, revision=revision).sha
        except Exception as error:
            raise SetupError(f"Cannot resolve Hugging Face model {source!r}: {error}") from error
    config = _load_config_dict(
        config_source,
        revision=resolved_revision or revision,
        local=is_local,
    )
    profile = resolve_profile(config)
    return InspectedModel(
        source=effective_source,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        is_local=is_local,
        config=config,
        inventory=profile.inventory(config),
    )


def _keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        result = {str(key).lower() for key in value}
        for child in value.values():
            result.update(_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value[:4]:
            result.update(_keys(child))
        return result
    return set()


def _local_dataset_metadata(path: Path) -> Any:
    candidate = path
    if path.is_dir():
        files = [
            item
            for pattern in ("*.json", "*.jsonl", "*.yaml", "*.yml")
            for item in sorted(path.glob(pattern))
        ]
        if not files:
            return None
        candidate = files[0]
    if candidate.suffix == ".jsonl":
        with candidate.open() as stream:
            line = stream.readline()
        return json.loads(line) if line else None
    if candidate.suffix == ".json":
        return json.loads(candidate.read_text())
    if candidate.suffix in {".yaml", ".yml"}:
        return yaml.safe_load(candidate.read_text())
    return None


def infer_dataset_modality(source: str) -> ModalityFinding:
    """Infer text versus multimodal data and return explicit evidence."""

    source = normalize_dataset_source(source)
    path = Path(source).expanduser()
    if path.exists():
        try:
            keys = _keys(_local_dataset_metadata(path))
        except (OSError, ValueError, json.JSONDecodeError, yaml.YAMLError):
            keys = set()
        media = sorted(keys & _MEDIA_KEYS)
        if media:
            return ModalityFinding("multimodal", f"local metadata contains {media}")
        name = path.name.lower()
        if any(token in name for token in ("image", "video", "vision", "audio", "vl")):
            return ModalityFinding("multimodal", f"local path name contains a media hint: {name}")
        if keys:
            return ModalityFinding("text", "local metadata contains no media fields")
        return ModalityFinding("unknown", "local dataset metadata could not be inspected")

    try:
        info = HfApi().dataset_info(source)
        evidence = " ".join(
            str(value)
            for value in (getattr(info, "tags", None), getattr(info, "card_data", None))
            if value
        ).lower()
    except Exception:
        evidence = ""
    if any(token in evidence for token in ("image", "video", "vision", "audio")):
        return ModalityFinding("multimodal", "Hugging Face dataset metadata contains media tags")
    if evidence:
        return ModalityFinding("text", "Hugging Face dataset metadata contains no media tags")
    return ModalityFinding("unknown", "dataset metadata was unavailable")
