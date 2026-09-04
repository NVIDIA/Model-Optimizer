# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dependency-light validation for materialized Puzzletron dataset payloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["vlm_materialization_is_complete"]

_VALIDATED_PAYLOADS: dict[Path, tuple[Any, ...]] = {}


def _payload_signature(
    root: Path,
    manifest: Mapping[str, Any],
) -> tuple[Any, ...] | None:
    try:
        manifest_digest = hashlib.sha256(
            json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        paths = [root / "samples.json"]
        for image in manifest.get("images", []):
            if not isinstance(image, Mapping) or not isinstance(image.get("path"), str):
                return None
            relative = Path(image["path"])
            if relative.is_absolute() or ".." in relative.parts:
                return None
            paths.append(root / relative)
        stats = []
        for path in paths:
            stat = path.stat()
            stats.append(
                (
                    str(path),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
    except (OSError, TypeError, ValueError):
        return None
    return (manifest_digest, *stats)


def vlm_materialization_is_complete(
    output_dir: str | Path,
    manifest: Mapping[str, Any],
    *,
    cache_success: bool = False,
) -> bool:
    """Return whether a VLM materialization matches its recorded payload manifest."""

    root = Path(output_dir).expanduser().absolute()
    signature = _payload_signature(root, manifest)
    if signature is None:
        return False
    if cache_success and _VALIDATED_PAYLOADS.get(root) == signature:
        return True
    sample_count = manifest.get("sample_count")
    samples_sha256 = manifest.get("samples_sha256")
    image_count = manifest.get("image_count")
    images = manifest.get("images")
    acquisition = manifest.get("acquisition")
    requested_samples = acquisition.get("num_samples") if isinstance(acquisition, Mapping) else None
    if (
        isinstance(sample_count, bool)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or not isinstance(samples_sha256, str)
        or isinstance(image_count, bool)
        or not isinstance(image_count, int)
        or image_count != sample_count
        or not isinstance(acquisition, Mapping)
        or acquisition.get("adapter") != "nemotron_vlm_v2"
        or isinstance(requested_samples, bool)
        or not isinstance(requested_samples, int)
        or requested_samples != sample_count
        or not isinstance(images, list)
        or len(images) != image_count
    ):
        return False

    try:
        samples_payload = (root / "samples.json").read_bytes()
        samples = json.loads(samples_payload)
    except (OSError, json.JSONDecodeError):
        return False
    if (
        hashlib.sha256(samples_payload).hexdigest() != samples_sha256
        or not isinstance(samples, list)
        or len(samples) != sample_count
    ):
        return False

    referenced_images: list[str] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            return False
        conversation = sample.get("conversation")
        if not isinstance(conversation, list):
            return False
        for message in conversation:
            if not isinstance(message, Mapping) or not isinstance(message.get("content"), list):
                return False
            for item in message["content"]:
                if not isinstance(item, Mapping):
                    return False
                if item.get("type") == "image":
                    path = item.get("image")
                    if not isinstance(path, str):
                        return False
                    referenced_images.append(path)

    recorded_images: list[str] = []
    for image in images:
        if not isinstance(image, Mapping):
            return False
        relative_value = image.get("path")
        digest = image.get("sha256")
        if not isinstance(relative_value, str) or not isinstance(digest, str):
            return False
        relative = Path(relative_value)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = root / relative
        try:
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                return False
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                return False
        except OSError:
            return False
        recorded_images.append(relative.as_posix())

    complete = sorted(recorded_images) == sorted(referenced_images)
    if complete and cache_success:
        _VALIDATED_PAYLOADS[root] = signature
    return complete
