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

"""Atomic bounded exports and validated retry for the automatic HF example."""

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from pathlib import Path

from modelopt.torch.export import export_hf_checkpoint

__all__ = []
_MANIFEST = ".autoquant-manifest.json"


@contextmanager
def export_lease(args, workspace):
    destination = Path(args.export_path).resolve()
    identity = hashlib.sha256(str(destination).encode()).hexdigest()
    stage = destination.with_name(f".{destination.name}-autoquant-{identity[:16]}")
    reservation = workspace.root / f"export-{identity}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with workspace._lock(destination.with_name(f".{destination.name}.autoquant.lock")):
        if not reservation.exists() and (stage.exists() or destination.exists()):
            raise ValueError("Refusing an existing export without this workspace's reservation")
        value = {
            "destination": str(destination),
            "stage": str(stage),
            "bytes": args.auto_quantize_resources["export_bytes"],
        }
        workspace.publish(
            reservation,
            json.dumps(value, sort_keys=True),
            reserve_bytes=value["bytes"],
            available_bytes=args.auto_quantize_export_available_bytes,
        )
        args.auto_quantize_export_stage = str(stage)
        yield


def _inventory(directory, limit, *, hash_files=True):
    files = {}
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink() or not (path.is_dir() or path.is_file()):
            raise ValueError("Export contains an unsupported filesystem entry")
        if path.is_file() and path.relative_to(directory).as_posix() != _MANIFEST:
            total += path.stat().st_size
            if total > limit:
                raise ValueError("Export exceeds its storage byte limit")
            digest = None
            if hash_files:
                with path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[str(path.relative_to(directory))] = {
                "bytes": path.stat().st_size,
                "sha256": digest,
            }
    return files


def export_pipeline(args, model):
    """Publish once, or verify an already published export against the current assignment."""
    destination, stage = Path(args.export_path), Path(args.auto_quantize_export_stage)
    resources = args.auto_quantize_resources
    assignment = hashlib.sha256(Path(args.auto_quantize_assignment).read_bytes()).hexdigest()
    if destination.is_symlink() or stage.is_symlink():
        raise ValueError("Automatic export directories cannot be symlinks")
    if destination.exists():
        receipt = destination / _MANIFEST
        if receipt.is_symlink() or receipt.stat().st_size > resources["metadata_bytes"]:
            raise ValueError("Export provenance exceeds its byte limit")
        manifest = json.loads(receipt.read_text())
        if manifest != {
            "assignment": assignment,
            "files": _inventory(destination, resources["export_bytes"] - receipt.stat().st_size),
        }:
            raise ValueError("Published export differs from the current assignment or payloads")
        args.checkpoint_exported = True
        return
    if stage.exists():
        _inventory(stage, resources["export_bytes"], hash_files=False)
        shutil.rmtree(stage)
    stage.mkdir()
    try:
        export_hf_checkpoint(
            model,
            export_dir=stage,
            max_export_bytes=resources["export_bytes"] - resources["metadata_bytes"],
            max_shard_size=min(1024**3, resources["host_bytes"] // 8),
        )
        if (stage / _MANIFEST).exists():
            raise ValueError("Source checkpoint uses a reserved export manifest filename")
        content = json.dumps(
            {
                "assignment": assignment,
                "files": _inventory(stage, resources["export_bytes"] - resources["metadata_bytes"]),
            },
            sort_keys=True,
        ).encode()
        if len(content) > resources["metadata_bytes"]:
            raise ValueError("Export provenance exceeds its byte limit")
        (stage / _MANIFEST).write_bytes(content)
        if destination.exists():
            raise ValueError("Export destination appeared before publication")
        os.rename(stage, destination)
        args.checkpoint_exported = True
    finally:
        if stage.exists():
            shutil.rmtree(stage)
