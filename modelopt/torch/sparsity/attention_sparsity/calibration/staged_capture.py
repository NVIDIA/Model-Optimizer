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

"""Opt-in host-stage receipts for the whole-prefill capture example."""

import os
import tempfile
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path

from ..plugins.mask_reuse_capture import CaptureContractError
from .checkpoint_manifest import read_stable_file_snapshot, stable_file_sha256

__all__ = ["CaptureStageFiles", "load_stage_receipt", "verify_controller_bridge"]

_HOST_BRIDGE_PATHS = frozenset(
    {
        "examples/vllm_serve/collect_mask_reuse.py",
        "modelopt/torch/sparsity/attention_sparsity/calibration/staged_capture.py",
    }
)


def load_stage_receipt(path, expected_sha256, parse):
    """Read a separately trusted producer receipt, not an arbitrary resume certificate."""
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise CaptureContractError("stage receipt requires an explicit trusted SHA256")
    snapshot = read_stable_file_snapshot(path, label="capture stage receipt")
    if snapshot.sha256 != expected_sha256:
        raise CaptureContractError("capture stage receipt SHA mismatch")
    return parse(snapshot.payload)


def verify_controller_bridge(old, new, path, expected_sha256, *, parse, serialize):
    """Permit only the exactly pinned, externally reviewed controller delta."""
    if old == new:
        if path is not None or expected_sha256 is not None:
            raise CaptureContractError("same-source resume must not supply a source bridge")
        return None
    if path is None or expected_sha256 is None:
        raise CaptureContractError("changed controller requires a pinned source bridge")
    bridge = load_stage_receipt(path, expected_sha256, parse)
    if {k: v for k, v in old.items() if k != "controller"} != {
        k: v for k, v in new.items() if k != "controller"
    }:
        raise CaptureContractError("source bridge cannot change capture semantics")
    before, after = old["controller"], new["controller"]
    if before["versions"] != after["versions"]:
        raise CaptureContractError("source bridge cannot change runtime versions")
    changes = [
        {"path": name, "before": before["files"].get(name), "after": after["files"].get(name)}
        for name in sorted(before["files"].keys() | after["files"].keys())
        if before["files"].get(name) != after["files"].get(name)
    ]
    if any(change["path"] not in _HOST_BRIDGE_PATHS for change in changes):
        raise CaptureContractError("source bridge contains non-host controller changes")
    expected = {
        "schema_version": 1,
        "old_request_sha256": sha256(serialize(old)).hexdigest(),
        "new_request_sha256": sha256(serialize(new)).hexdigest(),
        "controller_file_changes": changes,
    }
    if bridge != expected or not changes:
        raise CaptureContractError("source bridge does not bind the exact controller delta")
    return {**expected, "bridge_receipt_sha256": expected_sha256}


class CaptureStageFiles:
    """Own one no-overwrite staged prefix; only CPU finalization can qualify it."""

    NORMALIZED = "capture.normalized.staged.jsonl"

    def __init__(self, directory, *, serialize, parse):
        """Bind local stage paths to the collector's original JSON contract."""
        self.directory = Path(directory).resolve()
        self.serialize, self.parse = serialize, parse
        self.normalized = self.directory / self.NORMALIZED

    @contextmanager
    def exclusive(self):
        """Refuse concurrent staged writers without affecting the default collector."""
        # Staged ownership is POSIX-only; the monolithic collector needs no fcntl.
        import fcntl

        lock = self.directory / ".capture-stage.lock"
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(lock, flags, 0o600)
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise CaptureContractError("another capture stage owns this directory") from error
            yield
        finally:
            os.close(descriptor)

    def append_normalized(self, capture):
        """Durably append one measured row, excluding explicit dense chunks."""
        if capture.get("dense_exclusion"):
            return
        with self.normalized.open("ab") as handle:
            handle.write(self.serialize(capture))
            handle.flush()
            os.fsync(handle.fileno())

    def publish_derived(self, path, chunks, expected_sha256):
        """Publish validated derived bytes; preserve interrupted attempts for diagnosis."""
        path = Path(path)
        if path.exists() or path.is_symlink():
            if stable_file_sha256(path, label="derived capture artifact") != expected_sha256:
                raise CaptureContractError("existing derived capture artifact differs")
            return
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = sha256()
            for chunk in chunks:
                handle.write(chunk)
                digest.update(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if digest.hexdigest() != expected_sha256:
            raise CaptureContractError("derived capture artifact digest differs")
        os.link(temporary, path, follow_symlinks=False)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        # Only remove our completed temporary link; interrupted attempts remain untouched.
        temporary.unlink()

    def write_derived(self, name, value):
        """Idempotently publish one staged-only JSON receipt without replacing evidence."""
        payload = self.serialize(value)
        self.publish_derived(self.directory / name, (payload,), sha256(payload).hexdigest())

    def hashes(self, *, runtime):
        """Hash regular, stable staged files without parsing capture arrays."""
        names = ["capture.request.json", "capture.progress.jsonl", self.NORMALIZED]
        if runtime:
            names.append("capture.runtime.json")
        return {
            name: stable_file_sha256(self.directory / name, label="capture stage artifact")
            for name in names
        }

    def seal(self, journal, name, *, runtime, **details):
        """Publish an unqualified stage receipt only after durable artifact writes."""
        hashes = self.hashes(runtime=runtime)
        receipt = {
            "stage_schema_version": 1,
            "stage": name,
            "request_sha256": journal.request_sha,
            "completed_capture_count": journal.count,
            "planned_capture_count": len(journal.invocations),
            "chain_tail_sha256": journal.previous,
            "files": hashes,
            "sizes": {name: (self.directory / name).stat().st_size for name in hashes},
            "eligible_as_capture_manifest": False,
            **details,
        }
        journal._write(f"capture.{name}.json", receipt)
        return self.directory / f"capture.{name}.json"

    def verify(self, receipt, request, *, stage):
        """Verify a trusted producer receipt before adopting its prepared state."""
        if stage == "collected" and receipt.get("runtime_bound") is not True:
            raise CaptureContractError(
                "collection has no validated runtime; restart from the prior source"
            )
        if (
            receipt.get("stage_schema_version") != 1
            or receipt.get("stage") != stage
            or receipt.get("eligible_as_capture_manifest") is not False
            or receipt.get("request_sha256") != sha256(self.serialize(request)).hexdigest()
            or receipt.get("planned_capture_count") != len(request["invocations"])
            or type(receipt.get("completed_capture_count")) is not int
            or not 0 <= receipt["completed_capture_count"] <= len(request["invocations"])
        ):
            raise CaptureContractError("capture stage receipt contract differs")
        if receipt.get("files") != self.hashes(runtime=stage == "collected"):
            raise CaptureContractError("prepared capture bytes changed")
        if receipt.get("sizes") != {
            name: (self.directory / name).stat().st_size for name in receipt["files"]
        }:
            raise CaptureContractError("prepared capture sizes changed")
        if self.parse((self.directory / "capture.request.json").read_bytes()) != request:
            raise CaptureContractError("prepared capture request differs")
        if stage == "collected" and receipt.get("identity_verification") != "passed":
            raise CaptureContractError("collection identity verification failed")

    def verify_inherited_bytes(self, prepared):
        """Ensure appending new rows did not rewrite any prepared prefix byte."""
        for name, expected in prepared["files"].items():
            path = self.directory / name
            if path.is_symlink():
                raise CaptureContractError("staged prefix cannot become a symlink")

            def identity():
                value = path.stat()
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_mode,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            before = identity()
            remaining = prepared["sizes"][name]
            digest = sha256()
            with path.open("rb") as handle:
                while remaining:
                    block = handle.read(min(1024 * 1024, remaining))
                    if not block:
                        raise CaptureContractError("inherited staged prefix was truncated")
                    digest.update(block)
                    remaining -= len(block)
            if before != identity() or digest.hexdigest() != expected:
                raise CaptureContractError("inherited staged prefix changed during collection")
