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

"""Bounded metadata and process leases for the automatic HF example."""

import json
import os
import re
from contextlib import ExitStack, contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None

__all__ = []


class PipelineWorkspace:
    def __init__(self, directory, search_digest, metadata_bytes):
        if fcntl is None:
            raise ValueError("Automatic pipeline execution requires POSIX advisory locks")
        self.directory = Path(directory)
        self.root = self.directory / search_digest
        self.metadata_bytes = metadata_bytes

    @contextmanager
    def _lock(self, name, shared=False, blocking=False):
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / name).open("a") as stream:
            fcntl.flock(
                stream,
                (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | (0 if blocking else fcntl.LOCK_NB),
            )
            yield

    def bind(self, spec):
        # One search per workspace bounds accumulation across repeated launcher submissions;
        # target budgets and export destinations do not change the reusable search statistics.
        contract = {
            key: value
            for key, value in spec.items()
            if key not in {"recipe_yaml", "candidate", "candidate_signature", "export_path"}
        }
        self.publish(self.directory / "pipeline.json", json.dumps(contract, sort_keys=True))

    def publish(self, path, content, *, reserve_bytes=0, available_bytes=0):
        path = Path(path)
        encoded = content.encode()
        if path.parent not in (self.directory, self.root):
            raise ValueError("Pipeline metadata must stay inside its workspace")
        with self._lock(".metadata.lock", blocking=True):
            if any(
                item.name not in {self.root.name, "pipeline.json", ".metadata.tmp"}
                and not re.fullmatch(r"\.(metadata|run|slot-\d+)\.lock", item.name)
                for item in self.directory.iterdir()
            ):
                raise ValueError("Automatic execution requires a dedicated search workspace")
            temporary = self.directory / ".metadata.tmp"
            if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
                raise ValueError("Refusing an unmanaged metadata temporary")
            temporary.unlink(missing_ok=True)
            if path.is_symlink():
                raise ValueError("Pipeline metadata cannot be a symlink")
            current_bytes = 0
            for parent in (self.directory, self.root):
                if parent.is_symlink():
                    raise ValueError("Pipeline metadata directories cannot be symlinks")
                if parent.exists():
                    current_bytes += sum(
                        item.stat().st_size
                        for item in parent.iterdir()
                        if item.is_file() and not item.is_symlink()
                    )
            if current_bytes > self.metadata_bytes:
                raise ValueError("Pipeline metadata exceeds its byte limit")
            if path.exists():
                if path.read_bytes() != encoded:
                    raise ValueError("Conflicting automatic pipeline metadata")
                return
            if current_bytes + len(encoded) > self.metadata_bytes:
                raise ValueError("Pipeline metadata exceeds its byte limit")
            if reserve_bytes:
                reserved = sum(
                    json.loads(item.read_text())["bytes"]
                    for item in self.root.glob("export-*.json")
                )
                if reserved + reserve_bytes > available_bytes:
                    raise ValueError("Retained exports exceed the workspace storage limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                temporary.write_bytes(encoded)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)

    @contextmanager
    def run(self, candidate, concurrency):
        # The coordinator excludes all workers, including duplicate launch submissions.
        # Kernel-owned leases are released on process death, so retries need no PID guessing.
        with ExitStack() as stack:
            stack.enter_context(self._lock(".run.lock", shared=candidate is not None))
            if candidate is not None:
                for slot in range(concurrency):
                    try:
                        stack.enter_context(self._lock(f".slot-{slot}.lock"))
                        break
                    except BlockingIOError:
                        continue
                else:
                    raise ValueError("Automatic pipeline worker concurrency limit reached")
            yield
