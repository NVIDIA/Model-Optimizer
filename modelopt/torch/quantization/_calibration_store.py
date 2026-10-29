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

"""Atomic distributed calibration checkpoints with bounded generation retention."""

import io
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist

from modelopt.torch.utils import safe_load, safe_save

try:
    import fcntl
except ImportError:
    fcntl = None

__all__ = []


class _LimitedWriter(io.BufferedWriter):
    def __init__(self, stream, limit):
        super().__init__(stream)
        self.limit = limit

    def write(self, data):
        if self.limit is not None and self.tell() + len(data) > self.limit:
            raise ValueError("Calibration artifact exceeds its byte limit")
        return super().write(data)


def _collective_check(error, signature=None):
    if dist.is_initialized():
        errors: list[tuple[str | None, str | None]] = [(None, None)] * dist.get_world_size()
        dist.all_gather_object(errors, (error, signature))
    else:
        errors = [(error, signature)]
    if any(e for e, _ in errors) or len({s for _, s in errors}) != 1:
        raise ValueError("Calibration worker validation failed: " + str(errors))


class _CalibrationStore:
    def __init__(self, directory, candidate=None):
        if fcntl is None:
            raise ValueError("Calibration workers require POSIX advisory locks")
        self.directory = Path(directory).absolute()
        self.candidate = candidate
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.max_artifact_bytes = None

    def check_checkpoint(self, path):
        error = None
        try:
            if (
                path is not None
                and Path(path).is_file()
                and self.max_artifact_bytes is not None
                and Path(path).stat().st_size > self.max_artifact_bytes
            ):
                raise ValueError("Calibration artifact exceeds its byte limit")
        except (OSError, ValueError) as exc:
            error = str(exc)
        _collective_check(error)

    def load(self, path, limit):
        with Path(path).open("rb") as stream:
            if limit is not None and os.fstat(stream.fileno()).st_size > limit:
                raise ValueError("Calibration artifact exceeds its byte limit")
            return safe_load(stream)

    def _reclaim(self, destination):
        # The destination lock excludes writers/readers of replaceable checkpoints. Reclaim
        # only this store's private generation namespace, including interrupted publications.
        current = destination.resolve()
        stale = []
        for path in destination.parent.glob("." + destination.name + "-????????"):
            if path == current:
                continue
            if (
                path.is_symlink()
                or not path.is_dir()
                or any(
                    not (
                        p.is_file() and not p.is_symlink() and re.fullmatch(r"rank\d+\.pth", p.name)
                    )
                    and not (p.name == "link" and p.is_symlink() and p.resolve() == path)
                    for p in path.iterdir()
                )
            ):
                raise ValueError("Refusing to reclaim an unmanaged checkpoint generation")
            stale.append(path)
        for path in stale:
            shutil.rmtree(path)

    @contextmanager
    def exclusive(self, destination=None):
        path = Path(destination).absolute() if destination else self.directory / str(self.candidate)
        lock = None
        error = None
        if self.rank == 0:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                lock = path.with_name("." + path.name + ".lock").open("a")
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                if self.max_artifact_bytes is not None:
                    self._reclaim(path)
            except (OSError, ValueError) as exc:
                error = str(exc)
        try:
            _collective_check(error, str(path))
            yield
        finally:
            if lock is not None:
                lock.close()

    def _publish(self, destination, payload):
        error, generation, previous = None, None, None
        if self.rank == 0:
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists() and not destination.is_symlink():
                    raise ValueError(
                        "Worker checkpoints require a new path or a managed checkpoint symlink"
                    )
                if destination.is_symlink():
                    previous = destination.resolve(strict=True)
                    if (
                        previous.parent != destination.parent.resolve()
                        or not previous.name.startswith("." + destination.name + "-")
                        or {p.name for p in previous.iterdir()}
                        != {f"rank{rank}.pth" for rank in range(self.world_size)}
                    ):
                        raise ValueError("Refusing to replace an unmanaged checkpoint generation")
                generation = tempfile.mkdtemp(
                    prefix="." + destination.name + "-", dir=destination.parent
                )
            except Exception as exc:
                error = str(exc)
        _collective_check(error)
        if dist.is_initialized():
            values = [generation]
            dist.broadcast_object_list(values, src=0)
            generation = values[0]
        assert generation is not None
        try:
            error = None
            try:
                with (
                    (Path(generation) / f"rank{self.rank}.pth").open("wb") as stream,
                    _LimitedWriter(stream, self.max_artifact_bytes) as writer,
                ):
                    safe_save(payload, writer)
            except Exception as exc:
                error = str(exc)
            _collective_check(error)
            if self.rank == 0:
                try:
                    temporary = Path(generation) / "link"
                    os.symlink(generation, temporary)
                    os.replace(temporary, destination)
                except Exception as exc:
                    error = str(exc)
            _collective_check(error)
            # The coordinator holds the checkpoint lock through load, search and publication.
            # Candidate generations are never replaced, so no artifact reader can lose its input.
            error = None
            if self.rank == 0 and previous is not None:
                try:
                    shutil.rmtree(previous)
                except OSError as exc:
                    error = str(exc)
            _collective_check(error)
        except Exception:
            if self.rank == 0 and destination.resolve() != Path(generation):
                shutil.rmtree(generation, ignore_errors=True)
            raise
