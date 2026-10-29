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

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import torch.distributed as dist

from modelopt.torch.utils import safe_save

try:
    import fcntl
except ImportError:
    fcntl = None

__all__ = []


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
            except OSError as exc:
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
                safe_save(payload, Path(generation) / f"rank{self.rank}.pth")
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
