# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Executor interface for orchestration backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..schema import AttemptSpec, JobHandle, JobStatus

__all__ = ["Executor"]


class Executor(ABC):
    """Placement and process lifecycle backend."""

    backend: str

    @abstractmethod
    def submit(self, attempt: AttemptSpec) -> JobHandle:
        raise NotImplementedError

    @abstractmethod
    def poll(self, handles: Sequence[JobHandle]) -> list[JobStatus]:
        raise NotImplementedError

    @abstractmethod
    def cancel(self, handles: Sequence[JobHandle]) -> None:
        raise NotImplementedError

    @abstractmethod
    def recover(self, handle: JobHandle) -> JobStatus:
        raise NotImplementedError

    def fetch_logs(self, handle: JobHandle) -> tuple[str, ...]:
        return tuple(handle.metadata.get("log_paths", ()))
