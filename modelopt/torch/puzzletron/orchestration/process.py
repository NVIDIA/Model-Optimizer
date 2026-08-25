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

"""Shell-free process execution for dependency-light orchestration."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["ProcessResult", "run_argv"]


@dataclass(frozen=True)
class ProcessResult:
    """Captured result from one argv-only child process."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


async def _run_argv_async(argv: tuple[str, ...], *, cwd: str | Path | None) -> ProcessResult:
    # create_subprocess_exec passes argv directly to the child without a command shell.
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(cwd) if cwd is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    returncode = process.returncode
    if returncode is None:
        raise RuntimeError("child process did not terminate after communicate()")
    return ProcessResult(
        args=argv,
        returncode=returncode,
        stdout=stdout.decode(errors="replace"),
        stderr=stderr.decode(errors="replace"),
    )


def run_argv(argv: Sequence[str], *, cwd: str | Path | None = None) -> ProcessResult:
    """Run an explicit argv sequence without a shell and capture text output."""

    return asyncio.run(_run_argv_async(tuple(str(part) for part in argv), cwd=cwd))
