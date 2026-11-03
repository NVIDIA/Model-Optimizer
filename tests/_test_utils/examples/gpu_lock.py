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
"""A GPU lock that lets pytest-xdist workers share one GPU between small-model tests only."""

import contextlib
import fcntl
from pathlib import Path


@contextlib.contextmanager
def gpu_lock(lock_dir: Path, *, shared: bool):
    """Hold the GPU lock in ``lock_dir``: shared holders run together, an exclusive holder alone.

    A waiting exclusive holder keeps the turnstile, so new shared holders queue behind it. It then
    runs as soon as the shared holders already on the GPU finish, instead of after all of them.
    """
    with (
        open(lock_dir / "gpu.turnstile", "a") as turnstile,
        open(lock_dir / "gpu.lock", "a") as lock,
    ):
        fcntl.flock(turnstile, fcntl.LOCK_EX)
        try:
            fcntl.flock(lock, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        finally:
            fcntl.flock(turnstile, fcntl.LOCK_UN)
        yield
