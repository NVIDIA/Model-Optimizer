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
"""Linux subreaper for a single deployment. Run in a fresh, single-threaded process."""

import contextlib
import ctypes
import errno
import os
import select
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def _reap_workers(pgid, deadline):
    """After the group kill, reap exited children and stop any live detached subtrees."""
    escaped = False
    while time.monotonic() < deadline:
        # A single-threaded subreaper with no remaining children owns no live descendants.
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return escaped
        if pid:
            continue
        for pid in map(int, Path("/proc/thread-self/children").read_text().split()):
            # This is our own child. Until reaped, its PID cannot be reused.
            if os.waitpid(pid, os.WNOHANG)[0] == 0 and os.getpgid(pid) != pgid:
                os.kill(pid, signal.SIGKILL)
                escaped = True
        time.sleep(0.01)
    raise TimeoutError("Deployment workers did not exit")


def _supervise(control, grace, command):
    # Set this only in the supervisor. pytest may already own unrelated children.
    if ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError("Cannot adopt deployment workers")
    try:
        child = subprocess.Popen(command, start_new_session=True)
    except OSError as exc:
        control.sendall(f"errno {exc.errno or errno.EIO}".encode())
        return

    try:
        cancel = select.poll()
        cancel.register(control, select.POLLIN)
        while not os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG):
            # EOF also cancels the command if pytest exits or times out.
            if cancel.poll(50):
                break
    finally:
        # Keep the leader waitable until killpg, so its group ID is still reserved.
        deadline = time.monotonic() + grace
        with contextlib.suppress(ProcessLookupError):
            os.killpg(child.pid, signal.SIGKILL)
        code = child.wait(timeout=max(0, deadline - time.monotonic()))
        escaped = _reap_workers(child.pid, deadline)
    if not escaped:
        control.sendall(str(code).encode())


if __name__ == "__main__":
    with socket.socket(fileno=int(sys.argv[1])) as channel:
        _supervise(channel, float(sys.argv[2]), sys.argv[3:])
