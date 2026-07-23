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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Interactive terminal controls for the Puzzletron campaign orchestrator."""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from enum import Enum
from typing import TextIO

__all__ = ["InteractiveControlRequest", "ShutdownAction", "TerminalControls"]


class ShutdownAction(str, Enum):
    """User decision after requesting interactive controller shutdown."""

    CANCEL = "cancel"
    DETACH = "detach"
    CONTINUE = "continue"


class InteractiveControlRequest(Exception):
    """Interrupt controller work and open the interactive quit menu."""


class TerminalControls:
    """Read single-key controller commands without adding a terminal dependency."""

    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stderr
        try:
            self._fd = self.input_stream.fileno()
        except (AttributeError, OSError):
            self._fd = None
        self.enabled = bool(
            os.name == "posix"
            and self._fd is not None
            and getattr(self.input_stream, "isatty", lambda: False)()
        )
        self._saved_attrs = None
        self._active = False

    def start(self) -> None:
        if not self.enabled or self._active:
            return
        assert self._fd is not None
        try:
            self._saved_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except (OSError, termios.error):
            self.enabled = False
            self._saved_attrs = None
            return
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        assert self._fd is not None and self._saved_attrs is not None
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        finally:
            self._active = False

    def poll_quit(self) -> bool:
        """Return whether an available keypress requests the quit menu."""

        if not self._active or self._fd is None:
            return False
        readable, _, _ = select.select([self._fd], [], [], 0)
        if not readable:
            return False
        return os.read(self._fd, 1).lower() == b"q"

    @staticmethod
    def action_for_choice(choice: str) -> ShutdownAction | None:
        normalized = choice.strip().lower()
        if normalized in {"c", "cancel"}:
            return ShutdownAction.CANCEL
        if normalized in {"d", "detach", "k", "keep"}:
            return ShutdownAction.DETACH
        if normalized in {"", "r", "resume", "continue"}:
            return ShutdownAction.CONTINUE
        return None

    def choose_shutdown(self) -> ShutdownAction:
        """Prompt in cooked mode and return one explicit shutdown decision."""

        was_active = self._active
        if was_active:
            assert self._fd is not None and self._saved_attrs is not None
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        try:
            while True:
                self.output_stream.write(
                    "\nQuit Puzzletron? [c] cancel jobs + quit, "
                    "[k] keep jobs running + quit, [r] resume (default): "
                )
                self.output_stream.flush()
                choice = self.input_stream.readline()
                if choice == "":
                    return ShutdownAction.CANCEL
                action = self.action_for_choice(choice)
                if action is not None:
                    return action
                self.output_stream.write("Please choose c, k, or r.\n")
        finally:
            if was_active:
                assert self._fd is not None
                tty.setcbreak(self._fd)

    def choose_revisions(
        self, prompt: str, revision_ids: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Select revision IDs in cooked mode using numbers or exact IDs."""

        if not self.enabled:
            raise RuntimeError("manual selection requires an interactive terminal")
        was_active = self._active
        if was_active:
            assert self._fd is not None and self._saved_attrs is not None
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)
        try:
            self.output_stream.write(f"\n{prompt}\n")
            for index, revision_id in enumerate(revision_ids, start=1):
                self.output_stream.write(f"  {index:>4}. {revision_id}\n")
            while True:
                self.output_stream.write(
                    "Choose comma-separated numbers/IDs, 'all', or 'none': "
                )
                self.output_stream.flush()
                choice = self.input_stream.readline().strip()
                if choice.lower() == "all":
                    return revision_ids
                if choice.lower() == "none":
                    return ()
                selected = []
                valid = True
                for value in (item.strip() for item in choice.split(",") if item.strip()):
                    if value.isdigit() and 1 <= int(value) <= len(revision_ids):
                        selected.append(revision_ids[int(value) - 1])
                    elif value in revision_ids:
                        selected.append(value)
                    else:
                        valid = False
                        self.output_stream.write(f"Unknown selection: {value}\n")
                        break
                if valid and selected:
                    return tuple(dict.fromkeys(selected))
        finally:
            if was_active:
                assert self._fd is not None
                tty.setcbreak(self._fd)
