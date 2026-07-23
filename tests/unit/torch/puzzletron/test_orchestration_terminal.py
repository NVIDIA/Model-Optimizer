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

from __future__ import annotations

import io

from puzzletron_orchestrator.terminal import ShutdownAction, TerminalControls


def test_terminal_controls_poll_q_and_map_quit_choices(monkeypatch):
    class _TTY(io.StringIO):
        def fileno(self) -> int:
            return 42

        def isatty(self) -> bool:
            return True

    stream = _TTY()
    controls = TerminalControls(input_stream=stream, output_stream=io.StringIO())
    monkeypatch.setattr("puzzletron_orchestrator.terminal.termios.tcgetattr", lambda _fd: [1])
    monkeypatch.setattr("puzzletron_orchestrator.terminal.termios.tcsetattr", lambda *_args: None)
    monkeypatch.setattr("puzzletron_orchestrator.terminal.tty.setcbreak", lambda _fd: None)
    monkeypatch.setattr(
        "puzzletron_orchestrator.terminal.select.select",
        lambda *_args: ([42], [], []),
    )
    monkeypatch.setattr("puzzletron_orchestrator.terminal.os.read", lambda *_args: b"q")

    controls.start()
    assert controls.poll_quit() is True
    assert controls.action_for_choice("c") is ShutdownAction.CANCEL
    assert controls.action_for_choice("keep") is ShutdownAction.DETACH
    assert controls.action_for_choice("") is ShutdownAction.CONTINUE
    controls.stop()
