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

"""Terminal logging with an optional Rich live campaign dashboard."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from .dashboard import StageView

__all__ = ["OrchestratorLogger"]

_COLORS = {
    "dim": "\033[2m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "magenta": "\033[35m",
    "red": "\033[31m",
    "yellow": "\033[33m",
    "reset": "\033[0m",
}


class OrchestratorLogger:
    """Write timestamped controller progress to stderr with optional ANSI color."""

    def __init__(self, *, color: str = "auto", stream: TextIO | None = None) -> None:
        if color not in {"auto", "always", "never"}:
            raise ValueError(f"Unsupported color mode: {color!r}")
        self.stream = stream or sys.stderr
        self.color = color == "always" or (
            color == "auto"
            and "NO_COLOR" not in os.environ
            and bool(getattr(self.stream, "isatty", lambda: False)())
        )
        self._dashboard = None

    @property
    def live(self) -> bool:
        """Return whether an interactive dashboard is currently configured."""

        return self._dashboard is not None and self._dashboard.enabled

    def enable_dashboard(self) -> bool:
        """Enable the Rich live display when output is an interactive terminal."""

        if not self.color or not bool(getattr(self.stream, "isatty", lambda: False)()):
            return False
        from .dashboard import TerminalDashboard

        try:
            dashboard = TerminalDashboard(stream=self.stream)
        except Exception as error:  # noqa: BLE001 - presentation is best effort
            self.warning(f"live dashboard unavailable: {error}")
            return False
        self._dashboard = dashboard if dashboard.enabled else None
        return self.live

    def update_dashboard(
        self,
        stages: list[StageView],
        *,
        campaign_elapsed: float,
        drain_pending: bool = False,
    ) -> None:
        """Refresh the live campaign table."""

        if self._dashboard is not None:
            try:
                self._dashboard.update(
                    stages,
                    campaign_elapsed=campaign_elapsed,
                    drain_pending=drain_pending,
                )
            except Exception as error:  # noqa: BLE001 - presentation must not halt jobs
                dashboard = self._dashboard
                self._dashboard = None
                try:
                    dashboard.stop()
                except Exception:  # noqa: BLE001 - cleanup is best effort
                    pass
                self.warning(f"live dashboard disabled after render error: {error}")

    def stop_dashboard(self) -> None:
        """Finish the live display while preserving its final table."""

        if self._dashboard is not None:
            dashboard = self._dashboard
            self._dashboard = None
            try:
                dashboard.stop()
            except Exception:  # noqa: BLE001 - cleanup is best effort
                pass

    def _paint(self, text: str, color: str) -> str:
        if not self.color:
            return text
        return f"{_COLORS[color]}{text}{_COLORS['reset']}"

    def _write(self, symbol: str, label: str, message: str, color: str) -> None:
        if self._dashboard is not None:
            self._dashboard.set_last_event(f"{symbol} {label}: {message}")
            return
        timestamp = datetime.now().astimezone().strftime("%H:%M:%S")
        prefix = self._paint(f"{symbol} {label:<9}", color)
        clock = self._paint(timestamp, "dim")
        print(f"{clock} {prefix} {message}", file=self.stream, flush=True)

    def banner(self, message: str) -> None:
        self._write("◆", "PUZZLETRON", message, "blue")

    def plan(self, message: str) -> None:
        self._write("▣", "PLAN", message, "blue")

    def stage(self, message: str) -> None:
        self._write("▶", "STAGE", message, "cyan")

    def submit(self, message: str) -> None:
        self._write("↗", "SUBMIT", message, "blue")

    def pending(self, message: str) -> None:
        self._write("◌", "PENDING", message, "yellow")

    def running(self, message: str) -> None:
        self._write("●", "RUNNING", message, "cyan")

    def success(self, message: str) -> None:
        self._write("✓", "SUCCESS", message, "green")

    def skip(self, message: str) -> None:
        self._write("○", "SKIPPED", message, "green")

    def warning(self, message: str) -> None:
        self._write("!", "WARNING", message, "yellow")

    def error(self, message: str) -> None:
        self._write("✗", "FAILED", message, "red")

    def wait(self, message: str) -> None:
        self._write("…", "WAITING", message, "dim")

    def progress(self, message: str) -> None:
        self._write("↳", "PROGRESS", message, "cyan")

    def shutdown(self, message: str) -> None:
        self._write("■", "SHUTDOWN", message, "yellow")
