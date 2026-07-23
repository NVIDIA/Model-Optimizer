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

"""Rich terminal dashboard for live Puzzletron campaign progress."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, TextIO

__all__ = [
    "StageView",
    "TerminalDashboard",
    "format_duration",
    "progress_fraction",
    "progress_eta",
]

_RATIO = re.compile(r"(?<![\d.])(?P<current>\d+)\s*/\s*(?P<total>\d+)(?![\d.])")
_PERCENT = re.compile(r"(?P<percent>\d+(?:\.\d+)?)\s*%")


@dataclass(frozen=True)
class StageView:
    """One rendered campaign-stage row."""

    stage_id: str
    display_name: str
    status: str
    nodes: int
    tasks: int
    gpus: int
    progress: str
    elapsed_seconds: float | None = None
    eta_seconds: float | None = None
    current: float | None = None
    total: float | None = None


def progress_fraction(detail: str | None) -> tuple[float, float] | None:
    """Extract the most relevant current/total progress pair from a summary."""

    if not detail:
        return None
    ratios = list(_RATIO.finditer(detail))
    if ratios:
        match = ratios[-1]
        current = float(match.group("current"))
        total = float(match.group("total"))
        if total > 0 and 0 <= current <= total:
            return current, total
    match = _PERCENT.search(detail)
    if match:
        percent = float(match.group("percent"))
        if 0 <= percent <= 100:
            return percent, 100.0
    return None


def progress_eta(
    elapsed_seconds: float | None,
    current: float | None,
    total: float | None,
) -> float | None:
    """Estimate remaining duration from elapsed wall time and completed fraction."""

    if elapsed_seconds is None or current is None or total is None:
        return None
    if elapsed_seconds < 0 or total <= 0 or current <= 0 or current >= total:
        return None
    return elapsed_seconds * (total - current) / current


def format_duration(seconds: float | None, *, approximate: bool = False) -> str:
    """Format a duration for the compact dashboard columns."""

    if seconds is None or seconds < 0:
        return "—"
    rounded = int(seconds + 0.5)
    hours, remainder = divmod(rounded, 3600)
    minutes, secs = divmod(remainder, 60)
    value = f"{hours:02d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"
    return f"~{value}" if approximate else value


def _progress_bar(current: float | None, total: float | None, *, width: int = 14) -> str:
    if current is None or total is None or total <= 0:
        return ""
    fraction = min(1.0, max(0.0, current / total))
    complete = min(width, int(fraction * width))
    marker = "╸" if 0 < fraction < 1 and complete < width else ""
    empty = width - complete - len(marker)
    return f"{'━' * complete}{marker}{'─' * max(0, empty)} {fraction:>4.0%}"


class TerminalDashboard:
    """Maintain one inline-updating Rich table when stderr is interactive."""

    def __init__(self, *, stream: TextIO) -> None:
        self.enabled = False
        self._live: Any = None
        self._started = False
        self._last_event = "controller starting"
        try:
            from rich.console import Console
            from rich.live import Live

            console = Console(file=stream, force_terminal=True)
            self._live = Live(
                console=console,
                refresh_per_second=4,
                transient=False,
                vertical_overflow="visible",
            )
            self.enabled = True
        except ImportError:
            return

    def set_last_event(self, message: str) -> None:
        self._last_event = message

    def update(
        self,
        stages: list[StageView],
        *,
        campaign_elapsed: float,
        drain_pending: bool,
    ) -> None:
        if not self.enabled:
            return
        renderable = self._render(
            stages,
            campaign_elapsed=campaign_elapsed,
            drain_pending=drain_pending,
        )
        if not self._started:
            self._live.start()
            self._started = True
        self._live.update(renderable, refresh=True)

    def stop(self) -> None:
        if self.enabled and self._started:
            self._live.stop()
            self._started = False

    def _render(
        self,
        stages: list[StageView],
        *,
        campaign_elapsed: float,
        drain_pending: bool,
    ) -> Any:
        from rich.console import Group
        from rich.panel import Panel
        from rich.spinner import Spinner
        from rich.table import Table
        from rich.text import Text

        completed = sum(stage.status == "completed" for stage in stages)
        running = sum(stage.status == "running" for stage in stages)
        suffix = "  •  halted pending drain" if drain_pending else ""
        title = (
            f"Puzzletron Campaign  •  elapsed {format_duration(campaign_elapsed)}  •  "
            f"{running} running  •  {completed}/{len(stages)} complete{suffix}"
        )
        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("Status", width=11, no_wrap=True)
        table.add_column("Stage", ratio=2, no_wrap=True)
        table.add_column("Alloc", width=20, no_wrap=True)
        table.add_column("Progress", ratio=3)
        table.add_column("Elapsed", width=9, justify="right", no_wrap=True)
        table.add_column("ETA", width=10, justify="right", no_wrap=True)
        styles = {
            "completed": ("✓", "green"),
            "running": ("●", "blue"),
            "pending": ("○", "dim"),
            "waiting": ("⏸", "yellow"),
            "failed": ("!", "bold red"),
            "blocked": ("⊘", "red"),
        }
        for stage in stages:
            symbol, style = styles.get(stage.status, ("?", "dim"))
            status_renderable = (
                Spinner("dots", text=Text(" running", style=style), style=style)
                if stage.status == "running"
                else Text(f"{symbol} {stage.status}", style=style)
            )
            progress = Text(stage.progress)
            bar = _progress_bar(stage.current, stage.total)
            if bar:
                progress.append(f"  {bar}", style="blue")
            allocation = f"{stage.nodes}n · {stage.tasks}t · {stage.gpus}g"
            table.add_row(
                status_renderable,
                stage.display_name,
                allocation,
                progress,
                format_duration(stage.elapsed_seconds),
                format_duration(stage.eta_seconds, approximate=True),
            )
        footer = Text("Last event", style="dim")
        footer.append(f"  {self._last_event}")
        controls = Text("q / Ctrl-C: quit options", style="dim")
        return Panel(
            Group(table, Text(""), footer, controls),
            title=title,
            border_style="blue",
        )
