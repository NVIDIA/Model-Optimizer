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

import pytest

from puzzletron_orchestrator.dashboard import (
    StageView,
    TerminalDashboard,
    format_duration,
    progress_eta,
    progress_fraction,
)
from puzzletron_orchestrator.schema import AttemptSpec, CommandSpec, JobHandle, JobState, JobStatus
from puzzletron_orchestrator.state import CampaignStateStore


def test_progress_fraction_prefers_innermost_stage_counter():
    assert progress_fraction("fixed_smallest probe 1/2: step 71/128, loss 0.2") == (
        71.0,
        128.0,
    )
    assert progress_fraction("validated 4/72 specs (36 subblocks)") == (4.0, 72.0)
    assert progress_fraction("loading model") is None


def test_progress_eta_and_duration_formatting():
    assert progress_eta(60.0, 25.0, 100.0) == 180.0
    assert progress_eta(60.0, 0.0, 100.0) is None
    assert progress_eta(60.0, 100.0, 100.0) is None
    assert format_duration(65.0) == "01:05"
    assert format_duration(3661.0) == "01:01:01"
    assert format_duration(65.0, approximate=True) == "~01:05"
    assert format_duration(None) == "—"


def test_rich_dashboard_renders_allocation_status_eta_and_drain_state():
    console_module = pytest.importorskip("rich.console")
    dashboard = TerminalDashboard(stream=io.StringIO())
    renderable = dashboard._render(
        [
            StageView(
                stage_id="sort_sanity",
                display_name="Sort Sanity Check",
                status="running",
                nodes=1,
                tasks=1,
                gpus=8,
                progress="scoring batch 18/32",
                elapsed_seconds=662,
                eta_seconds=520,
                current=18,
                total=32,
            ),
            StageView(
                stage_id="slicing_sanity",
                display_name="Slicing Sanity Check",
                status="blocked",
                nodes=1,
                tasks=1,
                gpus=1,
                progress="blocked by width_sanity",
            ),
        ],
        campaign_elapsed=900,
        drain_pending=True,
    )
    output = io.StringIO()
    console_module.Console(file=output, force_terminal=False, width=160).print(renderable)
    text = output.getvalue()

    assert "halted pending drain" in text
    assert "Sort Sanity Check" in text
    assert "1n · 1t · 8g" in text
    assert "scoring batch 18/32" in text
    assert "~08:40" in text
    assert "blocked by width_sanity" in text
    assert "q / Ctrl-C: quit options" in text


def test_attempt_timestamps_support_stage_elapsed_time(tmp_path, monkeypatch):
    now = 100.0
    monkeypatch.setattr(
        "puzzletron_orchestrator.state.time.time",
        lambda: now,
    )
    store = CampaignStateStore(tmp_path)
    attempt = AttemptSpec(
        attempt_id="attempt-1",
        work_id="sort_sanity:0",
        stage_id="sort_sanity",
        command=CommandSpec(argv=("python", "worker.py")),
    )
    handle = JobHandle(
        backend="fake",
        handle_id="job-1",
        attempt_id=attempt.attempt_id,
    )
    store.save_attempt(attempt, handle, JobState.RUNNING.value)
    now = 145.0
    store.update_attempt_status(
        attempt.work_id,
        attempt.attempt_id,
        JobStatus(handle=handle, state=JobState.COMPLETED),
    )

    record = store.load_attempt(attempt.work_id, attempt.attempt_id)
    assert record is not None
    assert record["submitted_at"] == 100.0
    assert record["completed_at"] == 145.0
