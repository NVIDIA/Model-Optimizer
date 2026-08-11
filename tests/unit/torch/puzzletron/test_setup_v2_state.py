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

"""Failure-path tests for atomic PuzzleTron v2 wizard state persistence."""

from __future__ import annotations

import pytest

from puzzletron_setup import SetupError
from puzzletron_setup.v2 import state as state_module
from puzzletron_setup.v2.state import WizardState


def test_failed_atomic_replace_preserves_last_resumable_state(tmp_path, monkeypatch):
    state = WizardState.start(tmp_path / "campaign", defaults_path=None)
    state.set_field("model.source", "/models/teacher")
    durable_snapshot = state.path.read_bytes()

    def fail_replace(source, destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr(state_module.os, "replace", fail_replace)

    with pytest.raises(SetupError, match="Cannot save v2 setup state"):
        state.set_field("model.source", "/models/candidate")

    assert state.path.read_bytes() == durable_snapshot
    assert WizardState.resume(state.path).get_field("model.source") == "/models/teacher"
