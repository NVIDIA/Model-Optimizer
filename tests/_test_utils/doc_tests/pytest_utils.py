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

"""Common pytest collection and resource checks for executable documentation."""

import pytest

from .parser import resolve_scenarios
from .runner import run_scenario

__all__ = ["execute_scenario", "scenario_parameters"]


def scenario_parameters(path, scenarios, *, excluded=None):
    """Collect every scenario, enforcing a documented disposition for every named fence."""
    return [
        pytest.param(
            scenario, id=scenario.id, marks=pytest.mark.timeout(scenario.timeout_seconds + 30)
        )
        for scenario in resolve_scenarios(path, scenarios, excluded=excluded)
    ]


def execute_scenario(scenario, repo, tmp_path, monkeypatch):
    """Check declared resources and execute the same runner for any example."""
    if scenario.profile == "gpu":
        # GPU discovery is deliberately deferred until execution, not collection.
        import torch

        if not torch.cuda.device_count():
            pytest.skip("README scenario requires a CUDA device")
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "LOCAL_WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        monkeypatch.delenv(key, raising=False)
    print(run_scenario(scenario, repo, tmp_path))
