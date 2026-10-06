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

import importlib.util

import pytest

from .parser import parse_markdown
from .runner import run_scenario

__all__ = ["execute_scenario", "scenario_parameters"]


def scenario_parameters(path):
    """Collect every scenario, enforcing a documented disposition for every fence."""
    result = []
    for scenario in parse_markdown(path, require_coverage=True):
        marks = [pytest.mark.timeout(scenario.timeout_seconds + 30)]
        if scenario.manual:
            marks.append(pytest.mark.manual)
        result.append(pytest.param(scenario, id=scenario.id, marks=marks))
    return result


def execute_scenario(scenario, repo, tmp_path, monkeypatch):
    """Check declared resources and execute the same runner for any example."""
    for dependency in scenario.requires:
        try:
            available = importlib.util.find_spec(dependency)
        except ModuleNotFoundError:
            available = None
        if available is None:
            pytest.skip(f"README scenario requires {dependency}")
    if scenario.profile == "gpu" or scenario.min_gpus:
        # GPU discovery is deliberately deferred until execution, not collection.
        import torch

        needed = max(1, scenario.min_gpus)
        if torch.cuda.device_count() < needed:
            pytest.skip(f"README scenario requires {needed} CUDA devices")
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
