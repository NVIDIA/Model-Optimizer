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

"""Real Git source qualification belongs outside the fast unit-test lane."""

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


@pytest.mark.timeout(15)
def test_source_sha_uses_temporary_safe_directory_without_persistent_changes(monkeypatch, tmp_path):
    script = Path(__file__).resolve().parents[2] / ".github/scripts/fsdp2_measurement.py"
    spec = importlib.util.spec_from_file_location("fsdp2_git_measurement", script)
    measurement = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(measurement)
    # Real Git refusal without chown or persistent configuration changes.
    checkout = tmp_path / "checkout with spaces"
    checkout.mkdir()
    monkeypatch.chdir(checkout)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    monkeypatch.delenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", raising=False)
    subprocess.run(["git", "init", "-q"], check=True, timeout=5)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-qm",
            "fixture",
        ],
        check=True,
        timeout=5,
    )
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, timeout=5).strip()
    original_config = (checkout / ".git/config").read_bytes()
    monkeypatch.setenv("GITHUB_SHA", sha)
    monkeypatch.setenv("GITHUB_WORKSPACE", str(checkout))
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    refused = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
    )
    assert refused.returncode == 128 and "dubious ownership" in refused.stderr
    assert measurement.check_source() == sha
    monkeypatch.setenv("GITHUB_SHA", "wrong")
    with pytest.raises(ValueError, match="checkout"):
        measurement.check_source()
    monkeypatch.delenv("GITHUB_SHA")
    with pytest.raises(ValueError, match="checkout"):
        measurement.check_source()
    monkeypatch.setenv("GITHUB_SHA", sha)
    for wrong in (str(tmp_path), "checkout with spaces", ""):
        monkeypatch.setenv("GITHUB_WORKSPACE", wrong)
        with pytest.raises(ValueError, match=r"workspace|checkout"):
            measurement.check_source()
    monkeypatch.delenv("GITHUB_WORKSPACE")
    with pytest.raises(ValueError, match=r"workspace|checkout"):
        measurement.check_source()
    assert (checkout / ".git/config").read_bytes() == original_config
    # The exception must not persist even in this temporary repository.
    refused = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
    )
    assert refused.returncode == 128 and "dubious ownership" in refused.stderr
