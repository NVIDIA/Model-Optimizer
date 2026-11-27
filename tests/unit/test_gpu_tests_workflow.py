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

"""Fixed two-GPU measurement routing and fail-closed receipts, without GPU execution."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]


def _workflow():
    return yaml.safe_load((REPO / ".github/workflows/gpu_tests.yml").read_text())


@pytest.fixture
def measurement():
    path = REPO / ".github/scripts/fsdp2_measurement.py"
    spec = importlib.util.spec_from_file_location("fsdp2_measurement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _receipts(module, tmp_path):
    scratch = tmp_path / "pytest"
    scratch.mkdir()
    for rank in range(2):
        common = {
            "rank": rank,
            "world": 2,
            "torch": "test",
            "transformers": "test",
            "cuda": "test",
            "device": "test",
        }
        reader = dict(
            common,
            measurement="nested_fsdp_shared_read",
            exporter=False,
            lifecycle="resharded_unchanged",
            full_groups=5,
            full_bytes=40 * 1024**2,
            persistent_payload_bytes=20 * 1024**2,
            baseline_bytes=100,
            peak_bytes=200,
            incremental_peak_bytes=100,
            seconds=0.25,
            shape=[2048, 2048],
            dtype="bfloat16",
        )
        writer = dict(
            common,
            measurement="native_fsdp_fp8_export",
            actual_writer=True,
            lifecycle="resharded",
            false_windows=0,
            windows=[
                {
                    "layer": n,
                    "writeback": True,
                    "baseline_bytes": 100,
                    "peak_bytes": 200,
                    "incremental_peak_bytes": 100,
                    "seconds": 0.5,
                }
                for n in range(4)
            ],
        )
        for name, data in [
            (f"nested-read-{rank}.json", reader),
            (f"cuda-fsdp-export-{rank}.json", writer),
        ]:
            (scratch / name).write_text(json.dumps(data))
    junit = tmp_path / "junit.xml"
    junit.write_text(
        "<testsuites><testsuite>"
        + "".join(
            f'<testcase classname="{node.split("::")[0][:-3].replace("/", ".")}" name="{node.split("::")[1]}"/>'
            for node in module.NODEIDS
        )
        + "</testsuite></testsuites>"
    )
    return scratch, junit


def test_receipt_preserves_actual_values(measurement, tmp_path):
    scratch, junit = _receipts(measurement, tmp_path)
    records = measurement.validate_results(scratch, junit)
    assert len(records) == 4
    assert records[0]["incremental_peak_bytes"] == 100
    assert records[2]["windows"][0]["seconds"] == 0.5


@pytest.mark.parametrize(
    "corruption",
    ["skip", "missing", "duplicate", "world", "rank", "kind", "lifecycle", "writer", "windows"],
)
def test_receipts_fail_closed(measurement, tmp_path, corruption):
    scratch, junit = _receipts(measurement, tmp_path)
    path = scratch / "nested-read-0.json"
    if corruption == "skip":
        junit.write_text(junit.read_text().replace("/>", "><skipped/></testcase>", 1))
    elif corruption == "missing":
        path.unlink()
    elif corruption == "duplicate":
        (scratch / "duplicate").mkdir()
        (scratch / "duplicate" / path.name).write_text(path.read_text())
    else:
        if corruption in ("writer", "windows"):
            path = scratch / "cuda-fsdp-export-0.json"
        data = json.loads(path.read_text())
        key, value = {
            "world": ("world", 1),
            "rank": ("rank", 1),
            "kind": ("measurement", "proxy"),
            "lifecycle": ("lifecycle", "unknown"),
            "writer": ("actual_writer", False),
            "windows": ("windows", []),
        }[corruption]
        data[key] = value
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        measurement.validate_results(scratch, junit)


@pytest.mark.timeout(5)
def test_exact_two_devices_and_source_sha(measurement, monkeypatch, tmp_path):
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
    with pytest.raises(ValueError, match="exactly two"):
        measurement.check_devices(cuda)
    cuda.device_count = lambda: 2
    measurement.check_devices(cuda)
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


def test_focused_pytest_command_and_failure_status(measurement, tmp_path, monkeypatch):
    monkeypatch.setattr(measurement, "capture_metadata", lambda: {"sha": "abc"})
    monkeypatch.setattr(
        measurement.subprocess, "run", Mock(return_value=SimpleNamespace(returncode=7))
    )
    result = measurement.run(tmp_path / "output")
    assert result == 7
    args, kwargs = measurement.subprocess.run.call_args
    command = args[0]
    assert [arg for arg in command if arg.startswith("tests/")] == list(measurement.NODEIDS)
    assert all(arg not in command for arg in ("--run-manual", "--run-nightly", "--cov", "-n"))
    assert "--no-cov" in command and "--force-reruns" in command
    assert kwargs["env"]["PYTEST_ADDOPTS"] == kwargs["env"]["COVERAGE_PROCESS_START"] == ""
    assert kwargs["env"]["HF_HUB_OFFLINE"] == kwargs["env"]["HF_DATASETS_OFFLINE"] == "1"
    assert not (tmp_path / "output/receipt.json").exists()


def test_measurement_check_names_and_coverage_are_isolated():
    workflow = _workflow()
    jobs = workflow["jobs"]
    assert ".github/scripts/fsdp2_measurement.py" in jobs["pr-gate"]["with"]["files"].splitlines()
    assert "fsdp2-measurement" in workflow["concurrency"]["group"]
    assert "fsdp2-measurement-gate" in jobs["pr-gate"]["name"]
    assert (
        "fsdp2-measurement-full-suite-check-not-applicable" in jobs["gpu-pr-required-check"]["name"]
    )
    steps = jobs["gpu-tests"]["steps"]
    full = next(step for step in steps if step.get("name") == "Run gpu tests")
    focused = next(step for step in steps if step.get("name") == "Run fixed FSDP2 measurements")
    assert focused["run"].endswith("nox -s ${{ matrix.nox_session }}")
    assert focused["env"]["FSDP2_MEASUREMENT_RUNNER"] == jobs["gpu-tests"]["runs-on"]
    assert focused["env"]["PYTEST_ADDOPTS"] == focused["env"]["COVERAGE_PROCESS_START"] == ""
    assert "--run-manual --run-nightly" in full["env"]["PYTEST_ADDOPTS"]
    coverage = next(step for step in steps if step.get("name") == "Upload GPU coverage to Codecov")
    assert coverage["if"] == full["if"]
    artifact = next(step for step in steps if step.get("uses") == "actions/upload-artifact@v7")
    assert "always()" in artifact["if"] and "fsdp2-measurement" in artifact["if"]
    assert artifact["with"]["path"] == "${{ runner.temp }}/fsdp2-measurement"
