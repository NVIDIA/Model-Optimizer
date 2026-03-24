# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Integration test for Docker container launch via run_jobs.

Requires Docker to be installed and running. Uses python:3.12-slim
(lightweight, no GPU needed) to run a trivial script.

Run with: pytest -s  (stdin capture must be disabled for invoke/fabric)
"""

import os
import shutil
import subprocess

import pytest

docker_available = shutil.which("docker") is not None


@pytest.mark.skipif(not docker_available, reason="Docker not available")
class TestDockerLaunch:
    """End-to-end Docker launch test using subprocess to avoid pytest stdin capture issues."""

    def test_echo_script_via_launch(self, tmp_path):
        """Launch a Docker container via launch.py subprocess that runs 'echo hello'."""
        # Create a trivial script
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script = script_dir / "hello.sh"
        script.write_text("#!/bin/bash\necho 'HELLO_FROM_DOCKER'\n")
        script.chmod(0o755)

        # Create a YAML config
        yaml_content = """
job_name: test_hello
pipeline:
  task_0:
    script: scripts/hello.sh
    slurm_config:
      _factory_: "slurm_factory"
      container: python:3.12-slim
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)

        # Run launch.py as a subprocess (avoids pytest stdin capture issues)
        launcher_dir = os.path.join(os.path.dirname(__file__), "..")
        launcher_dir = os.path.abspath(launcher_dir)

        result = subprocess.run(
            [
                "uv",
                "run",
                "launch.py",
                "--yaml",
                str(yaml_path),
                f"hf_local={tmp_path}",
                "--yes",
            ],
            cwd=launcher_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # Check output
        assert "Version Report" in result.stdout
        assert "Launching" in result.stdout or "Entering Experiment" in result.stdout

    def test_failing_script_via_launch(self, tmp_path):
        """Launch a Docker container that exits 1 — launch.py should not crash."""
        script_dir = tmp_path / "scripts"
        script_dir.mkdir()
        script = script_dir / "fail.sh"
        script.write_text("#!/bin/bash\necho 'FAILING'\nexit 1\n")
        script.chmod(0o755)

        yaml_content = """
job_name: test_fail
pipeline:
  task_0:
    script: scripts/fail.sh
    slurm_config:
      _factory_: "slurm_factory"
      container: python:3.12-slim
"""
        yaml_path = tmp_path / "fail_test.yaml"
        yaml_path.write_text(yaml_content)

        launcher_dir = os.path.join(os.path.dirname(__file__), "..")
        launcher_dir = os.path.abspath(launcher_dir)

        result = subprocess.run(
            [
                "uv",
                "run",
                "launch.py",
                "--yaml",
                str(yaml_path),
                f"hf_local={tmp_path}",
                "--yes",
            ],
            cwd=launcher_dir,
            capture_output=True,
            text=True,
            timeout=300,
        )

        # launch.py should complete (exit 0) even if the job fails
        # The job failure is reported in stdout
        assert "Version Report" in result.stdout
