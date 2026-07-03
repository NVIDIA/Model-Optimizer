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

"""Behavior tests for the repository-local Puzzletron skill.

These tests are deterministic: no GPU, cluster, network, or ModelOpt import.
They exercise the progress scripts and shell payloads. Run with:

    python -m pytest .agents/skills/puzzletron/tests/test_skill.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = SKILL_DIR.parents[2]
MIP_PROGRESS = SKILL_DIR / "mip_progress.py"
ALL_PROGRESS = SKILL_DIR / "all_progress.py"


def run_progress(
    script: Path, tmp_path: Path, log_text: str | None = None
) -> subprocess.CompletedProcess:
    """Run a progress script against an optional temporary log."""
    if log_text is not None:
        (tmp_path / "log.txt").write_text(log_text, encoding="utf-8")

    return subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )


def mip_log(rates: str, *events: str) -> str:
    """Build a minimal MIP progress log."""
    return "\n".join((f"Compression rates: [{rates}]", *events, ""))


@pytest.fixture
def run_skill_payload(tmp_path):
    repo = tmp_path / "target repo"
    config = repo / "configs" / "test.yaml"
    config.parent.mkdir(parents=True)
    config.write_text("test: true\n", encoding="utf-8")
    trace = tmp_path / "trace"
    trace.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fake_torchrun(bin_dir)

    def run(
        payload_name,
        *,
        config_path="configs/test.yaml",
        derive_repo_root=False,
        torchrun_exit=0,
    ):
        cwd = tmp_path
        env = {
            **os.environ,
            "MODELOPT_CONFIG_PATH": config_path,
            "TRACE_DIR": str(trace),
            "FAKE_TORCHRUN_EXIT": str(torchrun_exit),
            "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        }
        if derive_repo_root:
            subprocess.run(
                ["git", "init", str(repo)],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            cwd = repo / "nested" / "workdir"
            cwd.mkdir(parents=True)
        else:
            env["MODELOPT_REPO_ROOT"] = str(repo)

        payload = puzzletron_run_payloads()[payload_name].replace("<nproc_per_node>", "2")
        result = subprocess.run(
            ["bash", "-c", payload],
            cwd=cwd,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result, repo, trace

    return run


# Progress reporting


@pytest.mark.parametrize(
    ("script", "message"),
    [
        (MIP_PROGRESS, "No log.txt found. Run /puzzletron mip first."),
        (ALL_PROGRESS, "No log.txt found. Run /puzzletron all first."),
    ],
    ids=["mip", "all"],
)
def test_progress_reports_missing_log(script, message, tmp_path):
    result = run_progress(script, tmp_path)

    assert result.returncode == 0
    assert result.stdout.strip() == message
    assert result.stderr == ""


def test_mip_progress_reports_completed_sweep(tmp_path):
    log_text = mip_log(
        "0.5, 0.6",
        "[2026-06-30 10:00:00] compression_rate=0.5",
        "[2026-06-30 10:01:00] compression_rate=0.6",
        "[2026-06-30 10:02:00] Results written to: /tmp/results.csv",
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[DONE\]\s+compression_rate=0\.5\s+1m 0s", result.stdout)
    assert re.search(r"\[DONE\]\s+compression_rate=0\.6\s+1m 0s", result.stdout)
    assert "Completed: 2/2 compression rates" in result.stdout
    assert "Remaining: done estimated" in result.stdout
    assert "Results:   /tmp/results.csv" in result.stdout
    assert result.stderr == ""


def test_mip_progress_reports_first_rate_running_with_next_rate_pending(tmp_path):
    log_text = mip_log(
        "0.5, 0.6",
        "[2026-06-30 10:00:00] compression_rate=0.5",
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[RUNNING\]\s+compression_rate=0\.5", result.stdout)
    assert re.search(r"\[ \]\s+compression_rate=0\.6\s+pending", result.stdout)
    assert "Completed: 0/2 compression rates" in result.stdout
    assert result.stderr == ""


def test_mip_progress_reports_later_rate_running_with_final_rate_pending(tmp_path):
    log_text = mip_log(
        "0.5, 0.6, 0.7",
        "[2026-06-30 10:00:00] compression_rate=0.5",
        "[2026-06-30 10:01:00] compression_rate=0.6",
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[DONE\]\s+compression_rate=0\.5\s+1m 0s", result.stdout)
    assert re.search(r"\[RUNNING\]\s+compression_rate=0\.6", result.stdout)
    assert re.search(r"\[ \]\s+compression_rate=0\.7\s+pending", result.stdout)
    assert "Completed: 1/3 compression rates" in result.stdout
    assert result.stderr == ""


def test_mip_progress_uses_next_rate_that_started(tmp_path):
    log_text = mip_log(
        "0.5, 0.6, 0.7",
        "[2026-06-30 10:00:00] compression_rate=0.5",
        "[2026-06-30 10:02:00] compression_rate=0.7",
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[DONE\]\s+compression_rate=0\.5\s+2m 0s", result.stdout)
    assert re.search(r"\[ \]\s+compression_rate=0\.6\s+pending", result.stdout)
    assert re.search(r"\[RUNNING\]\s+compression_rate=0\.7", result.stdout)
    assert result.stderr == ""


def test_mip_progress_ignores_timestampless_start(tmp_path):
    result = run_progress(
        MIP_PROGRESS,
        tmp_path,
        mip_log("0.5", "compression_rate=0.5"),
    )

    assert result.returncode == 0
    assert re.search(r"\[ \]\s+compression_rate=0\.5\s+pending", result.stdout)
    assert "[RUNNING]" not in result.stdout
    assert "Completed: 0/1 compression rates" in result.stdout
    assert result.stderr == ""


def test_mip_progress_accepts_timestamped_duplicate_after_timestampless_row(tmp_path):
    log_text = mip_log(
        "0.5",
        "compression_rate=0.5",
        "[2026-06-30 10:00:00] compression_rate=0.5",
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[RUNNING\]\s+compression_rate=0\.5", result.stdout)
    assert "Started:   10:00:00" in result.stdout
    assert result.stderr == ""


def test_mip_progress_rejects_duplicate_normalized_rates(tmp_path):
    result = run_progress(
        MIP_PROGRESS,
        tmp_path,
        mip_log("0.5, 0.50", "[2026-06-30 10:00:00] compression_rate=0.5"),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "Invalid log.txt: Compression rates must be unique after normalization."
    )
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("rate", ["..", "nan", "inf"], ids=["malformed", "nan", "inf"])
def test_mip_progress_rejects_malformed_rate_event_without_traceback(rate, tmp_path):
    result = run_progress(
        MIP_PROGRESS,
        tmp_path,
        mip_log("0.5", f"[2026-06-30 10:00:00] compression_rate={rate}"),
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "Invalid log.txt: compression_rate event must contain a finite number."
    )
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize("rates", ["", "0.5, abc"], ids=["empty", "malformed"])
def test_mip_progress_rejects_invalid_rate_list_without_traceback(rates, tmp_path):
    result = run_progress(MIP_PROGRESS, tmp_path, mip_log(rates))

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "Invalid log.txt: Compression rates must be a non-empty comma-separated list of numbers."
    )
    assert "Traceback" not in result.stderr


def test_all_progress_reports_running_pipeline(tmp_path):
    log_text = "\n".join(
        (
            "[2026-06-30 10:00:00] Puzzletron Progress 1/8: starting pipeline",
            "[2026-06-30 10:01:00] Puzzletron Progress 2/8: converting model",
            "calculate_losses_pipeline: 25% 10/40",
            "",
        )
    )

    result = run_progress(ALL_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[DONE\].*1/8: starting pipeline\s+1m 0s", result.stdout)
    assert re.search(r"\[RUNNING\].*2/8: converting model \(10/40 batches\)", result.stdout)
    assert re.search(r"\[ \].*3/8: pending", result.stdout)
    assert "Completed: 1/8 steps" in result.stdout
    assert result.stderr == ""


def test_mip_progress_reports_running_single_solve(tmp_path):
    log_text = "\n".join(
        (
            "[2026-06-30 10:00:00] Puzzletron Progress 7/8: running MIP",
            "After 12 nodes (3.5 seconds)",
            "",
        )
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert "MIP solve (sweep disabled)" in result.stdout
    assert re.search(r"\[RUNNING\]\s+MIP solve \(12 nodes, 3\.5s\)", result.stdout)
    assert "Started:   10:00:00" in result.stdout
    assert "Remaining: calculating..." in result.stdout
    assert result.stderr == ""


def test_mip_progress_reports_completed_single_solve(tmp_path):
    log_text = "\n".join(
        (
            "[2026-06-30 10:00:00] Puzzletron Progress 7/8: running MIP",
            "[2026-06-30 10:01:00] Results written to: /tmp/single.csv",
            "",
        )
    )

    result = run_progress(MIP_PROGRESS, tmp_path, log_text)

    assert result.returncode == 0
    assert re.search(r"\[DONE\]\s+MIP solve\s+1m 0s", result.stdout)
    assert "Remaining: done" in result.stdout
    assert "Results:   /tmp/single.csv" in result.stdout
    assert result.stderr == ""


def test_all_progress_reports_completed_pipeline(tmp_path):
    events = [
        f"[2026-06-30 10:0{step - 1}:00] Puzzletron Progress {step}/8: step {step}"
        for step in range(1, 9)
    ]
    events.append("[2026-06-30 10:07:00] Results written to: /tmp/puzzle")

    result = run_progress(ALL_PROGRESS, tmp_path, "\n".join((*events, "")))

    assert result.returncode == 0
    assert result.stdout.count("[DONE]") == 8
    assert "Completed: 8/8 steps" in result.stdout
    assert "Remaining: done estimated" in result.stdout
    assert "Results:   /tmp/puzzle" in result.stdout
    assert result.stderr == ""


# Skill documentation and execution contract


def test_skill_readme_links_to_example_readme():
    readme_text = (SKILL_DIR / "README.md").read_text(encoding="utf-8")
    link_match = re.search(r"\[examples/puzzletron/README\.md\]\(([^)]+)\)", readme_text)

    assert link_match is not None
    assert (SKILL_DIR / link_match.group(1)).resolve() == (
        REPO_ROOT / "examples" / "puzzletron" / "README.md"
    ).resolve()


def test_skill_rejects_zero_processes():
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    validation_patterns = re.findall(r"value does not match `([^`]+)`", skill_text)

    assert len(validation_patterns) == 2
    for pattern in validation_patterns:
        assert re.fullmatch(pattern, "1")
        assert re.fullmatch(pattern, "16")
        assert re.fullmatch(pattern, "0") is None


@pytest.mark.parametrize(
    "document",
    [
        SKILL_DIR / "SKILL.md",
        SKILL_DIR / "README.md",
    ],
    ids=["skill", "skill-readme"],
)
def test_skill_documents_transport_neutral_container_execution(document):
    text = document.read_text(encoding="utf-8")

    assert "allocated GPU/container environment" in text
    assert re.search(r"\bdirect(?:ly)?\b", text)
    assert "execution method" in text
    assert "Keep the coding agent on the local workstation." not in text


def test_example_readme_defers_agent_runtime_setup_to_skill_docs():
    text = (REPO_ROOT / "examples" / "puzzletron" / "README.md").read_text(encoding="utf-8")

    assert ".agents/skills/puzzletron/README.md" in text
    assert "MODELOPT_REPO_ROOT" not in text
    assert "MODELOPT_CONFIG_PATH" not in text


@pytest.mark.parametrize("payload_name", ["all", "mip"])
def test_skill_uses_portable_transport_neutral_payload(payload_name):
    payload = puzzletron_run_payloads()[payload_name]

    assert "/workspace/Model-Optimizer" not in payload
    assert "export PYTHONPATH" not in payload
    assert re.search(r"\b(?:ssh|sbatch|srun)\b", payload) is None


@pytest.mark.parametrize(
    "payload_name",
    ["all", "mip"],
)
def test_skill_payload_executes_hermetically_in_target(payload_name, run_skill_payload):
    result, repo, trace = run_skill_payload(payload_name)

    assert result.returncode == 0
    assert (trace / "cwd").read_text(encoding="utf-8").strip() == str(repo)
    argv = (trace / "argv").read_text(encoding="utf-8").splitlines()
    assert argv[:4] == ["--nproc_per_node", "2", "examples/puzzletron/main.py", "--config"]
    assert argv[4] == "configs/test.yaml"
    assert ("--mip-only" in argv) is (payload_name == "mip")
    assert "Puzzletron Progress 1/8: fake progress" in result.stdout
    assert "unfiltered diagnostic" not in result.stdout
    assert "unfiltered diagnostic" in (repo / "log.txt").read_text(encoding="utf-8")
    assert result.stderr == ""


def test_skill_payload_derives_repo_root_from_git_checkout(run_skill_payload):
    result, repo, trace = run_skill_payload("all", derive_repo_root=True)

    assert result.returncode == 0
    assert (trace / "cwd").read_text(encoding="utf-8").strip() == str(repo)


def test_skill_payload_missing_config_stops_before_torchrun(run_skill_payload):
    result, _, trace = run_skill_payload("all", config_path="missing.yaml")

    assert result.returncode == 2
    assert "Puzzletron config not found: missing.yaml" in result.stderr
    assert not (trace / "argv").exists()


def test_skill_payload_pipefail_preserves_torchrun_failure(run_skill_payload):
    result, _, _ = run_skill_payload("all", torchrun_exit=7)

    assert result.returncode == 7


# Helpers


def puzzletron_run_payloads() -> dict[str, str]:
    skill_text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    return dict(
        re.findall(
            r"^### (all|mip) \\<nproc_per_node\\>\n.*?```bash\n(.*?)```",
            skill_text,
            re.MULTILINE | re.DOTALL,
        )
    )


def _fake_torchrun(bin_dir: Path) -> None:
    script = bin_dir / "torchrun"
    script.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$PWD\" > \"$TRACE_DIR/cwd\"\n"
        "printf '%s\\n' \"$@\" > \"$TRACE_DIR/argv\"\n"
        "printf '%s\\n' 'Puzzletron Progress 1/8: fake progress'\n"
        "printf '%s\\n' 'unfiltered diagnostic'\n"
        "exit \"${FAKE_TORCHRUN_EXIT:-0}\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)


if __name__ == "__main__":
    sys.exit(__import__("pytest").main([__file__, "-q"]))
