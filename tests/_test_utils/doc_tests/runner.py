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

"""Run trusted documentation in an isolated process group with a whole-scenario deadline."""

import contextlib
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from .parser import DocTestError, parse_markdown

__all__ = ["run_scenario"]


def run_scenario(scenario, repo: Path, tmp: Path) -> str:
    """Execute one scenario and return captured output; terminate its group on every exit."""
    tmp = Path(tmp).resolve()
    tmp.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(Path(repo).resolve()),
            str(Path(__file__).resolve().parents[2]),
            environment.get("PYTHONPATH", ""),
        ]
    )
    with (
        tempfile.TemporaryDirectory(prefix="doc-test-", dir=tmp) as control_dir,
        tempfile.TemporaryFile() as output,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "_test_utils.doc_tests.runner",
                str(scenario.path),
                scenario.id,
                str(Path(repo).resolve()),
                str(tmp),
                control_dir,
            ],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=scenario.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        finally:
            # Give launchers time to forward termination to their own worker groups.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    process.poll()
                    os.killpg(process.pid, 0)
                    time.sleep(0.05)
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        output.seek(0)
        text = output.read().decode(errors="replace")
        for index, step in enumerate(scenario.steps):
            if step.kind == "run":
                text = text.replace(
                    str(Path(control_dir) / f"fence-{index}.sh"), str(scenario.path)
                )
    if timed_out or process.returncode:
        reason = (
            f"timed out after {scenario.timeout_seconds}s"
            if timed_out
            else f"exit {process.returncode}"
        )
        raise DocTestError(f"{scenario.path}:{scenario.line}: {scenario.id}: {reason}\n{text}")
    return text


def _execute(scenario, repo, tmp, control):
    ctx = SimpleNamespace(repo=repo, tmp=tmp, cwd=repo, env=dict(os.environ))
    namespace = {"ctx": ctx}

    def python_step(step):
        print(f"{scenario.path}:{step.line}: {step.kind}", flush=True)
        # Only explicitly opted-in, repository-owned Python is executed, just as in pytest.
        exec(compile("\n" * (step.line - 1) + step.code, str(scenario.path), "exec"), namespace)

    for step in scenario.steps:
        if step.kind == "setup":
            python_step(step)
    script = ["set -Eeuo pipefail", f"DOC_TEST_SOURCE={shlex.quote(str(scenario.path))}"]
    for index, step in enumerate(scenario.steps):
        if step.kind != "run":
            continue
        source = control / f"fence-{index}.sh"
        source.write_text("\n" * (step.line - 1) + step.code)
        script.append(f"printf '%s\\n' {shlex.quote(f'{scenario.path}:{step.line}: run')}")
        script.append("set -Eeuo pipefail")
        script.append(
            'trap \'printf "%s:%s: shell command failed\\n" "$DOC_TEST_SOURCE" "$LINENO" >&2\' ERR'
        )
        script.append(f"source {shlex.quote(str(source))}")
    script.extend(
        [
            f"pwd -P > {shlex.quote(str(control / 'cwd'))}",
            f"env -0 > {shlex.quote(str(control / 'env'))}",
        ]
    )
    shell = control / "scenario.sh"
    shell.write_text("\n".join(script) + "\n")
    # Inherit stdout so the parent retains partial output even if the scenario times out.
    result = subprocess.run(["bash", "--noprofile", "--norc", str(shell)], cwd=ctx.cwd, env=ctx.env)
    if result.returncode:
        raise DocTestError(f"shell exited with {result.returncode}")
    ctx.cwd = Path((control / "cwd").read_text().rstrip("\n"))
    ctx.env = dict(item.split("=", 1) for item in (control / "env").read_text().split("\0") if item)
    for step in scenario.steps:
        if step.kind == "verify":
            python_step(step)


if __name__ == "__main__":
    markdown, scenario_id, root, workspace, control_dir = sys.argv[1:]
    selected = next(s for s in parse_markdown(Path(markdown)) if s.id == scenario_id)
    _execute(selected, Path(root), Path(workspace), Path(control_dir))
