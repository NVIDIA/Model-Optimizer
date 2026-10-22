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
import importlib
import json
import os
import shlex
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from .parser import DocTestError, Scenario, Snippet

__all__ = ["run_scenario"]


def run_scenario(scenario, repo: Path, tmp: Path) -> str:
    """Execute one scenario and return captured output; terminate its group on every exit."""
    if scenario.path is None or not scenario.steps:
        raise DocTestError("Resolve snippet references before running a scenario")
    repo = Path(repo).resolve()
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
        manifest = Path(control_dir) / "scenario.json"
        manifest.write_text(json.dumps(asdict(scenario) | {"path": str(scenario.path)}))
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                "-m",
                "_test_utils.doc_tests.runner",
                str(manifest),
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
        completed = (Path(control_dir) / "completed").is_file()
        output.seek(0)
        text = output.read().decode(errors="replace")
        for index, step in enumerate(scenario.steps):
            if step.kind in {"run", "background"}:
                text = text.replace(
                    str(Path(control_dir) / f"fence-{index}.sh"), str(scenario.path)
                )
    if timed_out:
        reason = f"timed out after {scenario.timeout_seconds}s"
    elif process.returncode:
        reason = f"exit {process.returncode}"
    elif not completed:
        reason = "worker exited before completing the scenario"
    else:
        return text
    raise DocTestError(f"{scenario.path}:{scenario.line}: {scenario.id}: {reason}\n{text}")


def _execute(scenario, repo, tmp, control):
    ctx = SimpleNamespace(repo=repo, tmp=tmp, cwd=repo, env=dict(os.environ))
    called = completed = False

    def run(namespace=None):
        nonlocal called, completed
        if called:
            raise DocTestError("A scenario handler must call ctx.run() exactly once")
        called = True
        namespace = {} if namespace is None else namespace
        namespace.update(ctx=ctx, __name__="__main__", __file__=str(scenario.path))
        if scenario.steps[0].kind == "python":
            os.chdir(ctx.cwd)
            os.environ.clear()
            os.environ.update(ctx.env)
            for step in scenario.steps:
                print(f"{scenario.path}:{step.line}: {step.id}", flush=True)
                exec(
                    compile("\n" * (step.line - 1) + step.code, str(scenario.path), "exec"),
                    namespace,
                )
            ctx.cwd, ctx.env = Path.cwd(), dict(os.environ)
        else:
            _execute_shell(scenario, ctx, control)
        completed = True
        return namespace

    ctx.run = run
    if scenario.handler:
        module, name = scenario.handler.split(":")
        handler = getattr(importlib.import_module(module), name)
        handler(ctx)
    else:
        ctx.run()
    if not completed:
        raise DocTestError("Scenario handler did not complete ctx.run()")


def _execute_shell(scenario, ctx, control):
    script = ["set -Eeuo pipefail", f"DOC_TEST_SOURCE={shlex.quote(str(scenario.path))}"]
    for index, step in enumerate(scenario.steps):
        if step.kind not in {"run", "background"}:
            continue
        source = control / f"fence-{index}.sh"
        source.write_text("\n" * (step.line - 1) + step.code)
        script.append(f"printf '%s\\n' {shlex.quote(f'{scenario.path}:{step.line}: {step.id}')}")
        script.append(f"DOC_TEST_SNIPPET={shlex.quote(step.id)}")
        script.append("set -Eeuo pipefail")
        script.append(
            'trap \'printf "%s:%s: %s: shell command failed\\n" '
            '"$DOC_TEST_SOURCE" "$LINENO" "$DOC_TEST_SNIPPET" >&2\' ERR'
        )
        command = f"source {shlex.quote(str(source))}"
        if step.kind == "background":
            command += f" &\necho $! > {shlex.quote(str(control / 'background_pid'))}"
        script.append(command)
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
    if (control / "background_pid").exists():
        ctx.background_pid = int((control / "background_pid").read_text())


if __name__ == "__main__":
    manifest, root, workspace, control_dir = sys.argv[1:]
    payload = json.loads(Path(manifest).read_text())
    payload["path"] = Path(payload["path"])
    payload["steps"] = [Snippet(**step) for step in payload["steps"]]
    _execute(Scenario(**payload), Path(root), Path(workspace), Path(control_dir))
    (Path(control_dir) / "completed").touch()
