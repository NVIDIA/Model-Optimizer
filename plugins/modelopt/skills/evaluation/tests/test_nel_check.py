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

import os
import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
CHECK = SCRIPTS / "nel-check.sh"
GDPVAL = SCRIPTS / "nel-gdpval.sh"
VERSION_FILE = SCRIPTS / "nel-validated-version.sh"

VALIDATED = "0.2.6"


def _run(args, env=None, **kwargs):
    return subprocess.run([str(CHECK), *args], capture_output=True, text=True, env=env, **kwargs)


def _env_with_stub_nel(tmp_path, version_table):
    """Put a stub `nel` printing `version_table` on stdout (logs on stderr, as the real CLI does)."""
    nel = tmp_path / "nel"
    nel.write_text(
        "#!/usr/bin/env bash\n"
        'echo "[I] Centralized logging configured" >&2\n'
        f"cat <<'EOF'\n{version_table}\nEOF\n"
    )
    nel.chmod(0o755)
    env = os.environ.copy()
    # Prepend so the stub wins over any launcher already installed in the base env.
    env["PATH"] = f"{tmp_path}:{env['PATH']}"
    env.pop("NEL_ALLOW_UNVALIDATED", None)
    return env


def test_reports_validated_version():
    assert _run(["--version"]).stdout.strip() == VALIDATED


def test_reports_pip_spec():
    assert _run(["--spec"]).stdout.strip() == f"nemo-evaluator-launcher[all]=={VALIDATED}"


def test_accepts_validated_launcher(tmp_path):
    env = _env_with_stub_nel(tmp_path, f"nemo_evaluator_launcher: {VALIDATED}")
    result = _run([], env=env)
    assert result.returncode == 0
    assert f"nemo_evaluator_launcher: {VALIDATED} (validated)" in result.stdout


def test_rejects_stale_launcher(tmp_path):
    """The failure that motivated the pin: a base env already carrying an older launcher."""
    env = _env_with_stub_nel(tmp_path, "nemo_evaluator_launcher: 0.2.4")
    result = _run([], env=env)
    assert result.returncode == 1
    assert "0.2.4" in result.stderr
    assert f"pip install 'nemo-evaluator-launcher[all]=={VALIDATED}'" in result.stderr


def test_ignores_internal_package_version(tmp_path):
    """`nemo-evaluator-launcher-internal` prints its own row; only the launcher row gates."""
    env = _env_with_stub_nel(
        tmp_path,
        f"nemo_evaluator_launcher: {VALIDATED}\nnemo_evaluator_launcher_internal: 0.3.174+20260609",
    )
    result = _run([], env=env)
    assert result.returncode == 0, result.stderr


def test_rejects_stale_launcher_behind_internal_package(tmp_path):
    """An internal package at 0.3.x must not mask a stale 0.2.4 launcher."""
    env = _env_with_stub_nel(
        tmp_path,
        "nemo_evaluator_launcher: 0.2.4\nnemo_evaluator_launcher_internal: 0.3.174+20260609",
    )
    assert _run([], env=env).returncode == 1


def test_missing_launcher_is_actionable(tmp_path):
    env = os.environ.copy()
    env["PATH"] = "/usr/bin:/bin"
    result = _run([], env=env)
    assert result.returncode == 1
    assert "not found on PATH" in result.stderr
    assert f"pip install 'nemo-evaluator-launcher[all]=={VALIDATED}'" in result.stderr


def test_unparseable_version_output_fails_closed(tmp_path):
    env = _env_with_stub_nel(tmp_path, "some unexpected output")
    result = _run([], env=env)
    assert result.returncode == 1
    assert "could not read" in result.stderr


def test_escape_hatch_warns_and_marks_result_unvalidated(tmp_path):
    env = _env_with_stub_nel(tmp_path, "nemo_evaluator_launcher: 0.2.4")
    env["NEL_ALLOW_UNVALIDATED"] = "1"
    result = _run([], env=env)
    assert result.returncode == 0
    assert "(UNVALIDATED)" in result.stdout
    assert "dev/canary only" in result.stderr


@pytest.mark.parametrize("script", [CHECK, GDPVAL])
def test_scripts_share_one_validated_version(script):
    """Both entry points must resolve the same pin, so a bump cannot drift."""
    resolved = subprocess.run(
        ["bash", "-c", f'source "{VERSION_FILE}"; echo "$NEL_VALIDATED_VERSION"'],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert resolved == VALIDATED
    assert VALIDATED not in script.read_text(), (
        f"{script.name} hard-codes the version; source nel-validated-version.sh instead"
    )
