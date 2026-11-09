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
import re
import shlex
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parents[1] / "recipes" / "examples" / "gym" / "example_mrcr.yaml"
GATE = re.compile(
    r"\{% if config\.params\.limit_samples is not none %\}(.*?)\{% endif %\}", re.DOTALL
)


def _block(key, nth=0):
    """Return the lines nested under the ``nth`` ``key:`` in the template (stdlib-only)."""
    lines = TEMPLATE.read_text().splitlines()
    start = [i for i, line in enumerate(lines) if line.strip().startswith(f"{key}:")][nth]
    indent = len(lines[start]) - len(lines[start].lstrip())
    block = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        block.append(line)
    return lines[start], "\n".join(block)


def _render_gate(text, limit_samples):
    # Mirrors the Jinja semantics of the single gate the template uses.
    return GATE.sub(
        lambda m: (
            ""
            if limit_samples is None
            else m.group(1).replace("{{config.params.limit_samples}}", str(limit_samples))
        ),
        text,
    )


def test_limit_samples_defaults_to_full_run():
    header, _ = _block("limit_samples")
    assert header.split("#")[0].strip() == "limit_samples:"  # null => full run


@pytest.mark.parametrize(("limit_samples", "expected"), [(None, None), (5, "++limit=5")])
def test_limit_samples_gates_gym_limit(limit_samples, expected):
    _, rollout = _block("collect_rollout_params")
    assert len(GATE.findall(rollout)) == 1
    rendered = _render_gate(rollout, limit_samples)
    limits = re.findall(r"\+\+limit=\S+", rendered)
    assert limits == ([expected] if expected else [])


@pytest.mark.parametrize(("limit_samples", "expected"), [(None, []), (5, ["++limit=5"])])
def test_limit_samples_jinja_render(limit_samples, expected):
    jinja2 = pytest.importorskip("jinja2")
    _, rollout = _block("collect_rollout_params")
    rendered = jinja2.Template(rollout).render(
        config={
            "params": {
                "limit_samples": limit_samples,
                "temperature": 1.0,
                "top_p": 0.95,
                "parallelism": 512,
            },
            "output_dir": "/out",
        },
        target={"api_endpoint": {"url": "http://x"}},
    )
    assert re.findall(r"\+\+limit=\S+", rendered) == expected


def test_deployment_command_keeps_all_flags_under_shell_parsing():
    header, block = _block("command")
    assert header.strip() == "command: >-"  # the deployment serve command, a folded scalar
    folded = " ".join(line.strip() for line in block.splitlines() if line.strip())
    tokens = shlex.split(folded, comments=True)
    assert tokens == shlex.split(folded)  # no `#` turning the tail into a shell comment
    for flag in (
        "--max-model-len",
        "--gpu-memory-utilization",  # replacing `command:` drops the launcher's default flag
        "--kv-cache-dtype",
        "--reasoning-parser",
        "--enable-prefix-caching",
        "--enable-chunked-prefill",
        "--max-num-batched-tokens",
    ):
        assert flag in tokens


def _folded(key):
    _, block = _block(key)
    return " ".join(line.strip() for line in block.splitlines() if line.strip())


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs util-linux setsid")
def test_gym_rollout_keeps_token_off_argv_and_scratch_private(tmp_path):
    header, block = _block("command", nth=1)
    assert header.strip() == "command: |"  # the gym bootstrap, not the deployment command
    script = textwrap.dedent(block)
    for key in ("data_prep_params", "collect_rollout_params", "common_params"):
        script = script.replace(f"{{{{config.params.extra.nemo_gym.{key}}}}}", _folded(key))
    lines = re.sub(r"\{\{[^}]*\}\}", "x", _render_gate(script, None)).splitlines()
    # Keep `set -ex`, then run from data prep on; the bootstrap above it needs a real /opt/Gym.
    prep = next(i for i, line in enumerate(lines) if line.startswith("ng_prepare_benchmark"))
    script = "\n".join([lines[0], *lines[prep:]])

    bin_dir, out, scratch = tmp_path / "bin", tmp_path / "out", tmp_path / "scratch"
    for d in (bin_dir, out, scratch):
        d.mkdir()
    stubs = {
        "ng_prepare_benchmark": 'printf "%s\\n" "$@" > "$OUT/prepare"',
        "ng_e2e_collect_rollouts": (
            'printf "%s\\n" "$@" > "$OUT/rollouts"; stat -c %a "$TMPDIR"/* > "$OUT/scratch"; exit 3'
        ),
        "ray": "",  # never stop or kill real Ray daemons on the host
        "pkill": "",
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(f"#!/usr/bin/env bash\n{body}\n")
        (bin_dir / name).chmod(0o755)
    secret = "hf_test_secret_token"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "TMPDIR": str(scratch),
        "OUT": str(out),
        "HF_TOKEN": secret,
        "DUMMY_API_KEY": "dummy",
        "UV_CACHE_DIR": "/uv",
    }
    result = subprocess.run(
        ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 3  # the rollout's failure propagates
    assert secret not in (out / "prepare").read_text() + (out / "rollouts").read_text()  # argv
    assert secret not in result.stdout + result.stderr  # `set -x` trace
    assert (out / "scratch").read_text().split() == ["700"]  # one private mktemp dir
    assert not any(scratch.iterdir())  # removed even though the rollout failed
