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

import re
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).parents[1] / "recipes" / "examples" / "gym" / "example_mrcr.yaml"
GATE = re.compile(
    r"\{% if config\.params\.limit_samples is not none %\}(.*?)\{% endif %\}", re.DOTALL
)


def _block(key):
    """Return the lines nested under the first ``key:`` in the template (stdlib-only)."""
    lines = TEMPLATE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith(f"{key}:"))
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
        lambda m: ""
        if limit_samples is None
        else m.group(1).replace("{{config.params.limit_samples}}", str(limit_samples)),
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
