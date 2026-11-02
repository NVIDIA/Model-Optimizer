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

"""Exercise example CI routing without allocating GPU runners."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
ALL_LANES = {"torch", "vllm", "trtllm", "megatron", "diffusers", "onnx"}
QAT = "examples/llm_qat/README.md"
BRIDGE = "examples/megatron_bridge/README.md"


@pytest.mark.skipif(os.name != "posix", reason="Workflow uses Bash")
@pytest.mark.parametrize(
    ("paths", "common", "stacked", "non_pr", "expected"),
    [
        ([QAT], False, False, False, {"torch"}),
        ([BRIDGE], False, False, False, {"megatron"}),
        ([QAT, BRIDGE], False, False, False, {"torch", "megatron"}),
        (["examples/hf_ptq/README.md"], False, False, False, set()),
        ([], True, False, False, ALL_LANES),
        ([QAT, BRIDGE], False, True, False, set()),
        ([], False, False, True, ALL_LANES),
    ],
)
def test_example_readme_lane_routing(tmp_path, *, paths, common, stacked, non_pr, expected):
    workflow = yaml.safe_load((REPO / ".github/workflows/example_tests.yml").read_text())
    steps = workflow["jobs"]["pr-gate"]["steps"]
    groups = yaml.safe_load(
        next(step["with"]["files_yaml"] for step in steps if step.get("id") == "changed")
    )
    flags = {
        f"{group}_any_modified": any(path in groups[group] for path in paths)
        for group in ("torch_docs", "megatron_docs")
    }
    flags["common_any_modified"] = common
    lanes = next(step for step in steps if step.get("id") == "lanes")
    env = dict(os.environ, GITHUB_OUTPUT=str(tmp_path / "outputs"))
    for key, expression in lanes["env"].items():
        if key in {"NON_PR", "STACKED"}:
            enabled = non_pr if key == "NON_PR" else stacked
        else:
            terms = expression.removeprefix("${{ ").removesuffix(" }}").split(" || ")
            enabled = False
            for term in terms:
                match = re.fullmatch(r"steps\.changed\.outputs\.(\w+)(?: == 'true')?", term)
                assert match, f"Unsupported lane expression: {expression}"
                enabled |= flags.get(match[1], False)
        env[key] = str(enabled).lower()
    # Run the workflow's actual gate; no model subprocesses or GPU jobs are started.
    subprocess.run(["bash", "-e", "-c", lanes["run"]], env=env, check=True, timeout=5)
    outputs = dict(line.split("=", 1) for line in (tmp_path / "outputs").read_text().splitlines())
    assert {lane for lane in ALL_LANES if outputs[lane] == "true"} == expected
    assert outputs["any_changed"] == str(bool(expected)).lower()
