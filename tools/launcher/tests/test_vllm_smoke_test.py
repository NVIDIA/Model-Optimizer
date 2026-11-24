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

"""vllm_smoke_test.sh paths that stop before vLLM starts, and the SMOKE_JSON they leave."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from core import GlobalVariables, SandboxPipeline, SandboxTask2

_COMMON = Path(__file__).resolve().parents[1] / "common"
_STREAMING_DFLASH2 = (
    Path(__file__).resolve().parents[1] / "examples/Qwen/Qwen3-8B/hf_streaming_dflash2.yaml"
)
# service_utils.sh appends to modelopt/__init__.py only on local rank 0, so a rank set by an
# enclosing Slurm/MPI job would hide the failure under test.
_RANK_VARS = ("PMIX_LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", "SLURM_PROCID")


def _run(tmp_path, **env_overrides):
    """Run the script from a directory with no modules/, as rank 0. Returns (proc, JSON)."""
    (tmp_path / "common" / "specdec").mkdir(parents=True)
    for name in ("service_utils.sh", "specdec/vllm_smoke_test.sh"):
        shutil.copy(_COMMON / name, tmp_path / "common" / name)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool, body in (("pip", "exit 0"), ("nvidia-smi", "echo 1")):
        (bin_dir / tool).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / tool).chmod(0o755)
    (tmp_path / "train_out").mkdir()  # training left no exported-checkpoint-*

    env = {k: v for k, v in os.environ.items() if k not in _RANK_VARS}
    env.update(
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        HF_MODEL_CKPT="/hf-local/target",
        DRAFT_CKPT_DIR=str(tmp_path / "train_out"),
        SMOKE_JSON=str(tmp_path / "out" / "smoke.json"),
    )
    env.update(env_overrides)
    smoke_json = Path(env["SMOKE_JSON"])
    proc = subprocess.run(
        ["bash", "common/specdec/vllm_smoke_test.sh"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert smoke_json.exists(), proc.stdout + proc.stderr
    # Strict JSON: NaN/Infinity would break non-Python readers of the result.
    return proc, json.loads(smoke_json.read_text(), parse_constant=pytest.fail)


def test_sources_service_utils_outside_a_launcher_workdir(tmp_path):
    """The rank-0 append failing must not stop the script at `source`."""
    proc, result = _run(tmp_path)
    assert proc.returncode == 1
    assert result["stage"] == "draft_missing", result["reason"]


@pytest.mark.parametrize(
    ("env", "reason"),
    [
        ({"MIN_ACCEPTANCE_LENGTH": "1,5"}, "MIN_ACCEPTANCE_LENGTH='1,5' is not a finite number"),
        ({"MIN_ACCEPTANCE_LENGTH": "nan"}, "MIN_ACCEPTANCE_LENGTH='nan' is not a finite number"),
        ({"NUM_SPEC_TOKENS": "x"}, "NUM_SPEC_TOKENS='x' is not a positive integer"),
        # Not valid as the bare JSON number SPEC_CONFIG passes to vLLM.
        ({"NUM_SPEC_TOKENS": "08"}, "NUM_SPEC_TOKENS='08' is not a positive integer"),
        ({"SERVER_TIMEOUT": "15m"}, "SERVER_TIMEOUT='15m' is not a positive integer"),
    ],
)
def test_invalid_config_fails_with_a_result(tmp_path, env, reason):
    """A bad number fails before any work, and the trap still writes a readable result."""
    proc, result = _run(tmp_path, **env)
    assert proc.returncode == 1
    assert result["stage"] == "invalid_config"
    assert result["reason"] == reason
    assert result["pass"] is False


def test_smoke_only_rerun_keeps_the_original_verdict(tmp_path):
    """The example's documented rerun reads an earlier export but writes its own verdict.

    The script deletes SMOKE_JSON before it starts, so a failed rerun writing into the
    earlier experiment would erase that experiment's PASS.
    """
    example = yaml.safe_load(_STREAMING_DFLASH2.read_text())["pipeline"]
    task_2 = example["task_2"]
    pipeline = SandboxPipeline(
        task_2=SandboxTask2(script=task_2["script"], environment=task_2["environment"]),
        global_vars=GlobalVariables(
            **{**example["global_vars"], "output_dir": "/cicd/earlier/dflash2"}
        ),
    )
    # build_slurm_executor() mounts <job_dir>/cicd/<experiment> as /scratchspace and
    # <job_dir>/cicd as /cicd; "rerun" is this experiment, "earlier" the one it reads.
    cicd = tmp_path / "job" / "cicd"
    mounts = {"/scratchspace": cicd / "rerun", "/cicd": cicd}
    env = {
        key: re.sub(r"^/(scratchspace|cicd)\b", lambda m: str(mounts[m[0]]), value)
        for item in pipeline.tasks[0].environment
        for key, value in item.items()
    }
    earlier_verdict = cicd / "earlier" / "smoke_check.json"  # its own /scratchspace
    earlier_verdict.parent.mkdir(parents=True)
    earlier_verdict.write_text('{"pass": true}')

    proc, _ = _run(tmp_path, **{**env, "NUM_SPEC_TOKENS": "x"})  # fails before vLLM
    assert proc.returncode == 1
    assert earlier_verdict.read_text() == '{"pass": true}'
    result = json.loads((cicd / "rerun" / "smoke_check.json").read_text())
    assert result["stage"] == "invalid_config"
    assert result["pass"] is False
    assert result["draft"] == str(cicd / "earlier" / "dflash2" / "export")
