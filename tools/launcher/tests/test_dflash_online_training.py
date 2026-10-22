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

"""dflash_online_training.sh must not write a regression verdict for a failed training run."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

_COMMON = Path(__file__).resolve().parents[1] / "common"
_EXPORT = "modules/Model-Optimizer/examples/speculative_decoding/scripts/export_hf_checkpoint.py"


def _run(tmp_path, train_exit):
    """Run the script in a sandbox: stub pip/nvidia-smi/accelerate/export, real gate."""
    (tmp_path / "common" / "specdec").mkdir(parents=True)
    for name in ("service_utils.sh", "check_regression.py", "specdec/dflash_online_training.sh"):
        shutil.copy(_COMMON / name, tmp_path / "common" / name)
    (tmp_path / "modules/Model-Optimizer/modelopt").mkdir(
        parents=True
    )  # service_utils appends here
    export = tmp_path / _EXPORT
    export.parent.mkdir(parents=True)
    export.write_text(
        "import os, sys\n"
        "d = sys.argv[sys.argv.index('--export_path') + 1]\n"
        "os.makedirs(d, exist_ok=True)\n"
        "open(os.path.join(d, 'config.json'), 'w').write('{}')\n"
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool, body in (
        ("pip", "exit 0"),
        ("nvidia-smi", "echo 1"),
        ("accelerate", f"exit {train_exit}"),
    ):
        (bin_dir / tool).write_text(f"#!/bin/bash\n{body}\n")
        (bin_dir / tool).chmod(0o755)

    # A checkpoint that clears the threshold, and a passing verdict from an earlier run.
    out = tmp_path / "out"
    (out / "checkpoint-2000").mkdir(parents=True)
    (out / "checkpoint-2000" / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 4.0}]})
    )
    (out / "regression_check.json").write_text('{"pass": true}')

    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}", "MAX_FINAL_LOSS": "5.0"}
    env.pop("SLURM_PROCID", None)
    proc = subprocess.run(
        ["bash", "common/specdec/dflash_online_training.sh", f"training.output_dir={out}"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    verdict = out / "regression_check.json"
    return proc, json.loads(verdict.read_text()) if verdict.exists() else None


@pytest.mark.parametrize("train_exit", [0, 1])
def test_failed_training_writes_no_verdict(tmp_path, train_exit):
    """A crash must not let a surviving checkpoint, or an old verdict, pass the gate."""
    proc, verdict = _run(tmp_path, train_exit)
    if train_exit:
        assert verdict is None, verdict
        assert proc.returncode == 1, proc.stderr
        assert "training failed; skipping export and regression check" in proc.stderr
    else:  # control: the same sandbox grades the checkpoint when training succeeds
        assert verdict is not None, proc.stderr
        assert verdict["pass"] is True
        assert verdict["trainer_state"].endswith("checkpoint-2000/trainer_state.json")
