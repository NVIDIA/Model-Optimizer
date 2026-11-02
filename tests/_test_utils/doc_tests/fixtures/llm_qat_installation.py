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

"""QAT README installation checks in a disposable environment."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

__all__ = ["prepare_llm_qat_installation", "verify_llm_qat_installation"]


def prepare_llm_qat_installation(ctx):
    """Prepare the QAT requirements in a disposable venv without altering pytest."""
    venv = ctx.tmp / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    ctx.cwd = ctx.tmp
    target = ctx.tmp / "examples/llm_qat"
    target.mkdir(parents=True)
    shutil.copy2(ctx.repo / "examples/llm_qat/requirements.txt", target / "requirements.txt")
    ctx.env["PATH"] = str(venv / "bin") + os.pathsep + ctx.env["PATH"]
    ctx.env.pop("PYTHONPATH", None)
    ctx.env.pop("VIRTUAL_ENV", None)


def verify_llm_qat_installation(ctx):
    """Import the QAT dependencies and require ModelOpt to come from the new venv."""
    result = subprocess.run(
        [
            str(ctx.tmp / "venv/bin/python"),
            "-c",
            "import json, modelopt, accelerate, peft; print(json.dumps(modelopt.__file__))",
        ],
        cwd=ctx.cwd,
        env=ctx.env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(json.loads(result.stdout.strip().splitlines()[-1])).is_relative_to(ctx.tmp / "venv")
