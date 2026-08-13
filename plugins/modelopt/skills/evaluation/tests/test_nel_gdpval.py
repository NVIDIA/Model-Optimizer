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

SCRIPT = Path(__file__).parents[1] / "scripts" / "nel-gdpval.sh"


def test_launcher_uses_validated_pin_despite_environment_override(tmp_path):
    args_file = tmp_path / "uvx-args"
    uvx = tmp_path / "uvx"
    uvx.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$UVX_ARGS_FILE"\n')
    uvx.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "NEL_GDPVAL_SPEC": "nemo-evaluator-launcher[all]==0.0.0",
            "NEL_GDPVAL_VERSION": "0.0.0",
            "PATH": f"{tmp_path}:{env['PATH']}",
            "UVX_ARGS_FILE": str(args_file),
        }
    )

    subprocess.run([SCRIPT, "run", "--config", "gdpval.yaml"], env=env, check=True)

    assert args_file.read_text().splitlines() == [
        "--python",
        "3.10",
        "--from",
        "nemo-evaluator-launcher[all]==0.2.6",
        "nel",
        "run",
        "--config",
        "gdpval.yaml",
    ]
