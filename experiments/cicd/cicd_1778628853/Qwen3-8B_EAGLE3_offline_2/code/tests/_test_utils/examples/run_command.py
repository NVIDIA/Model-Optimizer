# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Utility functions for running example commands reused in multiple example tests."""

import os
import subprocess
from pathlib import Path

from _test_utils.torch.distributed.utils import get_free_port

MODELOPT_ROOT = Path(__file__).parents[3]


def extend_cmd_parts(cmd_parts: list[str], **kwargs):
    for key, value in kwargs.items():
        if value is not None:
            cmd_parts.extend([f"--{key}", str(value)])
    if kwargs.get("trust_remote_code", False):
        cmd_parts.append("--trust_remote_code")
    return cmd_parts


def run_example_command(
    cmd_parts: list[str],
    example_path: str,
    setup_free_port: bool = False,
    env: dict[str, str] | None = None,
):
    print(f"[{example_path}] Running command: {cmd_parts}")
    env = env or os.environ.copy()

    if setup_free_port:
        free_port = get_free_port()
        env["MASTER_PORT"] = str(free_port)

    subprocess.run(cmd_parts, cwd=MODELOPT_ROOT / "examples" / example_path, env=env, check=True)


def run_command_in_background(
    cmd_parts: list[str], example_path: str, stdout=None, stderr=None, text=True
):
    print(f"Running command in background: {' '.join(str(part) for part in cmd_parts)}")
    process = subprocess.Popen(
        cmd_parts,
        cwd=MODELOPT_ROOT / "examples" / example_path,
        stdout=stdout,
        stderr=stderr,
        text=text,
    )
    return process


def run_llm_ptq_command(*, model: str, quant: str, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
    kwargs.setdefault("calib", 16)

    cmd_parts = extend_cmd_parts(["scripts/huggingface_example.sh", "--no-verbose"], **kwargs)
    run_example_command(cmd_parts, "llm_ptq")


def run_vlm_ptq_command(*, model: str, quant: str, **kwargs):
    kwargs.update({"model": model, "quant": quant})
    kwargs.setdefault("tasks", "quant")
    kwargs.setdefault("calib", 16)

    cmd_parts = extend_cmd_parts(["scripts/huggingface_example.sh"], **kwargs)
    run_example_command(cmd_parts, "vlm_ptq")
