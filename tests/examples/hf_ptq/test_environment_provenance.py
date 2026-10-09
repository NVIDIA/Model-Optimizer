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

"""Environment probes reject stale source and inherited packages before imports."""

import importlib.metadata
import importlib.util
import pathlib
import struct
import subprocess
import sys

import pytest

SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3]
    / "plugins/modelopt/skills/ptq/scripts/verify_environment.py"
)


@pytest.mark.parametrize("wrong_ref", [True, False])
def test_reject_unsafe_provenance(tmp_path, wrong_ref):
    subprocess.run(["git", "init", str(tmp_path)], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path, text=True).strip()
    # Use the base interpreter so this process cannot inherit pytest's isolated environment.
    result = subprocess.run(
        [
            sys._base_executable,
            str(SCRIPT),
            "--source",
            str(tmp_path),
            "--ref",
            "0" * 40 if wrong_ref else ref,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert ("differs from" if wrong_ref else "without system-site-packages") in result.stderr


@pytest.mark.parametrize(
    ("tag", "machine", "accepted"),
    [
        ("sbsa", 183, True),
        ("aarch64", 183, False),
        ("sbsa", 62, False),
    ],
)
def test_vendor_exception_requires_exact_arm64_defect(tmp_path, tag, machine, accepted):
    spec = importlib.util.spec_from_file_location("environment_probe", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = tmp_path / "nvidia_cusparselt_cu13-0.8.1.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: nvidia-cusparselt-cu13\nVersion: 0.8.1\n")
    (metadata / "WHEEL").write_text(f"Tag: py3-none-manylinux2014_{tag}\n")
    library = tmp_path / "nvidia/cusparselt/lib/libcusparseLt.so.0"
    library.parent.mkdir(parents=True)
    header = bytearray(64)
    header[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", header, 18, machine)
    library.write_bytes(header)
    distribution = importlib.metadata.Distribution.at(metadata)
    if accepted:
        assert module.validate_cusparselt_sbsa(distribution, "aarch64")["elf_machine"] == 183
    else:
        with pytest.raises(RuntimeError, match="known ARM64 SBSA tag defect"):
            module.validate_cusparselt_sbsa(distribution, "aarch64")
