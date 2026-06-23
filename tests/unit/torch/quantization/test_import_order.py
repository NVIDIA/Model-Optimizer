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

"""Tests for quantization import order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_quant_linear_import_does_not_cycle_through_backends():
    """RealQuantLinear must be importable before backend registration is needed."""
    repo_root = Path(__file__).resolve().parents[4]
    code = "\n".join(
        [
            "from modelopt.torch.quantization.nn.modules.quant_linear import RealQuantLinear",
            "assert RealQuantLinear.__name__ == 'RealQuantLinear'",
        ]
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
