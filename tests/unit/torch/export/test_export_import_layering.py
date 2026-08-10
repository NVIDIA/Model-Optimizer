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

"""Pins the import layering of the unified HF export package.

The package is a DAG: ``hf_export_prep`` and ``hf_weight_export`` are leaves, the
exporters sit above them, and ``unified_export_hf`` dispatches to the exporters. A
cycle reintroduced anywhere in that graph would stay invisible under ``import
modelopt.torch.export``, because ``export/__init__.py`` always pulls
``unified_export_hf`` in first and that "good" order happens to resolve. Importing
each module first, in its own interpreter, is what actually exercises the invariant.
"""

import subprocess
import sys

import pytest

EXPORT_MODULES = [
    "hf_export_handlers",
    "hf_export_prep",
    "hf_weight_export",
    "model_utils",
    "moe_utils",
    "registry",
    "unified_export_diffusers",
    "unified_export_hf",
    "unified_export_hf_streaming",
]


@pytest.mark.parametrize("module", EXPORT_MODULES)
def test_module_imports_standalone(module):
    """Each export module imports cleanly when it is the first one loaded."""
    result = subprocess.run(
        [sys.executable, "-c", f"import modelopt.torch.export.{module}"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"importing modelopt.torch.export.{module} first failed — likely a new import "
        f"cycle:\n{result.stderr}"
    )
