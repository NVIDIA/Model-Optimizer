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

"""Regression tests for importing Puzzletron without optional AutoModel code."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Puzzletron imports fcntl-backed runtime modules that are unavailable on Windows",
)


def _run_without_automodel(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['nemo_automodel'] = None; " + script,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_lightweight_puzzletron_import_does_not_require_automodel() -> None:
    result = _run_without_automodel(
        "from modelopt.torch.puzzletron.identity import stable_hash; "
        "assert stable_hash({'ready': True}, prefix='ci')"
    )

    assert result.returncode == 0, result.stderr


def test_resolved_setup_config_is_available_only_from_its_explicit_module() -> None:
    result = _run_without_automodel(
        "import puzzletron_setup.v2 as setup_v2; "
        "from puzzletron_setup.v2.resolved import ResolvedCampaignConfig; "
        "assert not hasattr(setup_v2, 'ResolvedCampaignConfig')"
    )

    assert result.returncode == 0, result.stderr


def test_resolved_setup_config_import_does_not_initialize_torch() -> None:
    result = _run_without_automodel(
        "from puzzletron_setup.v2.resolved import ResolvedCampaignConfig; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name.startswith('modelopt.torch') for name in sys.modules)"
    )

    assert result.returncode == 0, result.stderr


def test_automodel_recipe_loader_reports_missing_dependency() -> None:
    result = _run_without_automodel(
        "from modelopt.torch.puzzletron.diagnostics.width_slice_equivalence "
        "import _replace_block_scoring_recipe; "
        "\ntry:\n _replace_block_scoring_recipe()"
        "\nexcept ImportError as error:"
        "\n assert 'requires a compatible NeMo AutoModel' in str(error)"
        "\nelse:\n raise AssertionError('AutoModel-backed recipe unexpectedly loaded')"
    )

    assert result.returncode == 0, result.stderr
