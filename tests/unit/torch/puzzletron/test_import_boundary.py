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

"""Regression tests for Puzzletron's optional-dependency import boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
PUZZLETRON_PACKAGE = "modelopt.torch.puzzletron"
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Puzzletron imports fcntl-backed runtime modules that are unavailable on Windows",
)


def test_lightweight_puzzletron_import_does_not_require_automodel() -> None:
    result = _run_fresh(
        f"from {PUZZLETRON_PACKAGE}.identity import stable_hash; "
        "assert stable_hash({'ready': True}, prefix='ci')",
        unavailable_module="nemo_automodel",
    )

    assert result.returncode == 0, result.stderr


def test_resolved_setup_config_is_available_only_from_its_explicit_module() -> None:
    result = _run_fresh(
        "import puzzletron_setup.v2 as setup_v2; "
        "from puzzletron_setup.v2.resolved import ResolvedCampaignConfig; "
        "assert not hasattr(setup_v2, 'ResolvedCampaignConfig')",
        unavailable_module="nemo_automodel",
    )

    assert result.returncode == 0, result.stderr


def test_resolved_setup_config_import_does_not_initialize_torch() -> None:
    result = _run_fresh(
        "from puzzletron_setup.v2.resolved import ResolvedCampaignConfig; "
        "assert 'torch' not in sys.modules; "
        "assert not any(name.startswith('modelopt.torch') for name in sys.modules)",
        unavailable_module="nemo_automodel",
    )

    assert result.returncode == 0, result.stderr


def test_prepare_dataset_uses_fire_only_for_cli_execution() -> None:
    result = _run_fresh(
        "import importlib.util, pathlib, types; loader = importlib.util.spec_from_file_location; "
        f"stub_names = ('datasets', 'numpy', '{PUZZLETRON_PACKAGE}.tools.logger'); "
        "sys.modules.update({name: types.ModuleType(name) for name in stub_names}); "
        "sys.modules[stub_names[-1]].mprint = lambda *args, **kwargs: None; "
        f"module_name = '{PUZZLETRON_PACKAGE}.dataset.prepare_dataset'; "
        "path = pathlib.Path(*module_name.split('.')).with_suffix('.py'); spec = loader(module_name, path); "
        "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
        "fire_module = types.ModuleType('fire'); fire_module.Fire = lambda component: print(component.__name__); "
        "sys.modules['fire'] = fire_module; spec = loader('__main__', path); "
        "module = importlib.util.module_from_spec(spec); module.__package__ = module_name.rpartition('.')[0]; "
        "module.__spec__ = None; spec.loader.exec_module(module)",
        unavailable_module="fire",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "process_and_save_dataset"


def test_automodel_recipe_loader_reports_missing_dependency() -> None:
    result = _run_fresh(
        f"from {PUZZLETRON_PACKAGE}.diagnostics.width_slice_equivalence "
        "import _replace_block_scoring_recipe; "
        "\ntry:\n _replace_block_scoring_recipe()"
        "\nexcept ImportError as error:"
        "\n assert 'requires a compatible NeMo AutoModel' in str(error)"
        "\nelse:\n raise AssertionError('AutoModel-backed recipe unexpectedly loaded')",
        unavailable_module="nemo_automodel",
    )

    assert result.returncode == 0, result.stderr


def _run_fresh(
    script: str, *, unavailable_module: str | None = None
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("COVERAGE_PROCESS_START", None)
    unavailable = f"sys.modules[{unavailable_module!r}] = None; " if unavailable_module else ""
    prelude = "import sys; " + unavailable
    return subprocess.run(
        [sys.executable, "-c", prelude + script],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
