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

"""Public-surface and optional-dependency isolation checks for core PDD."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import modelopt.torch.fastgen as fastgen
from modelopt.torch.fastgen.config import PDDConfig
from modelopt.torch.fastgen.flow_matching import (
    fusion_coefficients,
    integrate_interval_velocities,
    make_shifted_flow_grid,
)
from modelopt.torch.fastgen.loader import load_pdd_config
from modelopt.torch.fastgen.methods.pdd import (
    PDDLayerSpec,
    PDDModelAdapter,
    PDDOutputProjection,
    PDDPipeline,
    convert_to_pdd_output_projection,
    get_module_by_path,
    replace_module_by_path,
)

_CORE_SOURCES = (
    "modelopt/torch/fastgen/config.py",
    "modelopt/torch/fastgen/flow_matching.py",
    "modelopt/torch/fastgen/loader.py",
    "modelopt/torch/fastgen/methods/pdd.py",
)
_FORBIDDEN_IMPORT_ROOTS = {"diffusers", "fastgen", "nemo_automodel", "transformers"}
_FORBIDDEN_RELATIVE_COMPONENTS = {"plugins", "qwen_image", "qwen_image_pdd"}


def test_core_pdd_sources_do_not_import_model_or_framework_packages() -> None:
    repository = Path(__file__).resolve().parents[4]
    violations = []

    for relative_path in _CORE_SOURCES:
        source_path = repository / relative_path
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
                forbidden = [
                    name for name in imported if name.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS
                ]
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported = [node.module]
                if node.level == 0:
                    forbidden = [
                        name
                        for name in imported
                        if name.split(".", 1)[0] in _FORBIDDEN_IMPORT_ROOTS
                    ]
                else:
                    forbidden = [
                        name
                        for name in imported
                        if set(name.split(".")).intersection(_FORBIDDEN_RELATIVE_COMPONENTS)
                    ]
            else:
                continue
            violations.extend((relative_path, name) for name in forbidden)

    assert violations == []


def test_core_pdd_symbols_are_exported_from_the_public_package() -> None:
    expected = {
        "PDDConfig": PDDConfig,
        "PDDLayerSpec": PDDLayerSpec,
        "PDDModelAdapter": PDDModelAdapter,
        "PDDOutputProjection": PDDOutputProjection,
        "PDDPipeline": PDDPipeline,
        "convert_to_pdd_output_projection": convert_to_pdd_output_projection,
        "fusion_coefficients": fusion_coefficients,
        "get_module_by_path": get_module_by_path,
        "integrate_interval_velocities": integrate_interval_velocities,
        "load_pdd_config": load_pdd_config,
        "make_shifted_flow_grid": make_shifted_flow_grid,
        "replace_module_by_path": replace_module_by_path,
    }

    assert {name: getattr(fastgen, name) for name in expected} == expected


def test_fresh_core_import_does_not_load_model_plugins_or_frameworks() -> None:
    repository = Path(__file__).resolve().parents[4]
    script = r"""
import importlib.abc
import sys


class _UnavailableOptionalFrameworks(importlib.abc.MetaPathFinder):
    prefixes = ("diffusers", "fastgen", "nemo_automodel", "transformers")

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        del path, target
        if any(fullname == prefix or fullname.startswith(prefix + ".") for prefix in self.prefixes):
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"unavailable optional framework {fullname}", name=fullname)
        return None


blocker = _UnavailableOptionalFrameworks()
sys.meta_path.insert(0, blocker)
import modelopt.torch

baseline = set(sys.modules)
blocker.attempts.clear()
from modelopt.torch.fastgen import PDDConfig, PDDPipeline

assert all(symbol is not None for symbol in (PDDConfig, PDDPipeline))
assert blocker.attempts == []
loaded = set(sys.modules) - baseline
for prefix in (
    "diffusers",
    "fastgen",
    "nemo_automodel",
    "modelopt.torch.fastgen.plugins.qwen_image",
    "transformers",
):
    assert not any(name == prefix or name.startswith(prefix + ".") for name in loaded), (
        prefix,
        sorted(name for name in loaded if name == prefix or name.startswith(prefix + ".")),
    )
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repository,
        check=False,
        capture_output=True,
        close_fds=True,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_optional_plugins_remain_available_through_explicit_public_access() -> None:
    assert fastgen.plugins.__name__ == "modelopt.torch.fastgen.plugins"
