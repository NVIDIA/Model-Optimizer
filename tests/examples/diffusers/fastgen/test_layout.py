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

"""Closed layout contract for the FastGen Diffusers example."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_ROOT = _REPO_ROOT / "examples" / "diffusers" / "fastgen"


def test_fastgen_algorithm_entrypoints_are_namespaced() -> None:
    for algorithm in ("dmd2", "pdd"):
        package = _FASTGEN_ROOT / algorithm
        assert (package / "__init__.py").is_file()
        assert (package / "finetune.py").is_file()
        assert (package / "configs" / "qwen_image.yaml").is_file()

    assert not (_FASTGEN_ROOT / "dmd2_finetune.py").exists()
    assert not (_FASTGEN_ROOT / "pdd_finetune.py").exists()


def test_dmd2_config_retains_accepted_semantics() -> None:
    config_path = _FASTGEN_ROOT / "dmd2" / "configs" / "qwen_image.yaml"
    value = yaml.safe_load(config_path.read_text())
    assert (
        value["data"]["dataloader"]["_target_"]
        == "fastgen_data.build_text_to_image_multiresolution_dataloader"
    )
    assert value["data"]["dataloader"]["negative_prompt_embedding_path"] == (
        "negative_prompt_embedding.pt"
    )
    assert "metadata_index" not in value["data"]["dataloader"]


def test_dmd2_package_is_import_light() -> None:
    probe = f"""
import importlib, json, sys
sys.path.insert(0, {str(_FASTGEN_ROOT)!r})
before = set(sys.modules)
module = importlib.import_module('dmd2')
forbidden = ('torch', 'modelopt', 'nemo_automodel', 'diffusers', 'transformers')
loaded = sorted(name for name in set(sys.modules) - before if name.split('.')[0] in forbidden)
print(json.dumps({{'loaded': loaded, 'all': getattr(module, '__all__', None)}}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert json.loads(completed.stdout) == {"loaded": [], "all": []}


def test_dmd2_help_works_from_repo_root_without_training_imports() -> None:
    script = _FASTGEN_ROOT / "dmd2" / "finetune.py"
    probe = f"""
import json, runpy, sys
sys.argv = [{str(script)!r}, '--help']
before = set(sys.modules)
try:
    runpy.run_path({str(script)!r}, run_name='__main__')
except SystemExit as error:
    if error.code != 0:
        raise
forbidden = ('torch', 'modelopt', 'nemo_automodel', 'diffusers', 'transformers')
loaded = sorted(name for name in set(sys.modules) - before if name.split('.')[0] in forbidden)
print('__FASTGEN_LOADED__=' + json.dumps(loaded))
"""
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    assert "DMD2 Qwen-Image training" in completed.stdout
    assert "--config" in completed.stdout
    loaded_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("__FASTGEN_LOADED__=")
    )
    assert json.loads(loaded_line.partition("=")[2]) == []
