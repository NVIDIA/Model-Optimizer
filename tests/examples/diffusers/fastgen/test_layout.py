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
import re
import subprocess
import sys

import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
_FASTGEN_ROOT = _REPO_ROOT / "examples" / "diffusers" / "fastgen"
_EXPECTED_ROOT_ENTRIES = {
    "README.md",
    "dmd2",
    "fastgen_data",
    "make_negative_prompt_embedding.py",
    "preprocess",
    "preprocess_qwen_image.py",
    "pdd",
    "requirements.txt",
}
_EXPECTED_DMD2_FILES = {
    "README.md",
    "__init__.py",
    "checkpoint.py",
    "configs",
    "export_qwen_image.py",
    "finetune.py",
    "inference_qwen_image.py",
    "recipe.py",
}
_EXPECTED_PDD_FILES = {
    "README.md",
    "__init__.py",
    "artifacts.py",
    "checkpoint.py",
    "configs",
    "data.py",
    "export.py",
    "export_qwen_image.py",
    "finetune.py",
    "inference_runtime.py",
    "inference_qwen_image.py",
    "recipe.py",
    "training.py",
}
_TEXT_SUFFIXES = {".json", ".md", ".py", ".rst", ".sh", ".toml", ".txt", ".yaml", ".yml"}


def _old_name(prefix: str, suffix: str) -> str:
    return f"{prefix}_{suffix}"


def _old_modules() -> tuple[str, ...]:
    return (
        _old_name("dmd2", "finetune"),
        _old_name("dmd2", "recipe"),
        _old_name("fastgen", "checkpoint"),
        _old_name("export", "diffusers_qwen_image"),
        _old_name("inference", "dmd2_qwen_image"),
        _old_name("pdd", "artifacts"),
        _old_name("pdd", "checkpoint"),
        _old_name("pdd", "export"),
        _old_name("pdd", "finetune"),
        _old_name("pdd", "recipe"),
        _old_name("pdd", "training"),
        _old_name("export", "pdd_qwen_image"),
        _old_name("inference", "pdd_qwen_image"),
    )


def _source_text_files() -> list[pathlib.Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        _REPO_ROOT / relative
        for relative in completed.stdout.decode().split("\0")
        if relative
        and pathlib.Path(relative).suffix in _TEXT_SUFFIXES
        and (_REPO_ROOT / relative).is_file()
    ]


def test_fastgen_root_has_closed_shared_and_algorithm_ownership() -> None:
    root_entries = {path.name for path in _FASTGEN_ROOT.iterdir() if path.name != "__pycache__"}
    assert root_entries == _EXPECTED_ROOT_ENTRIES
    assert not (_FASTGEN_ROOT / "configs").exists()

    dmd2_entries = {
        path.name for path in (_FASTGEN_ROOT / "dmd2").iterdir() if path.name != "__pycache__"
    }
    assert dmd2_entries == _EXPECTED_DMD2_FILES
    assert {path.name for path in (_FASTGEN_ROOT / "dmd2" / "configs").iterdir()} == {
        "qwen_image.yaml"
    }
    pdd_entries = {
        path.name for path in (_FASTGEN_ROOT / "pdd").iterdir() if path.name != "__pycache__"
    }
    assert pdd_entries == _EXPECTED_PDD_FILES
    assert {path.name for path in (_FASTGEN_ROOT / "pdd" / "configs").iterdir()} == {
        "qwen_image.yaml"
    }


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


def test_repository_sources_have_no_flat_algorithm_paths() -> None:
    old_modules = _old_modules()
    stale = (
        *(f"{module}.py" for module in old_modules),
        "configs/" + _old_name("dmd2", "qwen_image") + ".yaml",
    )
    failures: list[str] = []
    for path in _source_text_files():
        if path == pathlib.Path(__file__):
            continue
        text = path.read_text(errors="strict")
        relative = path.relative_to(_REPO_ROOT).as_posix()
        failures.extend(f"{relative}: {token}" for token in stale if token in text)
        if path.suffix == ".py":
            failures.extend(
                f"{relative}: stale import {module}"
                for module in old_modules
                if re.search(rf"(?m)^\s*(?:from|import)\s+{re.escape(module)}(?:\s|\.|$)", text)
                or re.search(rf"['\"]{re.escape(module)}['\"]", text)
            )
    assert not failures, "\n".join(failures)


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
