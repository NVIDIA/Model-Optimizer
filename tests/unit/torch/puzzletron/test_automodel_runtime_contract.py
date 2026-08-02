# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the CPU package environment assembled by the Puzzletron Nox session.

This contract does not cover the CUDA, patched-vLLM, AIPerf, or full campaign
runtime documented for GPU execution.
"""

import json
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from packaging.version import Version

RUNTIME_CONTRACT = (
    Path(__file__).resolve().parents[4] / "examples/puzzletron/runtime_contract.json"
)


def _runtime_contract() -> dict[str, Any]:
    return json.loads(RUNTIME_CONTRACT.read_text(encoding="utf-8"))


def _automodel_source_commit() -> str:
    direct_url_text = distribution("nemo-automodel").read_text("direct_url.json")
    assert direct_url_text is not None, "AutoModel must retain direct-source provenance"
    direct_url = json.loads(direct_url_text)

    commit_id = direct_url.get("vcs_info", {}).get("commit_id")
    if commit_id:
        return str(commit_id)

    if direct_url.get("dir_info", {}).get("editable"):
        source_url = str(direct_url.get("url", ""))
        assert source_url.startswith("file:"), source_url
        source_path = Path(unquote(urlparse(source_url).path))
        result = subprocess.run(
            ["git", "-C", str(source_path), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    raise AssertionError("AutoModel must be installed from the pinned source checkout")


def test_puzzletron_v2_package_versions_match_runtime_contract() -> None:
    contract = _runtime_contract()
    automodel = contract["nemo_automodel"]
    assert isinstance(automodel, dict)

    assert sys.version_info[:2] == tuple(map(int, str(contract["python"]).split(".")))
    assert Version(version("torch")).release == Version(str(contract["torch"])).release
    assert Version(version("torchvision")).release == Version(
        str(contract["torchvision"])
    ).release
    assert version("transformers") == contract["transformers"]
    assert Version(version("nemo-automodel")).release == Version(
        str(automodel["version"])
    ).release


def test_puzzletron_v2_automodel_source_matches_runtime_contract() -> None:
    automodel = _runtime_contract()["nemo_automodel"]
    assert isinstance(automodel, dict)
    assert _automodel_source_commit() == automodel["commit"]
