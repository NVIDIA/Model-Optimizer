# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the pinned Puzzletron v2 AutoModel runtime contract."""

import json
import subprocess
import sys
from importlib.metadata import distribution, version
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.version import Version

from modelopt.torch.puzzletron.distillation.global_kd_recipe import (
    KnowledgeDistillationRecipeForNextTokenPrediction,
)
from modelopt.torch.puzzletron.plugins.automodel.scoring_recipe import ActivationScoringRecipe

AUTOMODEL_REF = "b22cd029d806197e249f2cc4a42c5de91713b772"


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


def test_puzzletron_v2_automodel_versions_and_recipe_imports() -> None:
    assert sys.version_info[:2] == (3, 12)
    assert Version(version("torch")).release[:3] == (2, 11, 0)
    assert Version(version("torchvision")).release[:3] == (0, 26, 0)
    assert Version(version("nemo-automodel")).release == (0, 5, 0)
    assert version("transformers") == "5.8.1"
    assert _automodel_source_commit() == AUTOMODEL_REF

    assert (
        KnowledgeDistillationRecipeForNextTokenPrediction.__name__
        == "KnowledgeDistillationRecipeForNextTokenPrediction"
    )
    assert ActivationScoringRecipe.__name__ == "ActivationScoringRecipe"
