# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Puzzletron's optional AutoModel import boundary."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def test_lightweight_submodule_import_does_not_require_automodel() -> None:
    script = (
        "import sys; "
        "sys.modules['nemo_automodel'] = None; "
        "from modelopt.torch.puzzletron.identity import stable_hash; "
        "assert stable_hash({'ready': True}, prefix='ci')"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_width_slice_use_reports_missing_automodel() -> None:
    script = """
import sys

sys.modules["nemo_automodel"] = None

from modelopt.torch.puzzletron.diagnostics.width_slice_equivalence import (
    _replace_block_scoring_recipe,
)

try:
    _replace_block_scoring_recipe()
except ImportError as error:
    assert "requires a compatible NeMo AutoModel installation" in str(error)
else:
    raise AssertionError("width-slice recipe unexpectedly imported without AutoModel")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
