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
"""Tests for the Sphinx warning-baseline ratchet."""

from collections import Counter

from tools.ci.check_sphinx_warnings import (
    compare_warning_counts,
    normalize_warning,
    read_warning_counts,
)


def test_normalize_warning_removes_workspace_and_source_lines(tmp_path):
    repo_root = tmp_path / "repo"
    warning = (
        f"{repo_root}/docs/source/generated/example.rst:49:<autosummary>:1: ERROR: broken reference"
    )

    assert normalize_warning(warning, repo_root) == (
        "docs/source/generated/example.rst:<autosummary>: ERROR: broken reference"
    )


def test_read_warning_counts_preserves_multiplicity(tmp_path):
    warning_log = tmp_path / "warnings.log"
    warning_log.write_text(
        "WARNING: inherited\nprogress\nWARNING: inherited\nERROR: inherited error\n",
        encoding="utf-8",
    )

    assert read_warning_counts(warning_log, tmp_path) == Counter(
        {"WARNING: inherited": 2, "ERROR: inherited error": 1}
    )


def test_compare_warning_counts_detects_replacements():
    unexpected, resolved = compare_warning_counts(
        Counter({"WARNING: inherited": 1, "WARNING: new": 1}),
        Counter({"WARNING: inherited": 2}),
    )

    assert unexpected == Counter({"WARNING: new": 1})
    assert resolved == Counter({"WARNING: inherited": 1})
