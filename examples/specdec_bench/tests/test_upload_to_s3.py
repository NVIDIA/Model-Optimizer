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

"""Tests for the pure-Python parts of upload_to_s3.py (no boto3 required).

Covers parse_s3_path edge inputs and _discover_runs flat-vs-sweep layout
detection — both flagged by cjluo as missing test coverage.
"""

import importlib.util
from pathlib import Path

import pytest

_PKG_ROOT = Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location("_upload_to_s3", _PKG_ROOT / "upload_to_s3.py")
upload_to_s3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(upload_to_s3)


class TestParseS3Path:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("s3://bucket/prefix", ("bucket", "prefix")),
            ("s3://bucket/prefix/", ("bucket", "prefix")),
            ("s3://bucket/a/b/c", ("bucket", "a/b/c")),
            ("s3://bucket/a/b/c/", ("bucket", "a/b/c")),
            # Empty-prefix forms: cjluo's edge case.
            ("s3://bucket", ("bucket", "")),
            ("s3://bucket/", ("bucket", "")),
        ],
    )
    def test_parsing(self, path, expected):
        assert upload_to_s3.parse_s3_path(path) == expected


def _make_run_dir(path: Path) -> Path:
    """Create a directory shaped like a specdec_bench run output."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "configuration.json").write_text("{}")
    (path / "timing.json").write_text("{}")
    return path


class TestIsRunDir:
    def test_sentinel_present(self, tmp_path):
        run = _make_run_dir(tmp_path / "r")
        assert upload_to_s3._is_run_dir(run) is True

    def test_empty_dir(self, tmp_path):
        assert upload_to_s3._is_run_dir(tmp_path) is False

    def test_non_sentinel_files(self, tmp_path):
        (tmp_path / "results.txt").write_text("")
        assert upload_to_s3._is_run_dir(tmp_path) is False


class TestDiscoverRuns:
    def test_single_run_dir(self, tmp_path):
        run = _make_run_dir(tmp_path / "myrun")
        queue = upload_to_s3._discover_runs(run, "results")
        assert queue == [(run, "results/myrun")]

    def test_flat_layout(self, tmp_path):
        root = tmp_path / "sweep_outputs"
        a = _make_run_dir(root / "a")
        b = _make_run_dir(root / "b")
        queue = upload_to_s3._discover_runs(root, "results")
        assert sorted(queue) == sorted(
            [(a, "results/sweep_outputs/a"), (b, "results/sweep_outputs/b")]
        )

    def test_sweep_layout(self, tmp_path):
        root = tmp_path / "sweep_outputs"
        sweep = root / "my_sweep"
        run1 = _make_run_dir(sweep / "001")
        run2 = _make_run_dir(sweep / "002")
        queue = upload_to_s3._discover_runs(root, "results")
        assert sorted(queue) == sorted(
            [
                (run1, "results/sweep_outputs/my_sweep/001"),
                (run2, "results/sweep_outputs/my_sweep/002"),
            ]
        )

    def test_empty_prefix_no_leading_slash(self, tmp_path):
        """cjluo's regression: empty s3_prefix_base must not produce keys with
        a leading '/' (which would create unwanted /<key> entries in S3)."""
        run = _make_run_dir(tmp_path / "myrun")
        queue = upload_to_s3._discover_runs(run, "")
        assert queue == [(run, "myrun")]
        for _, key in queue:
            assert not key.startswith("/")

    def test_empty_prefix_flat_layout(self, tmp_path):
        root = tmp_path / "sweep_outputs"
        _make_run_dir(root / "a")
        queue = upload_to_s3._discover_runs(root, "")
        for _, key in queue:
            assert not key.startswith("/")
            assert key.startswith("sweep_outputs/")

    def test_ignores_non_run_files(self, tmp_path):
        root = tmp_path / "mixed"
        _make_run_dir(root / "a")
        (root / "notes.txt").write_text("ignore me")
        queue = upload_to_s3._discover_runs(root, "results")
        assert len(queue) == 1
        assert queue[0][0].name == "a"
