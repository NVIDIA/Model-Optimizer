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

"""Tests for attaching speculation_profile.json at export time."""

import json

import pytest

from modelopt.torch.export.plugins.hf_spec_export import SpeculativeDecodingExporter


class _Exporter(SpeculativeDecodingExporter):
    """Minimal concrete exporter -- export() is irrelevant to profile transport."""

    def export(self, export_dir, dtype=None):
        raise NotImplementedError


@pytest.fixture
def exporter():
    return _Exporter(model=object())


def test_supplied_profile_is_copied_verbatim(tmp_path, exporter):
    src = tmp_path / "profile.json"
    payload = {
        "schema_version": "1.0",
        "measured": True,
        "conditional_accept_rates": [0.816082, 0.776577, 0.749591],
        "mean_accept_length": 2.924887,
    }
    src.write_text(json.dumps(payload))
    out = tmp_path / "export"
    out.mkdir()

    exporter.write_speculation_profile(out, src)

    assert json.loads((out / "speculation_profile.json").read_text()) == payload


def test_stub_is_written_when_none_supplied(tmp_path, exporter):
    out = tmp_path / "export"
    out.mkdir()

    exporter.write_speculation_profile(out, None)

    stub = json.loads((out / "speculation_profile.json").read_text())
    # "not measured" must be distinguishable from "predates the schema"; absent then
    # means a genuinely old checkpoint rather than an ambiguous one.
    assert stub["measured"] is False
    assert "specdec_bench" in stub["note"]


def test_missing_file_is_a_hard_error(tmp_path, exporter):
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(FileNotFoundError):
        exporter.write_speculation_profile(out, tmp_path / "nope.json")


@pytest.mark.parametrize("payload", ['{"no_version": true}', "[1, 2, 3]", '"a string"'])
def test_non_profile_json_is_rejected(tmp_path, exporter, payload):
    """Silently shipping an unrelated JSON file as a profile would be worse than failing."""
    src = tmp_path / "thing.json"
    src.write_text(payload)
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(ValueError, match="schema_version"):
        exporter.write_speculation_profile(out, src)


def test_malformed_json_is_rejected(tmp_path, exporter):
    """A truncated or corrupt file must fail on the parser, not slip through.

    Distinct from the cases above: those are valid JSON of the wrong shape, this one
    never parses. A half-written profile from an interrupted run is the realistic way
    to hit it.
    """
    src = tmp_path / "truncated.json"
    src.write_text('{"schema_version": "1.0", "conditional_accept_rates": [0.8,')
    out = tmp_path / "export"
    out.mkdir()
    with pytest.raises(json.JSONDecodeError):
        exporter.write_speculation_profile(out, src)


def test_schema_version_is_not_pinned(tmp_path, exporter):
    """Producers own the schema; pinning a version here would create a second source
    of truth that drifts. A future version must pass through untouched."""
    src = tmp_path / "profile.json"
    src.write_text(json.dumps({"schema_version": "99.0", "measured": True}))
    out = tmp_path / "export"
    out.mkdir()

    exporter.write_speculation_profile(out, src)

    assert json.loads((out / "speculation_profile.json").read_text())["schema_version"] == "99.0"
