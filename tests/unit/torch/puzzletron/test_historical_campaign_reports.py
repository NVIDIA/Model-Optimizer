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

"""Validate the curated index for retained historical Puzzletron reports."""

from pathlib import Path

import pytest
import yaml

EXPECTED_REPORT_IDS = {"nemotron3_nano_30b_a3b", "qwen3p5_9b"}
EXPECTED_REPORT_FIELDS = {
    "id",
    "display_name",
    "report",
    "classification",
    "support_status",
    "current_config",
    "known_limitations",
}
EXPECTED_CONFIG_RELATIONSHIPS = {"migration", "reconstruction"}


@pytest.fixture(scope="module")
def historical_report_index(project_root_path: Path) -> dict:
    index_path = project_root_path / "examples/puzzletron/reports/historical_report_index.yaml"
    return yaml.safe_load(index_path.read_text())


def test_historical_report_index_has_narrow_legacy_scope(historical_report_index: dict) -> None:
    assert set(historical_report_index) == {"schema_version", "purpose", "reports"}
    assert historical_report_index["schema_version"] == 1

    reports = historical_report_index["reports"]
    assert {report["id"] for report in reports} == EXPECTED_REPORT_IDS
    for report in reports:
        assert set(report) == EXPECTED_REPORT_FIELDS
        assert report["classification"] == "historical_unreproduced"
        assert report["support_status"] == "not_established"
        assert set(report["current_config"]) == {"path", "relationship", "note"}
        assert report["current_config"]["relationship"] in EXPECTED_CONFIG_RELATIONSHIPS
        assert report["known_limitations"]


def test_historical_report_index_references_existing_files(
    project_root_path: Path, historical_report_index: dict
) -> None:
    for report in historical_report_index["reports"]:
        assert (project_root_path / report["report"]).is_file()
        assert (project_root_path / report["current_config"]["path"]).is_file()


def test_historical_reports_are_reachable_from_main_docs(
    project_root_path: Path, historical_report_index: dict
) -> None:
    puzzletron_root = project_root_path / "examples/puzzletron"
    readme = (puzzletron_root / "README.md").read_text()
    summary = (puzzletron_root / "docs/historical_results.md").read_text()

    assert "(docs/historical_results.md)" in readme
    assert "(../reports/historical_report_index.yaml)" in summary
    for report in historical_report_index["reports"]:
        report_path = Path(report["report"]).relative_to("examples/puzzletron")
        report_link = (Path("..") / report_path).as_posix()
        config_path = Path(report["current_config"]["path"]).relative_to("examples/puzzletron")
        config_link = (Path("..") / config_path).as_posix()
        assert report["display_name"] in summary
        assert f"({report_link})" in summary
        assert f"({config_link})" in summary
