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

"""Validate the curated catalog for retained Puzzletron campaign reports."""

from pathlib import Path

import pytest
import yaml

__all__ = []

REQUIRED_REPORT_IDS = {"nemotron3_nano_30b_a3b", "qwen3p5_9b"}
EXPECTED_REPORT_FIELDS = {
    "id",
    "display_name",
    "report",
    "producer_state",
    "producer_revision",
    "reproduction_status",
    "support_status",
    "metadata_origin",
    "current_config",
    "known_limitations",
}
EXPECTED_CURRENT_STATUS = {
    "nemotron3_nano_30b_a3b": "migration",
    "qwen3p5_9b": "reconstruction",
}


@pytest.fixture(scope="module")
def campaign_report_index(project_root_path: Path) -> dict:
    index_path = project_root_path / "examples/puzzletron/reports/campaign_report_index.yaml"
    return yaml.safe_load(index_path.read_text())


def test_campaign_report_index_records_evidence_status(campaign_report_index: dict) -> None:
    assert set(campaign_report_index) == {"schema_version", "purpose", "reports"}
    assert campaign_report_index["schema_version"] == 1

    reports = campaign_report_index["reports"]
    report_ids = [report["id"] for report in reports]
    assert len(report_ids) == len(set(report_ids))
    assert set(report_ids) >= REQUIRED_REPORT_IDS
    for report in reports:
        assert set(report) == EXPECTED_REPORT_FIELDS
        assert report["producer_state"]
        assert report["producer_revision"]
        assert report["reproduction_status"]
        assert report["support_status"]
        assert report["metadata_origin"]
        assert set(report["current_config"]) == {"path", "relationship", "note"}
        assert report["current_config"]["relationship"]
        assert isinstance(report["known_limitations"], list)

    reports_by_id = {report["id"]: report for report in reports}
    for report_id, config_relationship in EXPECTED_CURRENT_STATUS.items():
        report = reports_by_id[report_id]
        assert report["producer_state"] == "development_snapshot"
        assert report["producer_revision"] == "unknown"
        assert report["reproduction_status"] == "not_reproduced"
        assert report["support_status"] == "not_established"
        assert report["metadata_origin"] == "curated_from_retained_report"
        assert report["current_config"]["relationship"] == config_relationship
        assert report["known_limitations"]


def test_campaign_report_index_references_existing_files(
    project_root_path: Path, campaign_report_index: dict
) -> None:
    for report in campaign_report_index["reports"]:
        assert (project_root_path / report["report"]).is_file()
        assert (project_root_path / report["current_config"]["path"]).is_file()


def test_campaign_reports_are_reachable_from_main_docs(
    project_root_path: Path, campaign_report_index: dict
) -> None:
    puzzletron_root = project_root_path / "examples/puzzletron"
    readme = (puzzletron_root / "README.md").read_text()
    summary = (puzzletron_root / "docs/campaign_reports.md").read_text()

    assert "(docs/campaign_reports.md)" in readme
    assert "(../reports/campaign_report_index.yaml)" in summary
    for report in campaign_report_index["reports"]:
        report_path = Path(report["report"]).relative_to("examples/puzzletron")
        report_link = (Path("..") / report_path).as_posix()
        config_path = Path(report["current_config"]["path"]).relative_to("examples/puzzletron")
        config_link = (Path("..") / config_path).as_posix()
        assert report["display_name"] in summary
        assert f"({report_link})" in summary
        assert f"({config_link})" in summary
        report_row = next(line for line in summary.splitlines() if report["display_name"] in line)
        assert f"`{report['producer_state']}`" in report_row
        assert f"`{report['producer_revision']}`" in report_row
        assert f"`{report['reproduction_status']}`" in report_row
        assert f"`{report['support_status']}`" in report_row
        assert f"`{report['current_config']['relationship']}`" in report_row
