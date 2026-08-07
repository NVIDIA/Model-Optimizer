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

"""Validate metadata for retained historical Puzzletron campaign results."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

CAMPAIGN_IDS = ("nemotron3_nano_30b_a3b", "qwen3p5_9b")
CAMPAIGN_DATA_MARKER = '<script id="campaign-data" type="application/json">'
REPORT_SCAN_CHUNK_BYTES = 1_048_576
REPORT_SUMMARY_MAX_BYTES = 16_777_216
REPORT_SUMMARY_FIELDS = (
    "stage_status",
    "merged_config",
    "activation_diagnostic_view",
)
SLICING_WARNING_KINDS = {"equivalence_tolerance", "descriptor_realization_gate"}
LEGACY_MIP_GOAL_FIELDS = {
    "num_params_ratio": "params",
    "target_latency_ratio": "runtime",
    "target_memory_ratio": "memory",
}


@pytest.fixture(scope="module")
def historical_campaigns(project_root_path: Path) -> dict:
    manifest_path = project_root_path / "examples/puzzletron/reports/historical_campaigns.yaml"
    return yaml.safe_load(manifest_path.read_text())


def test_historical_campaign_set(historical_campaigns: dict) -> None:
    assert historical_campaigns["schema_version"] == 1
    assert {campaign["id"] for campaign in historical_campaigns["campaigns"]} == set(CAMPAIGN_IDS)


def test_historical_result_matrix_campaign_set(
    project_root_path: Path, historical_campaigns: dict
) -> None:
    matrix_path = project_root_path / "examples/puzzletron/docs/historical_results.md"
    matrix_rows = _historical_campaign_rows(matrix_path.read_text())

    assert Counter(matrix_rows.keys()) == Counter(
        campaign["display_name"] for campaign in historical_campaigns["campaigns"]
    )


def test_historical_results_are_linked_from_main_docs(project_root_path: Path) -> None:
    readme = (project_root_path / "examples/puzzletron/README.md").read_text()

    assert "(docs/historical_results.md)" in readme


@pytest.mark.parametrize("campaign_id", CAMPAIGN_IDS)
def test_historical_campaign_matches_current_config(
    project_root_path: Path, historical_campaigns: dict, campaign_id: str
) -> None:
    campaign = _campaign_by_id(historical_campaigns, campaign_id)
    config_path = project_root_path / campaign["current_config"]
    model_config_path = project_root_path / campaign["model_config"]
    config = yaml.safe_load(config_path.read_text())
    model_config = yaml.safe_load(model_config_path.read_text())

    config_root = project_root_path / "examples/puzzletron/configs"
    model_default = "/" + str(model_config_path.relative_to(config_root).with_suffix(""))
    model_default += "@_global_"

    assert campaign["classification"] == "historical_unreproduced"
    assert campaign["reproduced_on_current_code"] is False
    assert campaign["support_claim"] is False
    assert model_default in config["defaults"]
    assert model_config["model_info"]["hf_repo"] == campaign["model_repo"]
    assert config["data"]["max_sample_length"] == campaign["reported_sequence_length"]
    assert config["embedding_pruning"]["widths"] == campaign["current_embedding_widths"]

    enabled_axes = {
        axis
        for axis, axis_config in model_config["search_space"]["axes"].items()
        if axis_config.get("enabled")
    }
    assert enabled_axes == set(campaign["current_model_axes"])
    assert set(campaign["reported_mip_profiles"]) <= set(config["mip"]["runs"])

    current_goal_dimensions = {
        dimension
        for run in config["mip"]["runs"].values()
        for dimension in (run.get("constraints") or {})
    }
    assert current_goal_dimensions == set(campaign["mip_goal_dimensions"])

    completed = set(campaign["completed_stages"])
    pending = set(campaign["pending_enabled_stages"])
    assert completed.isdisjoint(pending)
    assert campaign["reported_boundary_stage"] in completed
    for stage in pending:
        assert config[stage]["enabled"] is True


@pytest.mark.parametrize("campaign_id", CAMPAIGN_IDS)
def test_historical_campaign_matches_report_summary(
    project_root_path: Path, historical_campaigns: dict, campaign_id: str
) -> None:
    campaign = _campaign_by_id(historical_campaigns, campaign_id)
    report_path = project_root_path / campaign["report"]
    report = _load_report_summary(report_path)
    report_config = report["merged_config"]

    assert report_config["model_info"]["hf_repo"] == campaign["model_repo"]
    assert report_config["data"]["max_sample_length"] == campaign["reported_sequence_length"]
    assert report_config["embedding_pruning"]["widths"] == campaign["reported_embedding_widths"]
    assert len(report_config["_runtime"]["overrides"]) == campaign["report_config_override_count"]

    completed = {stage for stage, status in report["stage_status"].items() if status == "completed"}
    pending = {stage for stage, status in report["stage_status"].items() if status == "pending"}
    assert completed == set(campaign["completed_stages"])
    assert pending == set(campaign["pending_enabled_stages"])

    report_axes = {
        axis
        for axis, axis_config in report_config["search_space"]["axes"].items()
        if axis_config.get("enabled")
    }
    diagnostic_axes = {
        "embedding_width" if row["axis"] == "hidden_width" else row["axis"]
        for row in report["activation_diagnostic_view"]["rows"]
    }
    report_axes.update(diagnostic_axes)
    if report["stage_status"].get("depth_importance") == "completed":
        report_axes.add("conditional_depth")
    assert report_axes == set(campaign["reported_pruning_dimensions"])

    assert _report_profile_ids(report_config) == set(campaign["reported_mip_profiles"])
    assert _report_goal_dimensions(report_config) == set(campaign["mip_goal_dimensions"])

    slicing_findings = [
        finding
        for finding in report["activation_diagnostic_view"]["findings"]
        if finding["stage"] == "slicing_sanity" and finding["severity"] == "warning"
    ]
    finding_counts = Counter(finding["evidence"]["kind"] for finding in slicing_findings)
    assert set(finding_counts) <= SLICING_WARNING_KINDS

    slicing_warnings = campaign["known_slicing_warnings"]
    assert finding_counts == Counter(
        {
            "equivalence_tolerance": slicing_warnings["equivalence_tolerance_findings"],
            "descriptor_realization_gate": slicing_warnings["descriptor_realization_gate_findings"],
        }
    )


@pytest.mark.parametrize("campaign_id", CAMPAIGN_IDS)
def test_historical_result_matrix_row_matches_manifest(
    project_root_path: Path, historical_campaigns: dict, campaign_id: str
) -> None:
    campaign = _campaign_by_id(historical_campaigns, campaign_id)
    matrix_path = project_root_path / "examples/puzzletron/docs/historical_results.md"
    matrix = matrix_path.read_text()
    row = _historical_campaign_rows(matrix)[campaign["display_name"]]

    puzzletron_root = project_root_path / "examples/puzzletron"
    config_path = project_root_path / campaign["current_config"]
    report_path = project_root_path / campaign["report"]
    config_link = (Path("..") / config_path.relative_to(puzzletron_root)).as_posix()
    report_link = (Path("..") / report_path.relative_to(puzzletron_root)).as_posix()
    slicing_warnings = campaign["known_slicing_warnings"]
    report_metadata = row["Retained report metadata"]
    config_relationship = row["Current configuration relationship"]
    status = row["Reproduction and correctness status"]

    assert "not model-support claims" in matrix
    assert f"[Campaign report]({report_link})" in report_metadata
    assert f"sequence length: {campaign['reported_sequence_length']:,}" in report_metadata
    for profile_id in campaign["reported_mip_profiles"]:
        assert f"`{profile_id}`" in report_metadata
    assert f"reported boundary: `{campaign['reported_boundary_stage']}`" in report_metadata

    assert f"[default.yaml]({config_link})" in config_relationship
    relationship_labels = {
        "migrated_current_entry": "current-code migration",
        "reconstructed_current_entry": "reconstruction",
    }
    assert relationship_labels[campaign["config_relationship"]] in config_relationship
    if campaign["reported_embedding_widths"] != campaign["current_embedding_widths"]:
        assert f"{campaign['report_config_override_count']} overrides" in config_relationship
        for width in campaign["reported_embedding_widths"]:
            assert f"{width:,}" in config_relationship
        for width in campaign["current_embedding_widths"]:
            assert f"retains {width:,}" in config_relationship

    assert "Not reproduced on current code" in status
    assert f"{slicing_warnings['equivalence_tolerance_findings']} slicing-equivalence" in status
    descriptor_findings = slicing_warnings["descriptor_realization_gate_findings"]
    if descriptor_findings:
        assert f"{descriptor_findings} descriptor-realization-gate" in status
    else:
        assert "descriptor-realization-gate" not in status
    for stage in campaign["pending_enabled_stages"]:
        assert f"`{stage}`" in status


def _campaign_by_id(evidence: dict, campaign_id: str) -> dict:
    return next(campaign for campaign in evidence["campaigns"] if campaign["id"] == campaign_id)


def _load_report_summary(report_path: Path) -> dict:
    report_prefix = b""
    with report_path.open("rb") as report_file:
        while len(report_prefix) < REPORT_SUMMARY_MAX_BYTES:
            chunk = report_file.read(REPORT_SCAN_CHUNK_BYTES)
            if not chunk:
                break
            report_prefix += chunk
            try:
                campaign_data = report_prefix.decode().split(CAMPAIGN_DATA_MARKER, maxsplit=1)[1]
                decoder = json.JSONDecoder()
                summary = {}
                for field in REPORT_SUMMARY_FIELDS:
                    marker = f'"{field}": '
                    value_start = campaign_data.index(marker) + len(marker)
                    summary[field], _ = decoder.raw_decode(campaign_data, value_start)
            except (IndexError, UnicodeDecodeError, ValueError):
                continue
            return summary

    raise AssertionError(
        f"campaign summary exceeds the {REPORT_SUMMARY_MAX_BYTES}-byte scan limit: {report_path}"
    )


def _report_profile_ids(report_config: dict) -> set[str]:
    profile_ids = set(report_config.get("mip", {}).get("profiles") or {})

    def collect_profile_ids(value: object) -> None:
        if isinstance(value, dict):
            for key, nested_value in value.items():
                if key == "profile_id" and isinstance(nested_value, str):
                    profile_ids.add(nested_value)
                collect_profile_ids(nested_value)
        elif isinstance(value, list):
            for nested_value in value:
                collect_profile_ids(nested_value)

    collect_profile_ids(report_config)
    return profile_ids


def _report_goal_dimensions(report_config: dict) -> set[str]:
    mip_config = report_config["mip"]
    dimensions = {
        dimension
        for profile in (mip_config.get("profiles") or {}).values()
        for dimension in (profile.get("constraints") or {})
    }
    dimensions.update(
        LEGACY_MIP_GOAL_FIELDS[field]
        for field in (mip_config.get("human_constraints") or {})
        if field in LEGACY_MIP_GOAL_FIELDS
    )
    return dimensions


def _historical_campaign_rows(matrix: str) -> dict[str, dict[str, str]]:
    section = matrix.split("## Historical result summary", maxsplit=1)[1]
    section = section.split("## Evidence boundary", maxsplit=1)[0]
    table_rows = [line for line in section.splitlines() if line.startswith("|")]
    headers = [cell.strip() for cell in table_rows[0].strip("|").split("|")]
    return {
        cells[0]: dict(zip(headers, cells, strict=True))
        for row in table_rows[2:]
        if (cells := [cell.strip() for cell in row.strip("|").split("|")])
    }
