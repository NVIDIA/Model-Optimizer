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

from __future__ import annotations

from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.sanity_verdict import (
    SanityVerdict,
    complete_sanity_stage,
    finding_from_message,
)
from modelopt.torch.puzzletron.manifest import StageManifest


def test_complete_sanity_stage_allows_advisory_warnings_by_default(tmp_path: Path):
    config = {"experiment": {"dir": str(tmp_path)}}
    manifest = StageManifest(stage="width_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        outputs={"summary_path": "artifacts/width_sanity/summary.json"},
        verdict=SanityVerdict(
            passed=False,
            findings=[
                finding_from_message(
                    stage="width_sanity",
                    message="activation ranking is worse than reverse",
                )
            ],
        ),
    )

    assert result.status == "success"
    assert manifest.status == "success"
    assert manifest.outputs["passed"] is False
    assert manifest.outputs["verdict"] == "warning"
    assert manifest.outputs["blocking"] is False
    assert manifest.outputs["findings"]


def test_complete_sanity_stage_fails_advisory_warnings_when_strict(tmp_path: Path):
    config = {
        "experiment": {"dir": str(tmp_path)},
        "sanity": {"fail_on_warnings": True},
    }
    manifest = StageManifest(stage="width_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(
            passed=False,
            findings=[
                finding_from_message(
                    stage="width_sanity",
                    message="activation ranking is worse than reverse",
                )
            ],
        ),
    )

    assert result.status == "failed"
    assert manifest.status == "failed"
    assert manifest.outputs["passed"] is False
    assert manifest.outputs["verdict"] == "warning"
    assert manifest.outputs["blocking"] is False
    assert manifest.outputs["findings"]


def test_complete_sanity_stage_fails_sort_correctness_by_default(tmp_path: Path):
    config = {"experiment": {"dir": str(tmp_path)}}
    manifest = StageManifest(stage="sort_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(
            passed=False,
            findings=[
                finding_from_message(
                    stage="sort_sanity",
                    message="sorted teacher drift too large",
                )
            ],
        ),
    )

    assert result.status == "failed"
    assert manifest.status == "failed"
    assert manifest.outputs["verdict"] == "failed"
    assert manifest.outputs["blocking"] is True
    assert manifest.outputs["findings"][0]["severity"] == "error"


def test_warning_policy_cannot_downgrade_slicing_correctness_failure(tmp_path: Path):
    config = {
        "experiment": {"dir": str(tmp_path)},
        "sanity": {"fail_on_warnings": False},
    }
    manifest = StageManifest(stage="slicing_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(
            passed=False,
            findings=[
                finding_from_message(
                    stage="slicing_sanity",
                    message="dynamic and physical slices disagree",
                )
            ],
        ),
    )

    assert result.status == "failed"
    assert manifest.status == "failed"
    assert manifest.outputs["verdict"] == "failed"
    assert manifest.outputs["blocking"] is True


def test_folded_correctness_failure_preserves_advisory_finding_severity(tmp_path: Path):
    config = {"experiment": {"dir": str(tmp_path)}}
    manifest = StageManifest(stage="width_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(
            passed=False,
            blocking=True,
            findings=[
                finding_from_message(
                    stage="width_sanity", message="ranking quality regressed"
                ),
                finding_from_message(
                    stage="sort_sanity", message="sorted teacher drifted"
                ),
            ],
        ),
    )

    assert result.status == "failed"
    assert manifest.outputs["blocking"] is True
    assert [finding["severity"] for finding in manifest.outputs["findings"]] == [
        "warning",
        "error",
    ]


def test_complete_sanity_stage_strict_policy_does_not_fail_passed_verdict(tmp_path: Path):
    config = {
        "experiment": {"dir": str(tmp_path)},
        "sanity": {"fail_on_warnings": True},
    }
    manifest = StageManifest(stage="sort_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(passed=True),
    )

    assert result.status == "success"
    assert manifest.status == "success"


def test_finding_from_message_shape():
    finding = finding_from_message(stage="width_sanity", message="example", evidence={"x": 1})
    assert finding["stage"] == "width_sanity"
    assert finding["severity"] == "warning"
    assert finding["evidence"]["x"] == 1


def test_finding_from_message_accepts_correctness_severity():
    finding = finding_from_message(
        stage="sort_sanity",
        message="example",
        severity="error",
    )

    assert finding["severity"] == "error"
