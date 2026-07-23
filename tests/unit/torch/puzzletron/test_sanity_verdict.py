# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

from modelopt.torch.puzzletron.diagnostics.sanity_verdict import (
    SanityVerdict,
    complete_sanity_stage,
    finding_from_message,
)
from modelopt.torch.puzzletron.manifest import StageManifest


def test_complete_sanity_stage_allows_warnings_by_default(tmp_path: Path):
    config = {"experiment": {"dir": str(tmp_path)}}
    manifest = StageManifest(stage="sort_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        outputs={"summary_path": "artifacts/sort_sanity/summary.json"},
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

    assert result.status == "success"
    assert manifest.status == "success"
    assert manifest.outputs["passed"] is False
    assert manifest.outputs["verdict"] == "warning"
    assert manifest.outputs["findings"]


def test_complete_sanity_stage_fails_when_warnings_are_strict(tmp_path: Path):
    config = {
        "experiment": {"dir": str(tmp_path)},
        "sanity": {"fail_on_warnings": True},
    }
    manifest = StageManifest(stage="bypass_sanity", config=config)
    (tmp_path / "manifests").mkdir(parents=True)

    result = complete_sanity_stage(
        config,
        manifest,
        verdict=SanityVerdict(
            passed=False,
            findings=[
                finding_from_message(
                    stage="bypass_sanity",
                    message="overfit probe did not improve",
                )
            ],
        ),
    )

    assert result.status == "failed"
    assert manifest.status == "failed"
    assert manifest.outputs["passed"] is False
    assert manifest.outputs["verdict"] == "warning"
    assert manifest.outputs["findings"]


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
