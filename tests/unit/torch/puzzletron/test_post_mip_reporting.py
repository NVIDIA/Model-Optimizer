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

import json
from pathlib import Path
from types import SimpleNamespace

from modelopt.torch.puzzletron.post_mip.reporting import (
    build_post_mip_report_payloads,
    render_evaluation_report,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluation_report_compares_teacher_and_deduplicated_mip_origins(tmp_path: Path):
    architecture_id = "architecture_shared"
    revision_id = "revision_candidate"
    execution_id = "post_mip_execution_test"
    node_root = tmp_path / "artifacts/post_mip/nodes/quality"
    _write_json(
        tmp_path / "artifacts/post_mip/candidate_registry.json",
        {
            "architectures": {
                architecture_id: {
                    "origins": [
                        {"kind": "heterogeneous"},
                        {"kind": "homogeneous"},
                    ]
                },
                "architecture_invalid": "invalid",
            },
            "revisions": {
                revision_id: {
                    "architecture_id": architecture_id,
                    "artifact": {"hidden_width": 1024, "kind": "heterogeneous"},
                },
                "failed_revision": {"architecture_id": "architecture_invalid"},
            },
        },
    )
    _write_json(node_root / "summary.json", {"execution_identity": execution_id})
    _write_json(
        node_root / f"executions/{execution_id}/observations.json",
        [
            {
                "input_revision_id": revision_id,
                "status": "success",
                "metrics": {
                    "candidate.lm_loss": 0.7,
                    "candidate.token_accuracy_top_1": 0.8,
                    "candidate.token_accuracy_top_10": 0.95,
                    "reference.lm_loss": 0.5,
                    "reference.token_accuracy_top_1": 0.9,
                    "delta.lm_loss": 0.2,
                },
            },
            {
                "input_revision_id": "failed_revision",
                "status": "failed",
                "metrics": {"reference.lm_loss": None},
                "error": "evaluation failed",
            },
        ],
    )
    node = SimpleNamespace(
        stage_id="post.quality",
        node_id="quality",
        flow_id="post",
        node_type="evaluation",
    )

    payload = build_post_mip_report_payloads(tmp_path, (node,))[node.stage_id]
    rendered = render_evaluation_report(str(payload["section_id"]), payload)

    assert payload["observations"][0]["origin_kinds"] == ["heterogeneous", "homogeneous"]
    assert payload["observations"][1]["origin_kinds"] == []
    assert rendered.count("Reference checkpoint") == 1
    assert "Teacher" in rendered
    assert "Heterogeneous" in rendered
    assert "Homogeneous" in rendered
    assert "Top-1 token accuracy" in rendered
    assert "Top-10 token accuracy" in rendered
    assert "0.95" in rendered
    assert "evaluation failed" in rendered
    assert rendered.count("Shared physical measurement (same architecture)") == 2
