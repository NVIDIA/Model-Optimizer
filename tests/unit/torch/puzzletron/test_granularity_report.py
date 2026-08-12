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

from modelopt.torch.puzzletron.diagnostics.campaign_progress_report import (
    _nested_bypass_data,
    generate_campaign_progress_report,
)


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _result(index: int, kind: str, name: str, axis_value: int) -> dict:
    subblock = (
        {"kind": "ffn", "name": name, "intermediate_size": axis_value, "no_op": False}
        if kind == "ffn"
        else {
            "kind": "attention",
            "name": name,
            "num_query_heads": axis_value,
            "num_kv_heads": 2,
            "no_op": False,
        }
    )
    return {
        "i_solution": index,
        "args": {"eval_samples": 8},
        "puzzle_solution": {
            "single_sequence_replacement": {
                "parent_layer_indices": [0],
                "child_block_configs": [{"subblock_configs": [subblock]}],
                "weight_paths": [],
            },
            "subblock_replacement": {"layer_idx": 0, "kind": kind, "name": name},
        },
        "lm_loss": {"avg": 1.0 + index},
        "sliced_teacher_baseline": {"lm_loss": {"avg": 1.0}},
    }


def test_nested_bypass_report_selects_exact_subblock_losses(tmp_path: Path):
    _write(
        tmp_path / "manifests/bypass.json",
        {"config": {"bypass": {"enabled": True, "granularity": "subblock"}}},
    )
    _write(
        tmp_path / "artifacts/bypass/local_kd_loss_history.json",
        {
            "max_steps": 2,
            "records": [
                {
                    "step": 1,
                    "loss": 0.5,
                    "per_layer_loss": {"0": 0.5},
                    "per_subblock_loss": {
                        "0:mamba:linear_attn": 0.25,
                        "0:ffn:feed_forward": 0.75,
                    },
                    "elastic_selection": {"layers": []},
                }
            ],
        },
    )

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Exact semantic subblock" in document
    assert 'for="nested-bypass-unit-select">Sublayer</label>' in document
    assert 'for="nested-bypass-width-select">Hidden width</label>' in document
    assert '<select id="nested-bypass-width-select">' in document
    assert 'id="nested-bypass-axis-filters"' in document
    assert 'id="nested-bypass-config-summary"' in document
    assert "nested-bypass-config-select" not in document
    assert document.count("id='nested-bypass-unit-plot'") == 1
    assert "layer_0:mamba:linear_attn" in document
    assert "layer_0:ffn:feed_forward" in document
    assert "per_subblock_loss" in document


def test_nested_bypass_report_uses_dp_scatter_and_normalized_parameter_color(tmp_path: Path):
    bypass = tmp_path / "artifacts/bypass"
    _write(
        tmp_path / "manifests/bypass.json",
        {"config": {"bypass": {"enabled": True, "granularity": "subblock"}}},
    )
    _write(
        bypass / "local_kd_loss_history.json",
        {
            "max_steps": 1,
            "dp_observation_path": str(bypass / "dp_observations.jsonl"),
            "candidate_catalog_path": str(bypass / "candidate_catalog.json"),
            "records": [{"step": 1, "loss": 0.3}],
        },
    )
    observations = [
        {
            "step": 1,
            "dp_lane": lane,
            "granularity": "subblock",
            "layer_idx": 0,
            "subblock_kind": "ffn",
            "subblock_name": "ffn",
            "loss": loss,
            "candidate_id": candidate,
            "active_params": active,
            "teacher_params": 100,
            "parameter_ratio": 0.01,
            "hidden_width": 4096,
            "unused_large_payload": {"weights": list(range(32))},
        }
        for lane, loss, candidate, active in (
            (0, 0.2, "candidate-a:ffn:ffn", 50),
            (1, 0.4, "candidate-b:ffn:ffn", 75),
        )
    ]
    (bypass / "dp_observations.jsonl").write_text(
        json.dumps({"step": 1, "observations": observations}) + "\n"
    )
    _write(
        bypass / "candidate_catalog.json",
        {
            "candidate-a:ffn:ffn": {"kind": "ffn", "intermediate_size": 8},
            "candidate-b:ffn:ffn": {"kind": "ffn", "intermediate_size": 12},
            "unused:ffn:ffn": {"kind": "ffn", "intermediate_size": 16},
        },
    )

    nested = _nested_bypass_data(tmp_path)
    assert set(nested["candidate_catalog"]) == {
        "candidate-a:ffn:ffn",
        "candidate-b:ffn:ffn",
    }
    assert [row["parameter_ratio"] for row in nested["observations"]] == [0.5, 0.75]
    assert all("unused_large_payload" not in row for row in nested["observations"])
    assert nested["observation_diagnostics"] == {
        "input_count": 2,
        "emitted_count": 2,
        "invalid_parameter_counts": 0,
        "missing_catalog_entries": 0,
    }

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'for="nested-bypass-unit-select">Sublayer</label>' in document
    assert 'for="nested-bypass-width-select">Hidden width</label>' in document
    assert 'id="nested-bypass-axis-filters"' in document
    assert 'id="nested-bypass-config-summary"' in document
    assert document.count("id='nested-bypass-unit-plot'") == 1
    assert '"parameter_ratio": 0.5' in document
    assert '"parameter_ratio": 0.75' in document
    assert '"unused:ffn:ffn"' not in document
    assert "unused_large_payload" not in document
    assert "cmin:0,cmax:1" in document
    assert "Active / teacher params" in document
    assert "DP lane" in document
    assert "candidate_catalog" in document
    assert "type:'scattergl'" not in document
    assert "type:'scatter'" in document
    assert "charts.installConfigFocus(element,configKeys)" in document
    assert "charts.stableConfigKey" in document
    assert "pointsByUnit" in document
    assert "populateWidthSelector" in document
    assert "populateAxisFilters" in document
    assert "formatConfig" in document
    assert "selected configurations" in document
    assert 'id="nested-bypass-ema-alpha"' in document
    assert "EMA coefficient" in document
    assert "emaByStep" in document
    assert "catalog[point.candidate_id]" in document
    assert "JSON.stringify([point.hidden_width,point.candidate_id])" not in document


def test_nested_bypass_attention_filters_use_query_heads_per_kv_head(tmp_path: Path):
    bypass = tmp_path / "artifacts/bypass"
    _write(
        bypass / "local_kd_loss_history.json",
        {
            "max_steps": 1,
            "dp_observation_path": str(bypass / "dp_observations.jsonl"),
            "candidate_catalog_path": str(bypass / "candidate_catalog.json"),
            "records": [{"step": 1, "loss": 0.3}],
        },
    )
    observation = {
        "step": 1,
        "dp_lane": 0,
        "granularity": "subblock",
        "layer_idx": 0,
        "subblock_kind": "attention",
        "subblock_name": "attention",
        "loss": 0.3,
        "candidate_id": "attention-a",
        "active_params": 50,
        "teacher_params": 100,
    }
    (bypass / "dp_observations.jsonl").write_text(
        json.dumps({"step": 1, "observations": [observation]}) + "\n"
    )
    _write(
        bypass / "candidate_catalog.json",
        {
            "attention-a": {
                "kind": "attention",
                "name": "attention",
                "num_kv_heads": 2,
                "num_query_heads": 24,
                "qk_head_dim": 128,
            }
        },
    )

    data = _nested_bypass_data(tmp_path)
    config = data["candidate_catalog"]["attention-a"]

    assert config["num_kv_heads"] == 2
    assert config["num_query_heads_per_kv_head"] == 12
    assert "num_query_heads" not in config


def test_nested_bypass_report_surfaces_invalid_parameter_metadata(tmp_path: Path):
    bypass = tmp_path / "artifacts/bypass"
    _write(
        tmp_path / "manifests/bypass.json",
        {"config": {"bypass": {"enabled": True, "granularity": "subblock"}}},
    )
    _write(
        bypass / "local_kd_loss_history.json",
        {
            "max_steps": 1,
            "dp_observation_path": str(bypass / "dp_observations.jsonl"),
            "candidate_catalog_path": str(bypass / "candidate_catalog.json"),
            "records": [{"step": 1, "loss": 0.3}],
        },
    )
    (bypass / "dp_observations.jsonl").write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "step": 1,
                        "dp_lane": 0,
                        "granularity": "subblock",
                        "layer_idx": 0,
                        "subblock_kind": "ffn",
                        "subblock_name": "ffn",
                        "loss": 0.2,
                        "candidate_id": "missing:ffn:ffn",
                        "active_params": 50,
                        "teacher_params": 0,
                        "parameter_ratio": 123.0,
                        "hidden_width": 4096,
                    }
                ]
            }
        )
        + "\n"
    )
    _write(bypass / "candidate_catalog.json", {})

    nested = _nested_bypass_data(tmp_path)
    assert nested["observations"][0]["parameter_ratio"] is None
    assert nested["observation_diagnostics"]["invalid_parameter_counts"] == 1
    assert nested["observation_diagnostics"]["missing_catalog_entries"] == 1

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")
    assert "1 observation has missing or invalid parameter counts" in document
    assert "1 observation references a candidate missing from the catalog" in document


def test_nested_bypass_report_resolves_campaign_relative_artifact_paths(tmp_path: Path):
    root = tmp_path / "puzzle_runs" / "campaign"
    bypass = root / "artifacts" / "bypass"
    _write(
        bypass / "local_kd_loss_history.json",
        {
            "max_steps": 1,
            "dp_observation_path": "puzzle_runs/campaign/artifacts/bypass/dp_observations.jsonl",
            "candidate_catalog_path": "puzzle_runs/campaign/artifacts/bypass/candidate_catalog.json",
            "records": [{"step": 1, "loss": 0.3}],
        },
    )
    (bypass / "dp_observations.jsonl").write_text(
        json.dumps(
            {
                "observations": [
                    {
                        "step": 1,
                        "dp_lane": 0,
                        "granularity": "subblock",
                        "layer_idx": 0,
                        "subblock_kind": "ffn",
                        "subblock_name": "ffn",
                        "loss": 0.2,
                        "candidate_id": "candidate:ffn:ffn",
                        "active_params": 50,
                        "teacher_params": 100,
                        "hidden_width": 4096,
                    }
                ]
            }
        )
        + "\n"
    )
    _write(
        bypass / "candidate_catalog.json",
        {"candidate:ffn:ffn": {"kind": "ffn", "intermediate_size": 8}},
    )

    nested = _nested_bypass_data(root)
    assert len(nested["observations"]) == 1
    assert set(nested["candidate_catalog"]) == {"candidate:ffn:ffn"}
    assert Path(nested["observation_path"]) == bypass / "dp_observations.jsonl"
    assert Path(nested["candidate_catalog_path"]) == bypass / "candidate_catalog.json"
