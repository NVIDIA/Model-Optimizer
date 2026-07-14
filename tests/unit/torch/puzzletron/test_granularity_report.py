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


def test_report_renders_granularity_coverage_and_subblock_owned_axes(tmp_path: Path):
    _write(
        tmp_path / "manifests/scoring.json",
        {
            "status": "success",
            "config": {
                "depth": {"granularity": "subblock"},
                "calc_subblock_stats": {"runtime_stats": {"granularity": "block"}},
                "scoring": {"granularity": "subblock"},
                "bypass": {"enabled": False, "granularity": "block"},
            },
        },
    )
    _write(
        tmp_path / "subblock_replacement_manifest.json",
        {
            "mode": "replace_one_subblock",
            "canonical_entry_count": 25600,
            "subblock_solution_count": 3,
            "full_search_space_preserved": True,
        },
    )
    result_dir = tmp_path / "single_subblock_replacement_solutions--validation"
    _write(result_dir / "solution_0.json", _result(0, "ffn", "ffn", 8))
    _write(result_dir / "solution_1.json", _result(1, "attention", "mixer", 4))

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert "Granularity and artifact coverage" in document
    assert "Depth</th><td><span class='granularity-badge'>subblock" in document
    assert "vLLM runtime</th><td><span class='granularity-badge'>block" in document
    assert "Replacement scoring</th><td><span class='granularity-badge'>subblock" in document
    assert "Bypass</th><td><span class='granularity-badge'>block (disabled)" in document
    assert "25,600" in document
    assert "2 / 3" in document
    assert "additive subblock deltas" in document
    assert "Replace-one-subblock scoring" in document
    assert '"subblock_kind": "ffn"' in document
    assert '"subblock_name": "mixer"' in document


def test_nested_bypass_report_selects_exact_subblock_losses(tmp_path: Path):
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
    assert 'for="nested-bypass-config-select">Layer configuration</label>' in document
    assert '<select id="nested-bypass-config-select">' in document
    assert document.count("id='nested-bypass-unit-plot'") == 1
    assert "layer_0:mamba:linear_attn" in document
    assert "layer_0:ffn:feed_forward" in document
    assert "per_subblock_loss" in document


def test_nested_bypass_report_uses_dp_scatter_and_normalized_parameter_color(tmp_path: Path):
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
            "parameter_ratio": active / 100,
            "hidden_width": 4096,
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
        },
    )

    result = generate_campaign_progress_report(tmp_path)
    document = Path(result["html"]).read_text(encoding="utf-8")

    assert 'for="nested-bypass-unit-select">Sublayer</label>' in document
    assert document.count("id='nested-bypass-unit-plot'") == 1
    assert '"parameter_ratio": 0.5' in document
    assert '"parameter_ratio": 0.75' in document
    assert "cmin:0,cmax:1" in document
    assert "Active / teacher params" in document
    assert "DP lane" in document
    assert "candidate_catalog" in document
    assert "type:'scattergl'" not in document
    assert "type:'scatter'" in document
    assert "charts.installConfigFocus(element,configKeys)" in document
    assert "charts.stableConfigKey" in document
    assert "populateConfigSelector" in document
    assert "configSelect.value==='ALL'" in document
    assert "catalog[point.candidate_id]" in document
    assert "JSON.stringify([point.hidden_width,point.candidate_id])" not in document
