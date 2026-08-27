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

from modelopt.torch.puzzletron.stages.diagnostics import (
    _diagnostic_checkpoint_needs_rebuild,
    _hidden_width_ranking_verdict,
    _parent_sweep_sanity_verdict,
    _ratio_aligned_hidden_widths,
    _select_layers,
    _write_hidden_only_diagnostic_artifacts,
)


def test_hidden_width_targets_apply_requested_ratios_and_alignment():
    assert _ratio_aligned_hidden_widths(4096, [0.875, 0.25], alignment=256) == [
        3584,
        1024,
    ]


def test_hidden_width_verdict_can_treat_original_prefix_as_diagnostic_only():
    values = {"activation": 38.0, "random": 31.0, "reverse": 95.0}

    diagnostic = _hidden_width_ranking_verdict(
        values,
        tolerance=0.0,
        require_beats_random=False,
    )
    strict = _hidden_width_ranking_verdict(
        values,
        tolerance=0.0,
        require_beats_random=True,
    )

    assert diagnostic == {
        "passed": True,
        "beats_random": False,
        "beats_reverse": True,
        "require_beats_random": False,
    }
    assert strict["passed"] is False


def test_hidden_only_diagnostic_finishes_without_empty_parent_sweep(tmp_path):
    artifacts = tmp_path / "artifacts"
    temporary = tmp_path / "diagnostics" / "temporary"
    temporary.mkdir(parents=True)
    (temporary / "reverse.bin").write_bytes(b"temporary")
    hidden = {
        "hidden_width": 1344,
        "teacher_hidden_width": 2688,
        "primary_metric": "raw_replacement_loss",
        "passed": True,
        "beats_random": True,
        "beats_reverse": True,
        "rows": [],
    }

    summary = _write_hidden_only_diagnostic_artifacts(
        artifacts_dir=artifacts,
        temporary_root=temporary,
        hidden_width_summary=hidden,
        cleanup_reverse=True,
    )

    assert summary["status"] == "complete"
    assert summary["axes"] == ["hidden_width"]
    assert summary["parent_sweep"] == {"status": "not_applicable"}
    assert not temporary.exists()
    assert (
        json.loads((artifacts / "activation_diagnostic_summary.json").read_text())["hidden_width"][
            "passed"
        ]
        is True
    )
    assert (artifacts / "activation_diagnostic_table.md").is_file()
    assert (artifacts / "activation_diagnostic_scores.csv").is_file()
    assert (
        json.loads((artifacts / "diagnostic_cleanup.json").read_text())[
            "reverse_checkpoint_removed"
        ]
        is True
    )


def test_diagnostic_retry_rebuilds_partial_indexed_checkpoint(tmp_path):
    checkpoint = tmp_path / "reverse"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "sorted_permutations.json").write_text('{"x": [0]}')
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "a": "model-00001-of-00002.safetensors",
                    "b": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"first")

    assert _diagnostic_checkpoint_needs_rebuild(checkpoint) is True

    (checkpoint / "model-00002-of-00002.safetensors").write_bytes(b"second")
    assert _diagnostic_checkpoint_needs_rebuild(checkpoint) is False


def test_random_diagnostic_layers_are_deterministic_and_axis_specific():
    eligible = list(range(24))

    ffn = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="ffn_intermediate",
    )
    repeated = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="ffn_intermediate",
    )
    gdn = _select_layers(
        eligible,
        3,
        selection="random",
        seed=1234,
        axis="gdn_key_groups",
    )

    assert len(ffn) == 3
    assert ffn == repeated
    assert ffn != gdn
    assert ffn == sorted(ffn)


def test_parent_sweep_sort_miss_is_blocking_but_width_miss_remains_advisory():
    verdict = _parent_sweep_sanity_verdict(
        {
            "passed": False,
            "findings": [{"stage": "width_sanity", "message": "ranking regressed"}],
        },
        {
            "passed": False,
            "findings": [{"stage": "width_sanity", "message": "teacher drifted"}],
        },
    )

    assert verdict.passed is False
    assert verdict.blocking is True
    assert verdict.findings == [
        {"stage": "width_sanity", "message": "ranking regressed"},
        {"stage": "sort_sanity", "message": "teacher drifted", "severity": "error"},
    ]
