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

import pytest

from modelopt.torch.puzzletron.artifact_coverage import (
    verify_campaign_artifacts,
    verify_real_campaign_artifacts,
)


def _complete(**overrides):
    values = {
        "expected_scores": {"s0", "s1"},
        "present_scores": {"s0", "s1"},
        "expected_runtimes": {"r0"},
        "present_runtimes": {"r0"},
        "expected_depths": {0, 1},
        "present_depths": {0, 1},
        "expected_identity": {"checkpoint": "teacher", "data": "valid"},
        "observed_identity": {"checkpoint": "teacher", "data": "valid"},
        "bypass_enabled": False,
    }
    values.update(overrides)
    return verify_campaign_artifacts(**values)


def test_complete_campaign_reports_exact_coverage():
    report = _complete()

    assert report.complete
    assert report.rows["scores"].missing == ()
    assert report.rows["runtimes"].extra == ()
    report.require_complete()


def test_missing_score_components_are_a_hard_error():
    report = _complete(present_scores={"s0"})

    assert not report.complete
    assert report.rows["scores"].missing == ("s1",)
    with pytest.raises(RuntimeError, match="scores.*s1"):
        report.require_complete()


def test_identity_mismatch_is_a_hard_error():
    report = _complete(observed_identity={"checkpoint": "other", "data": "valid"})

    with pytest.raises(RuntimeError, match="checkpoint"):
        report.require_complete()


def test_bypass_enabled_is_rejected_for_no_bypass_campaign():
    report = _complete(bypass_enabled=True)

    with pytest.raises(RuntimeError, match="bypass"):
        report.require_complete()


def test_real_campaign_gate_uses_atomic_results_shard_markers_and_depth_scenarios(tmp_path):
    import json

    (tmp_path / "single_subblock_replacement_solutions--validation").mkdir()
    for index in range(2):
        (
            tmp_path
            / "single_subblock_replacement_solutions--validation"
            / f"solution_{index}.json"
        ).write_text("{}")
    (tmp_path / "runtime_cache/shards/cache").mkdir(parents=True)
    for index in range(2):
        (tmp_path / "runtime_cache/shards/cache" / f"shard_{index:04d}.json").write_text("{}")
        (tmp_path / "runtime_cache/shards/cache" / f"shard_{index:04d}.done").touch()
    (tmp_path / "depth/iterative").mkdir(parents=True)
    (tmp_path / "depth/iterative/trajectory.json").write_text(
        json.dumps(
            {"scenarios": [{"removals": []}, {"removals": [{"layer_idx": 0, "kind": "ffn"}]}]}
        )
    )
    (tmp_path / "subblock_replacement_manifest.json").write_text(
        json.dumps({"subblock_solution_count": 2})
    )

    report = verify_real_campaign_artifacts(
        tmp_path,
        expected_depth_scenarios=2,
        bypass_enabled=False,
    )

    assert report.complete
    report.require_complete()


def test_real_campaign_gate_rejects_scoring_checkpoint_identity_mismatch(tmp_path):
    import json

    (tmp_path / "single_subblock_replacement_solutions--validation").mkdir()
    (tmp_path / "single_subblock_replacement_solutions--validation/solution_0.json").write_text(
        "{}"
    )
    (tmp_path / "runtime_cache/shards/cache").mkdir(parents=True)
    (tmp_path / "runtime_cache/shards/cache/shard_0000.json").write_text("{}")
    (tmp_path / "runtime_cache/shards/cache/shard_0000.done").touch()
    (tmp_path / "depth/iterative").mkdir(parents=True)
    (tmp_path / "depth/iterative/trajectory.json").write_text(json.dumps({"scenarios": [{}]}))
    (tmp_path / "subblock_replacement_manifest.json").write_text(
        json.dumps({"subblock_solution_count": 1, "full_search_space_preserved": True})
    )
    (tmp_path / "subblock_distributed_eval/campaign").mkdir(parents=True)
    (tmp_path / "subblock_distributed_eval/campaign/manifest.json").write_text(
        json.dumps(
            {
                "model": {"checkpoint_dir": "/observed/teacher"},
                "data": {"scoring": {"eval_samples": 128, "block_size": 16384}},
            }
        )
    )

    report = verify_real_campaign_artifacts(
        tmp_path,
        expected_depth_scenarios=1,
        bypass_enabled=False,
        expected_checkpoint_dir="/expected/teacher",
        expected_data_identity={"eval_samples": 128, "block_size": 16384},
    )

    with pytest.raises(RuntimeError, match="checkpoint_dir"):
        report.require_complete()
