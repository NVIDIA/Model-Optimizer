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

"""CPU contracts for the Llama 3.1 8B legacy campaign port."""

from pathlib import Path

import pytest
import yaml

from puzzletron_orchestrator.config import load_experiment_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODEL_PATH = REPOSITORY_ROOT / "examples/puzzletron/configs/families/llama/llama3p1_8b/model.yaml"
CAMPAIGN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/llama/llama3p1_8b/runs/pruneffn_memory.yaml"
)


def test_llama3p1_8b_model_and_legacy_ffn_domain_are_pinned() -> None:
    model = yaml.safe_load(MODEL_PATH.read_text())

    assert model["input_hf_model_path"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert model["model_info"]["hf_revision"] == "0e9e39f249a16976918f6564b8830bc894c89659"
    assert model["pruning"]["intermediate_size_list"] == [3072, 5888, 8704, 11520]
    assert model["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 14336,
            "values": [3072, 5888, 8704, 11520],
        }
    }


def test_legacy_memory_campaign_keeps_noops_and_both_hard_bounds() -> None:
    campaign = yaml.safe_load(CAMPAIGN_PATH.read_text())

    assert [entry["axis_ids"] for entry in campaign["pruning"]["activation_passes"]] == [
        ["ffn_intermediate"]
    ]
    assert campaign["pruning"]["eval_samples"] == 1000
    assert campaign["pruning"]["micro_batch_size"] == 4
    assert campaign["build_library"]["include_noops"] is True
    assert campaign["search_space"]["no_op"] == {
        "subblocks": ["attention", "ffn"],
        "whole_block": False,
        "cartesian": False,
    }
    assert campaign["depth_importance"]["enabled"] is False
    assert campaign["embedding_pruning"]["widths"] == [4096]

    assert campaign["vllm_stats"]["runtime_stats"]["enabled"] is False
    assert campaign["vllm_stats"]["batch_sizes"] == [64, 96, 128]
    assert campaign["vllm_stats"]["prefill_seq_len"] == 4096
    assert campaign["vllm_stats"]["generation_seq_len"] == 4096

    run = campaign["mip"]["runs"]["pruneffn-memory"]
    assert campaign["mip"]["subblock_stats_args"] == {"batch_sizes": [96]}
    assert run["constraints"] == {
        "memory": {"at": {"legacy-batch96": {"max": 78000}}},
        "params": {"max": 7000000000},
    }
    assert run["search_space"] == {
        "depth": [0],
        "embedding": [4096],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }
    assert run["solver"] == {
        "backend": "auto",
        "num_solutions": 1,
        "min_hamming_distance": 1,
        "max_seconds_per_solution": 60,
    }


def test_legacy_realization_flow_materializes_then_validates_one_candidate() -> None:
    campaign = yaml.safe_load(CAMPAIGN_PATH.read_text())
    flow = campaign["post_mip"]["flows"]["legacy-pruneffn-memory"]

    assert flow["source"]["run"] == "pruneffn-memory"
    assert list(flow["nodes"]) == ["best_mip", "materialized", "final_eval", "best"]
    assert flow["nodes"]["best_mip"]["top_k"] == 1
    assert flow["nodes"]["materialized"]["input"] == "best_mip"
    assert flow["nodes"]["final_eval"]["input"] == "materialized"
    assert flow["nodes"]["final_eval"]["config"]["eval_samples"] == 128
    assert flow["nodes"]["final_eval"]["config"]["val_dataset_name"] == "valid"


@pytest.mark.parametrize("smoke", [False, True])
def test_corrected_sweep_preserves_joint_scoring_and_all_capped_targets(
    tmp_path, monkeypatch, smoke
) -> None:
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.delenv("PUZZLETRON_RUN_ROOT", raising=False)
    filename = (
        "pruneffn_memory_joint_sweep_smoke.yaml" if smoke else "pruneffn_memory_joint_sweep.yaml"
    )
    config = load_experiment_config(CAMPAIGN_PATH.with_name(filename))

    assert config["search_space"]["no_op"] == {
        "subblocks": ["attention", "ffn"],
        "whole_block": True,
        "cartesian": True,
    }
    assert config["replacement_scoring"]["granularity"] == "block"
    mip = config["mip"]
    assert mip["score_granularity"] == "block"
    assert (
        mip["single_block_replacement_validation_dir"]
        == config["replacement_scoring"]["block_output_dir"]
    )
    assert mip["canonical_solutions_path"] == config["replacement_scoring"]["block_solutions_path"]
    run = mip["runs"]["pruneffn-memory"]
    assert run["constraints"]["params"] == {"max": 7000000000}
    assert list(run["variants"]) == [f"memory-0{target}" for target in (50, 60, 70, 80, 90)]
    for target, variant in zip((50, 60, 70, 80, 90), run["variants"].values()):
        assert variant["constraints"] == {
            "memory": {"at": {"legacy-batch96": {"max": f"{target}%"}}}
        }
    nodes = config["post_mip"]["flows"]["legacy-pruneffn-memory"]["nodes"]
    assert nodes["best_mip"]["top_k"] == 5
    assert nodes["final_eval"]["config"]["eval_samples"] == (2 if smoke else 128)
    assert config["pruning"]["eval_samples"] == (8 if smoke else 1000)
    assert config["replacement_scoring"]["eval_samples"] == (2 if smoke else 8)
    assert config["depth_importance"]["enabled"] is False
