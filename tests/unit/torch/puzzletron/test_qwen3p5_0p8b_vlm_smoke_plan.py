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

"""CPU plan contracts for the Qwen 3.5 0.8B VLM MIP smoke."""

from pathlib import Path

from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUN_PATH = (
    REPOSITORY_ROOT
    / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/mip_vlm_smoke.yaml"
)
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/qwen3p5_0p8b"
RUNNER_PATH = ORCHESTRATION_ROOT / "runner.slurm.yaml"
EXECUTION_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/execution.single_gpu.yaml"
)


def _compile_plan(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("PUZZLETRON_DATASET_PATH", str(tmp_path / "dataset"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    return compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )


def test_qwen3p5_0p8b_vlm_smoke_compiles_the_bounded_ffn_only_route(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = _compile_plan(monkeypatch, tmp_path)
    config = plan.experiment_config
    stage_ids = tuple(stage.stage_id for stage in plan.stages)

    assert stage_ids == (
        "convert",
        "width_importance",
        "sort",
        "sort_sanity",
        "width_sanity",
        "slicing_sanity",
        "build_library",
        "replacement_scoring",
        "mip",
    )
    assert all(stage.total_gpus == 1 for stage in plan.stages)
    assert config["model"]["source"] == "Qwen/Qwen3.5-0.8B"
    assert config["model"]["revision"] == "2fc06364715b967f1860aea9cf38778875588b17"
    assert config["model"]["descriptor_override"] == "qwen3_5"
    assert config["model"]["force_hf"] is False
    assert config["model"]["trust_remote_code"] is False
    assert config["data"]["path"] == str(tmp_path / "dataset")
    assert config["data"]["revision"] == "fixture-revision"
    assert config["data"]["processor_identity"] == (
        "Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17"
    )
    assert config["data"]["modality"] == "multimodal"
    assert config["data"]["layout"] == "padded_varlen"
    assert config["tokenize_data"]["enabled"] is False
    assert config["tokenize_data"]["caches"] == []
    assert config["sort"]["deferred_axes"] == [
        "kv_groups",
        "q_heads_per_group",
        "gdn_key_groups",
        "gdn_value_heads_per_group",
        "gdn_key_head_dim",
        "gdn_value_head_dim",
    ]
    assert config["sort_sanity"]["automodel"]["lm_head_backend"] == "streaming"
    assert config["sort_sanity"]["max_abs_lm_loss_delta"] == 0.002
    assert config["sort_sanity"]["max_abs_reverse_lm_loss_delta"] == 0.002
    assert config["replacement_scoring"]["automodel"]["lm_head_backend"] == "streaming"
    for section in (
        "pruning",
        "sort_sanity",
        "width_sanity",
        "depth_importance",
        "replacement_scoring",
    ):
        assert config[section]["packed_token_cache_path"] is None
    assert config["search_space"]["axes"] == {
        "ffn_intermediate": {
            "enabled": True,
            "teacher_value": 3584,
            "values": [3072, 2048],
        }
    }
    assert config["width_sanity"]["axes"] == ["ffn_intermediate"]
    assert config["mip"]["runs"]["params-90"]["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "all"},
    }
