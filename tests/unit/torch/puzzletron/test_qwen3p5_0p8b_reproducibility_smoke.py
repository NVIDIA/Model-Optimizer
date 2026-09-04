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

"""Dependency-light contract for the Qwen3.5 0.8B reproducibility smoke."""

import json
from pathlib import Path
from typing import TYPE_CHECKING

from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_orchestrator.controller import CampaignController
from puzzletron_orchestrator.executors.base import Executor
from puzzletron_orchestrator.identity import stable_hash
from puzzletron_orchestrator.schema import AttemptSpec, JobHandle, JobState, JobStatus
from puzzletron_orchestrator.stages import semantic_stage_config

if TYPE_CHECKING:
    from collections.abc import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
RUN_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/families/qwen3_5/qwen3p5_0p8b/runs/"
    "vlm_reproducibility_smoke.yaml"
)
RUNNER_PATH = (
    REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration/runner.slurm.example.yaml"
)
ORCHESTRATION_ROOT = REPOSITORY_ROOT / "examples/puzzletron/configs/orchestration"
EXECUTION_PATH = ORCHESTRATION_ROOT / "qwen3p5_0p8b/execution.vlm_reproducibility_smoke.yaml"
EXPECTED_ROOT = REPOSITORY_ROOT / "examples/puzzletron/expected"


class _ReportOnlyExecutor(Executor):
    backend = "report-only"

    def __init__(self) -> None:
        self.submitted_stage_ids: list[str] = []
        self.attempts: dict[str, AttemptSpec] = {}

    def submit(self, attempt: AttemptSpec) -> JobHandle:
        self.submitted_stage_ids.append(attempt.stage_id)
        handle = JobHandle(self.backend, attempt.attempt_id, attempt.attempt_id)
        self.attempts[handle.handle_id] = attempt
        return handle

    def poll(self, handles: "Sequence[JobHandle]") -> list[JobStatus]:
        for handle in handles:
            attempt = self.attempts[handle.handle_id]
            if attempt.stage_id == "final_report":
                puzzle_dir = Path(
                    attempt.command.argv[attempt.command.argv.index("--puzzle-dir") + 1]
                )
                report_dir = puzzle_dir / "artifacts/campaign_report"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "campaign_report.html").write_text("<html></html>\n")
                (report_dir / "report_manifest.json").write_text("{}\n")
        return [JobStatus(handle=handle, state=JobState.COMPLETED) for handle in handles]

    def cancel(self, handles: "Sequence[JobHandle]") -> None:
        del handles

    def recover(self, handle: JobHandle) -> JobStatus:
        return JobStatus(handle=handle, state=JobState.COMPLETED)


def test_reproducibility_smoke_is_bounded_unattended_and_comparison_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    plan = compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )
    config = plan.experiment_config
    mip = config["mip"]["runs"]["params-90"]
    nodes = config["post_mip"]["flows"]["reproducibility-smoke"]["nodes"]

    assert config["replacement_scoring"]["eval_samples"] == 2
    assert config["dataset_path"] == str(tmp_path / "run" / "datasets/nemotron_vlm_v2")
    assert config["prepare_dataset"] | {
        "evaluation_hf_home": None,
        "evaluation_datasets": None,
    } == {
        "enabled": True,
        "adapter": "nemotron_vlm_v2",
        "output": config["dataset_path"],
        "subsets": ["sparsetables", "plotqa_cot", "wiki_en"],
        "num_samples": 8,
        "seed": 42,
        "max_shards_per_subset": 1,
        "revision": "fixture-revision",
        "evaluation_tasks": ["realworldqa", "mmmu_val", "mvbench"],
        "evaluation_hf_home": None,
        "evaluation_datasets": None,
    }
    assert config["prepare_dataset"]["evaluation_hf_home"] == str(tmp_path / "hf-home")
    assert set(config["prepare_dataset"]["evaluation_datasets"]) == {
        "realworldqa",
        "mmmu_val",
        "video_mmmu",
        "mvbench",
        "mmvu_val",
        "videomme",
        "longvideobench_val_v",
        "mlvu_dev",
        "perceptiontest_val_mc",
    }
    prepare = plan.stages[0]
    convert = plan.stages[1]
    assert prepare.stage_id == "prepare_dataset"
    assert prepare.resource == "cpu"
    assert convert.stage_id == "convert"
    assert convert.parents == ("prepare_dataset",)
    stages = {stage.stage_id: stage for stage in plan.stages}
    cpu_stage_ids = {
        "prepare_dataset",
        "convert",
        "build_library",
        "mip",
        "post.reproducibility-smoke.retain_variants",
        "post.reproducibility-smoke.result",
    }
    for stage_id in cpu_stage_ids:
        assert stages[stage_id].resource == "cpu"
        assert stages[stage_id].total_gpus == 0
    assert all(
        stage.total_gpus == 1 for stage in plan.stages if stage.stage_id not in cpu_stage_ids
    )

    assert mip["constraints"] == {"params": {"min": "75%", "max": "100%"}}
    assert mip["search_space"] == {
        "depth": [0],
        "embedding": [1024],
        "axes_default": "teacher",
        "axes": {"ffn.intermediate_size": "teacher"},
    }
    assert mip["variants"]["query-heads-only"]["search_space"]["axes"] == {
        "attention.num_kv_heads": {"default": "teacher", "layers": {19: [2]}},
        "attention.q_per_group": {"default": "teacher", "layers": {19: [3]}},
    }
    assert mip["variants"]["kv-groups-only"]["search_space"]["axes"] == {
        "attention.num_kv_heads": {"default": "teacher", "layers": {19: [1]}},
        "attention.q_per_group": {"default": "teacher", "layers": {19: [4]}},
    }
    combined = mip["variants"]["all-reducible-axes"]["search_space"]
    assert combined["depth"] == [1]
    assert combined["embedding"] == [960]
    assert combined["axes"] == {
        "ffn.intermediate_size": {"default": "teacher", "layers": {1: [3328]}},
        "attention.num_kv_heads": {"default": "teacher", "layers": {19: [1]}},
        "attention.q_per_group": {"default": "teacher", "layers": {19: [3]}},
        "mamba.num_groups": {"default": "teacher", "layers": {0: [14]}},
        "mamba.num_heads": {"default": "teacher", "layers": {0: [14]}},
        "mamba.state_dim": {"default": "teacher", "layers": {0: [112]}},
        "mamba.head_dim": {"default": "teacher", "layers": {0: [112]}},
    }
    assert not any(node["type"] == "manual_filter" for node in nodes.values())
    assert config["post_mip"]["flows"]["reproducibility-smoke"]["source"] == {
        "run": "params-90",
        "variants": "all",
        "objectives": "all",
    }
    assert nodes["retain_variants"] == {
        "type": "filter",
        "input": "online_eval",
        "mode": "top_k",
        "metric": "online_eval.lm_loss",
        "direction": "minimize",
        "top_k": 3,
    }
    assert nodes["materialized"]["input"] == "retain_variants"
    assert nodes["pre_kd_smoke"]["input"] == "materialized"
    assert nodes["pre_kd_smoke"]["config"]["profile"] == "qwen35_vlm_core3_24row_smoke_v2"
    assert nodes["result"]["config"]["row_manifest"] == ("profile:core-3_24-examples_r1-vllm")
    assert nodes["kd_2"]["input"] == "pre_kd_smoke"
    assert nodes["kd_2"]["config"]["resume"] is True
    assert nodes["kd_2"]["exposure"]["cumulative_examples"] == 2
    assert nodes["post_kd_smoke"]["input"] == "kd_2"
    assert nodes["result"]["input"] == "post_kd_smoke"
    assert (
        nodes["serving_mechanics"]["config"]
        | {
            "request_count": 3,
            "warmup_request_count": 1,
            "repetitions": 1,
            "concurrency": [1],
            "image_batch_sizes": [1],
        }
        == nodes["serving_mechanics"]["config"]
    )
    assert tuple(stage.stage_id for stage in plan.stages)[-1] == (
        "post.reproducibility-smoke.serving_mechanics"
    )


def test_complete_reproducibility_smoke_resume_submits_no_compatible_stage(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    plan = compile_campaign_plan(
        experiment_config_path=RUN_PATH,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(EXECUTION_PATH),
        stage_filter="full",
    )
    first_executor = _ReportOnlyExecutor()
    controller = CampaignController(plan, executor=first_executor, poll_interval_seconds=0.01)
    original_stage_execution_identity = CampaignController._stage_execution_identity

    def stage_execution_identity(self, node, work_plan=None):
        if not node.stage_id.startswith("post."):
            return original_stage_execution_identity(self, node, work_plan)
        work_plan = work_plan or adapter_for_stage(node).plan(self.plan, node)
        return stable_hash(
            {
                "stage": node.stage_id,
                "semantic_config": semantic_stage_config(
                    self.plan.experiment_config, node.stage_id
                ),
                "compiled_node": self._compiled_nodes[node.stage_id],
                "work_items": [
                    {
                        "work_id": item.work_id,
                        "shard_index": item.shard_index,
                        "shard_count": item.shard_count,
                        "gpus_per_instance": item.gpus_per_instance,
                    }
                    for item in work_plan.items
                ],
            },
            prefix=f"{node.stage_id}_test_execution",
        )

    monkeypatch.setattr(CampaignController, "_stage_execution_identity", stage_execution_identity)
    for node in plan.stages:
        adapter = adapter_for_stage(node)
        work_plan = adapter.plan(plan, node)
        for index, item in enumerate(work_plan.items):
            attempt = controller._bind_attempt_to_stage_execution(
                node,
                work_plan,
                adapter.command(
                    plan=plan,
                    node=node,
                    item=item,
                    attempt_id=f"completed-{index}",
                    runner=plan.runner,
                ),
            )
            controller.store.save_attempt(attempt, None, JobState.COMPLETED.value)

    completed_stage_ids = set()
    finalized_stage_ids = []

    def completion_for_finalized_stage(config, stage_id):
        del config
        return stage_id in completed_stage_ids

    def finalize_completed_attempts(_self, node):
        finalized_stage_ids.append(node.stage_id)
        completed_stage_ids.add(node.stage_id)
        return True

    monkeypatch.setattr(
        "puzzletron_orchestrator.controller.stage_is_complete",
        completion_for_finalized_stage,
    )
    monkeypatch.setattr(CampaignController, "_finalize_stage", finalize_completed_attempts)

    first = controller.run()
    second_executor = _ReportOnlyExecutor()
    second = CampaignController(plan, executor=second_executor, poll_interval_seconds=0.01).run()

    assert len(plan.stages) > 1
    assert set(finalized_stage_ids) == completed_stage_ids
    assert first["report_status"] == second["report_status"] == "completed"
    assert first_executor.submitted_stage_ids == ["final_report"]
    assert second_executor.submitted_stage_ids == []


def _compile(monkeypatch, tmp_path: Path, run_path: Path, execution_path: Path):
    monkeypatch.setenv("PUZZLETRON_RUN_ROOT", str(tmp_path / run_path.stem))
    monkeypatch.setenv("PUZZLETRON_DATASET_REVISION", "fixture-revision")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))
    return compile_campaign_plan(
        experiment_config_path=run_path,
        runner=load_runner_config(RUNNER_PATH),
        execution=load_execution_config(execution_path),
        stage_filter="full",
    )


def test_qwen3p5_smoke_expectation_has_qualified_stable_runtime_values(
    monkeypatch, tmp_path: Path
) -> None:
    smoke = _compile(monkeypatch, tmp_path, RUN_PATH, EXECUTION_PATH)
    smoke_result = smoke.experiment_config["post_mip"]["flows"]["reproducibility-smoke"]["nodes"][
        "result"
    ]["config"]
    contract = json.loads((EXPECTED_ROOT / "qwen3p5_0p8b_vlm_smoke_v1.json").read_text())
    observation = json.loads((EXPECTED_ROOT / contract["observation"]).read_text())
    required = {
        field["name"] for field in contract["fields"] if field["classification"] != "informational"
    }
    informational = {
        field["name"] for field in contract["fields"] if field["classification"] == "informational"
    }

    assert contract["schema"] == "modelopt.puzzletron-expected-results/v1"
    assert observation["schema"] == "modelopt.puzzletron-reference-observation/v1"
    assert observation["contract_id"] == contract["id"]
    assert required <= set(observation["values"])
    assert set(observation["values"]) - required <= informational
    exact_result_fields = {
        field["name"] for field in contract["fields"] if field["pointer"] == "/exact_result"
    }
    assert exact_result_fields
    assert all(observation["values"][name] is None for name in exact_result_fields)
    field_classifications = {field["name"]: field["classification"] for field in contract["fields"]}
    assert all(field_classifications[name] == "informational" for name in exact_result_fields)
    assert observation["qualification"]["status"] == "qualified-stable-runtime-values"
    stable_exact_suffixes = {
        "architecture_id",
        "axis_inventory",
        "denominators",
        "evaluator_backend_limitations",
        "evaluator_generation_policy",
        "evaluator_lmms_eval_revision",
        "evaluator_output_budget_contract",
        "evaluator_profile",
        "hidden_size",
        "kd_cumulative_examples",
        "kd_cumulative_steps",
        "kd_effective_tokens",
        "kd_global_batch_size",
        "kd_max_sample_length",
        "kd_token_upper_bound",
        "num_hidden_layers",
        "parameter_counts",
        "pre_kd_checkpoint_fingerprint",
        "reference_checkpoint_fingerprint",
        "row_outcomes",
        "selected_sample_ids",
        "stage_completion",
        "tensor_count",
    }
    for index in range(3):
        assert field_classifications[f"curve_{index}_evaluator_contract"] == "informational"
        assert observation["values"][f"curve_{index}_profile"] == smoke_result["profile"]
        assert (
            observation["values"][f"curve_{index}_manifest"] == smoke_result["row_manifest_sha256"]
        )
        assert observation["values"][f"curve_{index}_steps"] == 2
        stable_exact_fields = {f"curve_{index}_{suffix}" for suffix in stable_exact_suffixes}
        assert stable_exact_fields <= required
        assert all(field_classifications[name] == "exact" for name in stable_exact_fields)
        assert all(observation["values"][name] is not None for name in stable_exact_fields)
