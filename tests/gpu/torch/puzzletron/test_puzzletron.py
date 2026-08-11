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

"""Hermetic GPU coverage for the public Puzzletron campaign route."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
import torch
import yaml
from _test_utils.torch.transformers_models import create_tiny_qwen3_5_dir
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM

from modelopt.torch.puzzletron.orchestration.adapters.stage_compat import stage_is_complete
from modelopt.torch.puzzletron.pipeline_config import pipeline_config_from_path
from modelopt.torch.puzzletron.stages.graph import enabled_stage_ids
from puzzletron_orchestrator.adapters.registry import adapter_for_stage
from puzzletron_orchestrator.compiler import (
    compile_campaign_plan,
    load_execution_config,
    load_runner_config,
)
from puzzletron_setup.v2.wizard import _DEFAULT_DATA_SOURCE, _DEFAULT_MODEL_SOURCE, run_wizard_v2

if TYPE_CHECKING:
    from collections.abc import Sequence

    from puzzletron_setup.v2.prompts import PromptChoice


class _DefaultsBackend:
    """Select resolved defaults while supplying the test campaign directory."""

    def __init__(self, campaign_dir: Path) -> None:
        self.campaign_dir = campaign_dir

    def text(self, message: str, default: str) -> Any:
        if message == "Campaign directory:":
            return str(self.campaign_dir)
        return default

    def select(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        default: Any,
    ) -> Any:
        if message == "Model:":
            return _DEFAULT_MODEL_SOURCE
        if message == "Dataset:":
            return _DEFAULT_DATA_SOURCE
        if message == "Post Mip:":
            return "customize"
        if message.startswith("Post-MIP flow for "):
            return "none"
        if default is not None:
            return default
        return next(choice.value for choice in choices if choice.disabled is None)

    def checkbox(
        self,
        message: str,
        choices: Sequence[PromptChoice],
        defaults: Sequence[Any],
    ) -> Any:
        del message, choices
        return list(defaults)


def _save_messages_dataset(path: Path) -> None:
    response = (
        "Compression removes redundant parameters while preserving useful model behavior. " * 16
    ).strip()
    messages = [
        {"role": "user", "content": "What is model compression?"},
        {"role": "assistant", "content": response},
    ]
    rows = [{"messages": messages}] * 8
    DatasetDict(
        {
            "train": Dataset.from_list(rows),
            "validation": Dataset.from_list(rows),
        }
    ).save_to_disk(str(path))


def _artifact_digests(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}


def _nested_values(value: Any, key: str) -> list[Any]:
    values = []
    if isinstance(value, dict):
        values.extend(item for name, item in value.items() if name == key)
        for item in value.values():
            values.extend(_nested_values(item, key))
    elif isinstance(value, list):
        for item in value:
            values.extend(_nested_values(item, key))
    return values


def _tensor_values(value: Any):
    if torch.is_tensor(value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensor_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_values(item)


def _run_campaign(
    project_root_path: Path,
    smoke_bundle: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(project_root_path / "examples/puzzletron/orchestrate.py"),
            "--experiment",
            str(smoke_bundle / "experiment.yaml"),
            "--runner",
            str(smoke_bundle / "runner.yaml"),
            "--execution",
            str(smoke_bundle / "execution.yaml"),
            "--stage",
            "full",
            "--local",
            "--poll-interval",
            "0.05",
            "--color",
            "never",
            "--override",
            "tokenize_data.workers=1",
            "--override",
            "+replacement_scoring.automodel.lm_head_backend=streaming",
        ],
        cwd=project_root_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )


def _assert_campaign_succeeded(
    completed: subprocess.CompletedProcess[str],
    smoke_root: Path,
) -> None:
    if completed.returncode == 0:
        return
    logs = sorted(
        smoke_root.glob("logs/**/*.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    log_tail = logs[-1].read_text(errors="replace")[-12000:] if logs else "no task log found"
    pytest.fail(
        "Tiny Qwen Puzzletron campaign failed.\n"
        f"stdout tail:\n{completed.stdout[-12000:]}\n"
        f"stderr tail:\n{completed.stderr[-12000:]}\n"
        f"latest task-log tail:\n{log_tail}"
    )


@pytest.mark.timeout(1200)
def test_tiny_qwen_campaign_uses_current_public_route(
    project_root_path: Path,
    tmp_path: Path,
) -> None:
    """Run every required stage through the public orchestrator; requires one CUDA GPU."""

    assert torch.cuda.is_available(), "Puzzletron GPU CI requires CUDA"
    assert torch.cuda.device_count() == 1, "Puzzletron GPU CI requires one visible GPU"
    assert torch.equal(torch.arange(4, device="cuda").cpu(), torch.arange(4))
    model_dir = create_tiny_qwen3_5_dir(
        tmp_path / "model",
        with_tokenizer=True,
        hidden_size=512,
        intermediate_size=768,
        max_position_embeddings=128,
        num_hidden_layers=2,
        layer_types=["full_attention"] * 2,
    )
    dataset_dir = tmp_path / "dataset"
    campaign_dir = tmp_path / "campaign"
    result_root = tmp_path / "results"
    cache_dir = tmp_path / "cache"
    defaults_path = tmp_path / "defaults.yaml"
    _save_messages_dataset(dataset_dir)
    defaults_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "model": {
                    "source": str(model_dir),
                    "trust_remote_code": False,
                    "force_hf": False,
                },
                "data": {
                    "source": str(dataset_dir),
                    "modality": "text",
                    "layout": "fixed",
                    "sequence_length": 32,
                },
                "pruning": {
                    "depth_remove": 0,
                    "depth_importance_samples": 2,
                    "width_importance_samples": 2,
                    "replacement_samples": 2,
                    "sort_sanity": False,
                    "width_sanity": False,
                    "slicing_sanity": False,
                    "replacement_granularity": "block",
                    "axes": {
                        "hidden_width": {"values": [256]},
                        "kv_groups": {"values": [2]},
                        "q_heads_per_group": {"values": [2]},
                        "ffn_intermediate": {"values": [768, 512, 256]},
                        "gdn_key_groups": {"values": [2]},
                        "gdn_value_heads_per_group": {"values": [2]},
                        "gdn_key_head_dim": {"values": [8]},
                        "gdn_value_head_dim": {"values": [8]},
                    },
                    "bypass": {"enabled": False},
                },
                "vllm": {
                    "enabled": False,
                    "prefill_seq_len": 32,
                    "generation_seq_len": 8,
                    "batch_size": 1,
                    "max_num_seqs": 1,
                },
                "mip": {
                    "goal_metric": "params",
                    "goal_value": "90%",
                    "num_solutions": 1,
                },
                "stages": {
                    "width_importance": {"batch": 1},
                    "replacement_scoring": {"batch": 1, "instances": 1},
                },
                "output": {"result_root": str(result_root)},
                "infrastructure": {
                    "gpus_per_node": 1,
                    "execution_contract": {
                        "repository": str(project_root_path),
                        "venv": sys.prefix,
                        "container": None,
                        "container_mounts": None,
                        "prerun_commands": [],
                        "postrun_commands": [],
                    },
                },
            },
            sort_keys=False,
        )
    )

    generated = run_wizard_v2(
        resume=None,
        defaults_path=defaults_path,
        backend=_DefaultsBackend(campaign_dir),
    )
    smoke_bundle = generated / "smoke"
    for name in ("experiment.yaml", "runner.yaml", "execution.yaml"):
        assert (smoke_bundle / name).is_file()

    overrides = [
        "tokenize_data.workers=1",
        "+replacement_scoring.automodel.lm_head_backend=streaming",
    ]
    config = pipeline_config_from_path(smoke_bundle / "experiment.yaml", overrides=overrides)
    assert config["embedding_pruning"]["widths"] == [256]
    post_mip_flows = config["post_mip"]["flows"]
    assert len(post_mip_flows) == 1
    serving_config = next(iter(post_mip_flows.values()))["nodes"]["serving"]["config"]
    assert serving_config["input_tokens"] == 32
    assert serving_config["output_tokens"] == 8
    compiled_plan = compile_campaign_plan(
        experiment_config_path=smoke_bundle / "experiment.yaml",
        runner=load_runner_config(smoke_bundle / "runner.yaml"),
        execution=load_execution_config(smoke_bundle / "execution.yaml"),
        overrides=overrides,
        stage_filter="full",
    )
    replacement_node = next(
        node for node in compiled_plan.stages if node.stage_id == "replacement_scoring"
    )
    replacement_work = adapter_for_stage(replacement_node).plan(
        compiled_plan, replacement_node
    )
    assert replacement_node.instances == 1
    assert replacement_node.gpus_per_instance == 1
    assert replacement_node.total_gpus == 1
    assert replacement_node.nodes == 1
    assert len(replacement_work.items) == 1
    assert replacement_work.items[0].metadata["worker_count"] == 1

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
            "HF_DATASETS_OFFLINE": "1",
            "HF_HOME": str(cache_dir / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_CACHE": str(cache_dir / "datasets"),
            "TORCH_HOME": str(cache_dir / "torch"),
            "TRANSFORMERS_OFFLINE": "1",
            "XDG_CACHE_HOME": str(cache_dir / "xdg"),
        }
    )
    smoke_root = result_root / "smoke"
    completed = _run_campaign(project_root_path, smoke_bundle, environment)
    _assert_campaign_succeeded(completed, smoke_root)

    stages = enabled_stage_ids(config)
    assert config["model"]["descriptor_override"] == "qwen3_5_text"
    assert all(stage_is_complete(config, stage) for stage in stages)

    manifests = [smoke_root / "manifests" / f"{stage}.json" for stage in stages]
    assert all(json.loads(path.read_text())["status"] == "success" for path in manifests)

    pass_manifests = list(
        smoke_root.glob("pruning/pruning_scores/automodel/*/activation_passes_manifest.json")
    )
    assert len(pass_manifests) == 1
    score_files = list(pass_manifests[0].parent.glob("*/rank_*.pth"))
    assert score_files
    score_tensors = [
        tensor
        for score_file in score_files
        for tensor in _tensor_values(torch.load(score_file, map_location="cpu", weights_only=False))
    ]
    assert score_tensors
    assert all(tensor.numel() and torch.isfinite(tensor).all() for tensor in score_tensors)

    replacement_summary = json.loads(
        (smoke_root / "artifacts/replacement_scoring/summary.json").read_text()
    )
    assert replacement_summary["widths"] == [256]
    assert replacement_summary["scenario_count"] == 1
    replacement_results = [
        json.loads(path.read_text())
        for path in smoke_root.glob(
            "scenarios/*/distributed_eval/replacement_scoring/results/**/*.json"
        )
    ]
    assert replacement_results
    assert all(
        result["provenance"]["score_device_type"] == "cuda"
        and result["provenance"]["visible_cuda_device_count"] == 1
        for result in replacement_results
    )

    for checkpoint in (smoke_root / "ckpts/teacher", smoke_root / "ckpts/sorted_teacher"):
        assert (checkpoint / "config.json").is_file()
        assert list(checkpoint.glob("*.safetensors"))
    candidate_library = json.loads((smoke_root / "candidate_library.json").read_text())
    assert 256 in _nested_values(candidate_library, "intermediate_size")
    assert 512 in _nested_values(candidate_library, "intermediate_size")

    active_profiles = json.loads((smoke_root / "mip/active_profiles.json").read_text())
    assert active_profiles["status"] == "success"
    grids = [
        json.loads((smoke_root / "mip" / "profiles" / profile_id / "mip_grid.json").read_text())
        for profile_id in active_profiles["profile_ids"]
    ]
    assert all(grid["status"] == "success" for grid in grids)
    solution_paths = [
        Path(scenario["solution_path"])
        for grid in grids
        for scenario in grid["scenarios"]
        if scenario["status"] == "feasible"
    ]
    solutions = [solution for path in solution_paths for solution in json.loads(path.read_text())]
    assert solutions
    assert 256 in _nested_values(solutions, "intermediate_size")

    model = AutoModelForCausalLM.from_pretrained(
        smoke_root / "ckpts/sorted_teacher",
        dtype=torch.bfloat16,
        local_files_only=True,
    ).cuda()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 2, 3, 4]], device="cuda")).logits
    assert torch.isfinite(logits).all()

    durable_paths = [*manifests, *pass_manifests]
    durable_paths.extend(smoke_root.glob("mip/profiles/*/mip_grid.json"))
    before_resume = _artifact_digests(durable_paths)
    resumed = _run_campaign(project_root_path, smoke_bundle, environment)
    _assert_campaign_succeeded(resumed, smoke_root)
    assert _artifact_digests(durable_paths) == before_resume
