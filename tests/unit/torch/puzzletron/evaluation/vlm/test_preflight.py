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

"""Tests for VLM profile preflight and runtime settings."""

import hashlib
import json
import os
from pathlib import Path

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import (
    contracts,
    evaluator,
    preflight,
    profile,
    suites,
    tasks,
)
from examples.puzzletron.evaluation.vlm import run as evaluation
from modelopt.torch.puzzletron.evaluation import lmms
from tests.unit.torch.puzzletron.evaluation.vlm._test_utils import (
    _full_inputs,
    _use_offline_fakes,
    _write_checkpoint,
    _write_checkpoint_at,
    _write_core3_teacher_snapshot,
    _write_lmms_tasks,
)


def test_versioned_profile_preflight_reports_immutable_contract(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    name = "short-v1"
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            name,
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)

    contract = contracts.load_profile(name)
    assert prepared.report["profile_name"] == name
    assert prepared.report["profile_fingerprint"] == contract.fingerprint
    assert prepared.report["sample_set"] == contract.sample_set
    assert prepared.report["backend_profile"] == contract.backend_profile
    assert prepared.report["evaluator_profile"] == contract.evaluator_profile
    assert prepared.report["source_tasks"] == list(contract.source_tasks)
    assert prepared.report["quick_selected_rows"] == 344
    assert prepared.report["quick_row_identities"] == suites.manifest_row_identities(
        prepared.quick_manifest
    )
    assert prepared.report["quick_task_denominators"] == suites.manifest_task_denominators(
        prepared.quick_manifest
    )


def test_native_profile_builds_qwen35_backend_settings(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "judge-free-8_690-examples_r1-native",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
    (tmp_path / "tasks").mkdir()
    settings = preflight.settings(
        args,
        tasks_root=tmp_path / "tasks",
        configured_tasks=("modelopt_vlm_benchmark_mvbench",),
        prepared=prepared,
    )

    assert prepared.report["model_backend"] == "qwen3_5"
    assert settings["model"] == "qwen3_5"
    assert settings["checkpoint_arg"] == "pretrained"
    assert settings["model_args"] == {
        "attn_implementation": "sdpa",
        "device": "cuda",
        "device_map": "cuda",
        "enable_thinking": False,
        "fps": 2,
        "max_frames": 32,
    }
    assert "reasoning_parser" not in settings
    assert not (tmp_path / "tasks/modelopt_qwen35_no_think.jinja").exists()


@pytest.mark.parametrize(
    ("profile_name", "expected_eager"),
    [("core-3_24-examples_r1-vllm", True), ("core-3_344-examples_r1-vllm", None)],
)
def test_vllm_profile_forwards_runtime_settings(
    monkeypatch, tmp_path, profile_name, expected_eager
):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            profile_name,
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()
    settings = preflight.settings(
        args,
        tasks_root=tasks_root,
        configured_tasks=("modelopt_vlm_benchmark_realworldqa",),
        prepared=prepared,
    )
    argv, _, _ = lmms._build_command(
        settings,
        checkpoint=str(model),
        output_path=tmp_path / "lmms-results",
    )
    model_args = argv[argv.index("--model_args") + 1]

    assert settings["model_args"].get("enforce_eager") is expected_eager
    assert ("enforce_eager=True" in model_args) is (expected_eager is True)
    assert settings["model_args"]["attention_config"] == {"flash_attn_version": 2}
    assert 'attention_config={"flash_attn_version":2}' in model_args


def test_core3_full_teacher_profile_population_expectations_follow_group_shard(
    monkeypatch, tmp_path
):
    model, hf_home = _write_core3_teacher_snapshot(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("mvbench",))
    _use_offline_fakes(monkeypatch, lmms_root)
    media = hf_home / profile.VLM_BENCHMARK_DATASETS["mvbench"].media_dir
    media.mkdir(parents=True)
    (media / "sample").write_bytes(b"media")
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "core-3_full_r1-native",
            "--profile-task",
            "mvbench",
            "--profile-task-shard",
            "3/8",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
    expected_leaves = ("episodic_reasoning", "moving_direction", "egocentric_navigation")

    assert prepared.profile_task_leaves == expected_leaves
    assert evaluator._expected_task_populations(prepared, ("modelopt_vlm_benchmark_mvbench",)) == {
        suites.task_name("mvbench", leaf=leaf): 200 for leaf in expected_leaves
    }


def test_core3_full_teacher_profile_accepts_snapshot_symlink(monkeypatch, tmp_path):
    snapshot, hf_home = _write_core3_teacher_snapshot(tmp_path)
    checkpoint_alias = tmp_path / "teacher"
    checkpoint_alias.symlink_to(snapshot, target_is_directory=True)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa",))
    _use_offline_fakes(monkeypatch, lmms_root)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint_alias),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "core-3_full_r1-native",
            "--profile-task",
            "realworldqa",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)

    assert prepared.source_tasks == ("realworldqa",)
    assert prepared.report["model_pin"]["revision"] == snapshot.name


@pytest.mark.parametrize("kind", ["untracked-copy", "wrong-revision"])
def test_core3_full_teacher_profile_rejects_unpinned_checkpoint(monkeypatch, tmp_path, kind):
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    if kind == "untracked-copy":
        model = _write_checkpoint(tmp_path)
    else:
        parent = hf_home / "hub/models--Qwen--Qwen3.5-0.8B/snapshots"
        parent.mkdir(parents=True)
        model = _write_checkpoint_at(parent / "different")
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "core-3_full_r1-native",
            "--profile-task",
            "realworldqa",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match="requires the exact local Hub snapshot"):
        preflight.prepare(args)


def test_core3_full_teacher_profile_rejects_settings_override(monkeypatch, tmp_path):
    model, hf_home = _write_core3_teacher_snapshot(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa",))
    _use_offline_fakes(monkeypatch, lmms_root)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "core-3_full_r1-native",
            "--profile-task",
            "realworldqa",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match="do not allow settings overrides"):
        evaluator.evaluate(args, settings_overrides={"model": "vllm"})


def test_historical_short_profile_preserves_vllm_backend(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "short-v1",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
    (tmp_path / "tasks").mkdir()
    settings = preflight.settings(
        args,
        tasks_root=tmp_path / "tasks",
        configured_tasks=("modelopt_vlm_benchmark_realworldqa",),
        prepared=prepared,
    )

    assert prepared.report["lmms_eval_revision"] == checkpoint.LMMS_EVAL_LEGACY_REVISION
    assert settings["model"] == "vllm"
    assert settings["checkpoint_arg"] == "model"
    assert settings["reasoning_parser"] == "qwen3"
    assert settings["model_args"]["max_frame_num"] == 32
    assert Path(settings["model_args"]["chat_template"]).exists()


@pytest.mark.parametrize(
    ("option", "value"),
    [("--seed", "7"), ("--batch-size", "8")],
)
def test_versioned_profile_rejects_runtime_override(monkeypatch, tmp_path, option, value):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "short-v1",
            option,
            value,
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match=f"{option} cannot override"):
        preflight.prepare(args)


def test_all_row_profile_task_preserves_contract_identity(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    profile_name = "full-v1"
    task = "realworldqa"
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            profile_name,
            "--profile-task",
            task,
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)

    assert prepared.source_tasks == (task,)
    assert prepared.report["quick_selected_rows"] is None
    assert prepared.report["quick_row_identities"] is None
    assert prepared.report["quick_task_denominators"] is None
    assert (
        prepared.report["profile_fingerprint"] == contracts.load_profile(profile_name).fingerprint
    )


def test_exact_row_profile_group_shard_partitions_rows_and_leaves(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "judge-free-8_690-examples_r1-native",
            "--profile-task",
            "mvbench",
            "--profile-task-shard",
            "3/8",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)

    expected_leaves = ("episodic_reasoning", "moving_direction", "egocentric_navigation")
    assert prepared.profile_task_leaves == expected_leaves
    assert prepared.quick_manifest is not None
    manifest_rows = prepared.quick_manifest["tasks"]["mvbench"]["rows"]
    assert len(manifest_rows) == 24
    assert {row["leaf_task"] for row in manifest_rows} == {
        f"mvbench_{leaf}" for leaf in expected_leaves
    }
    manifest_selection = prepared.quick_manifest["tasks"]["mvbench"]["selection"]
    assert manifest_selection["population_rows"] == 600
    assert manifest_selection["selected_rows"] == 24
    assert [stratum["name"] for stratum in manifest_selection["strata"]] == list(expected_leaves)
    assert manifest_selection["selected_index_quantiles"] == {
        "method": "lower-order-statistic",
        "p0": 12,
        "p25": 37,
        "p50": 87,
        "p75": 137,
        "p100": 187,
    }
    assert (
        manifest_selection["selected_row_identities_sha256"]
        == hashlib.sha256(
            json.dumps(manifest_rows, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
    )
    assert prepared.report["quick_selected_rows"] == 24
    assert prepared.report["quick_row_identities"] == suites.manifest_row_identities(
        prepared.quick_manifest
    )
    assert prepared.report["quick_task_denominators"] == {
        "mvbench": {"population_rows": 600, "selected_rows": 24}
    }
    assert (
        evaluator._expected_task_populations(prepared, ("modelopt_vlm_benchmark_mvbench",)) is None
    )
    assert prepared.report["quick_manifest_sha256"] == suites.manifest_sha256(
        prepared.quick_manifest
    )
    assert (
        prepared.report["profile_fingerprint"]
        == contracts.load_profile("judge-free-8_690-examples_r1-native").fingerprint
    )
    tasks_root, _ = tasks.prepare(
        tmp_path / "results",
        suite=prepared.suite,
        source_tasks=prepared.source_tasks,
        profile_task_leaves=prepared.profile_task_leaves,
        dataset_snapshots=prepared.dataset_snapshots,
        quick_manifest=prepared.quick_manifest,
    )
    group = json.loads((tasks_root / "modelopt_vlm_benchmark_mvbench.yaml").read_text())
    assert group["task"] == [f"modelopt_vlm_benchmark_mvbench_{leaf}" for leaf in expected_leaves]


def test_empty_exact_row_leaf_filter_reaches_manifest_validation():
    contract = contracts.load_profile("core-3_344-examples_r1-native")
    manifest = contract.exact_rows
    assert manifest is not None
    entry = manifest["tasks"]["mvbench"]
    manifest["tasks"] = {
        "mvbench": preflight._shard_exact_row_task(
            entry,
            task="mvbench",
            leaves=("not-selected",),
        )
    }

    with pytest.raises(ValueError, match="must select at least one row"):
        suites.validate_exact_rows_manifest(
            manifest,
            expected_revision=str(contract.manifest["lmms_eval_revision"]),
            expected_tasks=("mvbench",),
        )


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (("--suite", "short", "--profile-task", "realworldqa"), "requires"),
        (("--profile", "short-v1", "--profile-task", "realworldqa"), "supported only"),
        (("--profile", "full-v1", "--profile-task-shard", "0/8"), "requires"),
        (
            (
                "--profile",
                "full-v1",
                "--profile-task",
                "realworldqa",
                "--profile-task-shard",
                "0/8",
            ),
            "supports only",
        ),
    ],
)
def test_profile_task_rejects_invalid_parent(monkeypatch, tmp_path, selection, message):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            *selection,
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match=message):
        preflight.prepare(args)


@pytest.mark.parametrize(
    ("extra", "environment", "message"),
    [
        ([], {"OPENAI_API_KEY": "token"}, "explicit --allow-judge-calls"),
        (["--allow-judge-calls"], {}, "judge credentials are missing"),
    ],
)
def test_full_profile_fails_closed_without_judge_authorization_or_credentials(
    monkeypatch, tmp_path, extra, environment, message
):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    for name in ("OPENAI_API_KEY", "AZURE_API_KEY", "AZURE_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--suite",
            "full",
            "--hf-home",
            str(hf_home),
            "--mmvu-judge-api-type",
            "openai",
            "--mmvu-judge-model",
            "judge",
            *extra,
        ]
    )
    with pytest.raises(ValueError, match=message):
        preflight.prepare(args)


def test_video_reader_validation_is_limited_to_video_suites(monkeypatch):
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: None)

    preflight._verify_video_reader(("realworldqa", "mmmu_val"))
    with pytest.raises(RuntimeError, match="decord-compatible reader"):
        preflight._verify_video_reader(("mvbench",))


def test_native_backend_validation_requires_qwen_vision_utilities(monkeypatch):
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: None)

    preflight._verify_backend_dependencies("vllm")
    with pytest.raises(RuntimeError, match="qwen-vl-utils"):
        preflight._verify_backend_dependencies("qwen3_5")


def test_credential_scope_restores_inherited_values(monkeypatch):
    expected = {
        name: f"secret-{index}"
        for index, name in enumerate(checkpoint.HUGGINGFACE_CREDENTIAL_NAMES)
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)
    with checkpoint.without_huggingface_credentials():
        assert all(name not in os.environ for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES)
    assert {name: os.environ.get(name) for name in expected} == expected
