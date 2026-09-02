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

"""High-value behavior tests for local VLM evaluation workflows."""

import importlib.util
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import (
    contracts,
    evaluator,
    post_mip,
    preflight,
    profile,
    suites,
    tasks,
)
from examples.puzzletron.evaluation.vlm import model as vlm_model
from examples.puzzletron.evaluation.vlm import run as evaluation

_QWEN_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "text_config": {
        "hidden_size": 1024,
        "intermediate_size": 3584,
        "model_type": "qwen3_5_text",
        "num_attention_heads": 8,
        "num_hidden_layers": 24,
        "num_key_value_heads": 2,
        "vocab_size": 248320,
    },
}
_TASK_CONFIGS = {name: item.task_config for name, item in profile.VLM_BENCHMARK_DATASETS.items()}


def test_direct_launcher_does_not_shadow_standard_library_profile():
    script = Path(evaluation.__file__).absolute()
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import runpy, sys; "
                "sys.path.insert(0, sys.argv[1]); "
                "runpy.run_path(sys.argv[2], run_name='modelopt_vlm_launcher'); "
                "import cProfile; "
                "assert callable(cProfile.run)"
            ),
            str(script.parent),
            str(script),
        ],
        check=True,
    )


def _write_checkpoint(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_QWEN_CONFIG) + "\n")
    (model / "preprocessor_config.json").write_text("{}\n")
    (model / "chat_template.jinja").write_text(
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
    )
    return model


def test_no_think_template_is_local_and_requires_checkpoint_switch(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    (checkpoint_path / "chat_template.jinja").write_text(
        "{% if enable_thinking is defined and enable_thinking is true %}"
        "<think>\n{% else %}<think>\n\n</think>\n\n{% endif %}"
    )
    generated = vlm_model.no_think_chat_template(checkpoint_path, tasks_root)
    assert generated.parent == tasks_root
    (checkpoint_path / "chat_template.jinja").write_text("unsupported\n")
    with pytest.raises(ValueError, match="cannot disable thinking"):
        vlm_model.no_think_chat_template(checkpoint_path, tasks_root)


def test_no_think_template_rejects_unsafe_checkpoint_expression(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    (checkpoint_path / "chat_template.jinja").write_text(
        "{{ ''.__class__.__mro__ }}"
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% endif %}"
    )
    tasks_root = tmp_path / "tasks"
    tasks_root.mkdir()

    with pytest.raises(ValueError, match="chat template is invalid"):
        vlm_model.no_think_chat_template(checkpoint_path, tasks_root)


def test_checkpoint_contract_accepts_only_matching_realized_anymodel(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["AnyModel"],
        base_architecture="Qwen3_5ForConditionalGeneration",
    )
    config_path.write_text(json.dumps(config) + "\n")

    vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config["base_architecture"] = "OtherForConditionalGeneration"
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="AnyModel base_architecture"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config.update(
        architectures=["AnyModel", "Qwen3_5ForConditionalGeneration"],
        base_architecture="Qwen3_5ForConditionalGeneration",
    )
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="AnyModel base_architecture"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


def test_checkpoint_contract_accepts_other_positive_qwen35_geometry(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config["text_config"].update(
        hidden_size=2560,
        intermediate_size=9728,
        num_attention_heads=20,
        num_hidden_layers=40,
        num_key_value_heads=4,
    )
    config_path.write_text(json.dumps(config) + "\n")

    vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config["text_config"]["hidden_size"] = 0
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="invalid Qwen 3.5 geometry"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


@pytest.mark.parametrize("processor_content", [None, "", "[]\n", "{\n", b"\xff"])
def test_checkpoint_contract_requires_valid_local_processor_assets(tmp_path, processor_content):
    checkpoint_path = _write_checkpoint(tmp_path)
    processor_path = checkpoint_path / "preprocessor_config.json"
    if processor_content is None:
        processor_path.unlink()
    elif isinstance(processor_content, bytes):
        processor_path.write_bytes(processor_content)
    else:
        processor_path.write_text(processor_content)

    with pytest.raises(ValueError, match="processor asset"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


def test_checkpoint_contract_rejects_malformed_companion_processor_asset(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    (checkpoint_path / "video_preprocessor_config.json").write_text("{\n")

    with pytest.raises(ValueError, match="video_preprocessor_config.json"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")


def _write_lmms_tasks(root: Path, tasks: tuple[str, ...]) -> Path:
    lmms_root = root / "lmms_eval"
    for task in tasks:
        task_configs = [_TASK_CONFIGS[task]]
        if task == "video_mmmu":
            task_configs.extend(
                f"tasks/videommmu/{leaf}.yaml" for leaf in suites.VIDEO_MMMU_LEAF_TASKS
            )
        elif task == "mvbench":
            task_configs.extend(
                f"tasks/mvbench/mvbench_{leaf}.yaml" for leaf in suites.MVBENCH_LEAF_TASKS
            )
        for relative_path in task_configs:
            config = lmms_root / relative_path
            config.parent.mkdir(parents=True, exist_ok=True)
            config.write_text(f"task: {task}\n")
    return lmms_root


def _use_offline_fakes(monkeypatch, lmms_root: Path) -> None:
    monkeypatch.setattr(tasks, "_lmms_eval_root", lambda: lmms_root)
    monkeypatch.setattr(
        checkpoint,
        "verify_lmms_eval_revision",
        lambda expected=checkpoint.LMMS_EVAL_REVISION: expected,
    )
    monkeypatch.setattr(
        suites,
        "offline_dataset_snapshot",
        lambda hf_home, task, revision: hf_home / ".snapshots" / task / revision,
    )
    monkeypatch.setattr(
        tasks,
        "verify_offline",
        lambda _root, tasks, **_kwargs: {
            "configured_tasks": list(tasks),
            "status": "passed",
        },
    )


def _full_inputs(monkeypatch, tmp_path):
    model = _write_checkpoint(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, profile.VLM_BENCHMARK_TASKS)
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    for dataset in profile.VLM_BENCHMARK_VIDEO_DATASETS.values():
        media = hf_home / dataset.media_dir
        media.mkdir(parents=True, exist_ok=True)
        (media / "sample").write_bytes(b"media")
    return model, hf_home


def _quick_manifest(path: Path) -> Path:
    counts = {"realworldqa": 64, "mmmu_val": 120}
    tasks = {
        task: {
            "dataset_revision": profile.VLM_BENCHMARK_DATASETS[task].revision,
            "rows": [
                {
                    "source_row_index": index,
                    "source_sample_id": (
                        f"test:{index}" if task == "realworldqa" else f"question-{index}"
                    ),
                }
                for index in range(count)
            ],
        }
        for task, count in counts.items()
    }
    tasks["mvbench"] = {
        "dataset_revision": profile.VLM_BENCHMARK_DATASETS["mvbench"].revision,
        "rows": [
            {
                "leaf_task": f"mvbench_{leaf}",
                "source_row_index": index,
                "source_sample_id": f"{leaf}:{index}",
            }
            for leaf in suites.MVBENCH_LEAF_TASKS
            for index in range(8)
        ],
    }
    path.write_text(
        json.dumps(
            {
                "schema": "modelopt.vlm-benchmark-quick/v1",
                "lmms_eval_revision": checkpoint.LMMS_EVAL_REVISION,
                "tasks": tasks,
            }
        )
        + "\n"
    )
    return path


def test_short_profile_materializes_pinned_tasks_and_vllm_backend(monkeypatch, tmp_path, capsys):
    model = _write_checkpoint(tmp_path)
    source_tasks = ("realworldqa", "mmmu_val")
    lmms_root = _write_lmms_tasks(tmp_path, source_tasks)
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    output = tmp_path / "results"
    for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES:
        monkeypatch.setenv(name, f"inherited-{name.lower()}")
    calls = []

    def fake_runner(checkpoint_path, *, output_root, settings):
        calls.append(
            {
                "checkpoint": checkpoint_path,
                "credentials": {
                    name: os.environ.get(name) for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES
                },
                "output_root": output_root,
                "settings": settings,
            }
        )
        result_path = output_root / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n")
        return {
            "attempt": len(calls),
            "metrics": {"accuracy": len(calls) / 10},
            "output_root": str(output_root),
            "result_path": str(result_path),
        }

    monkeypatch.setattr(checkpoint, "run_lmms_eval_checkpoint", fake_runner)
    argv = [
        "--checkpoint",
        str(model),
        "--output-dir",
        str(output),
        "--suite",
        "short",
        "--hf-home",
        str(hf_home),
    ]

    assert evaluation.main(argv) == 0

    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "modelopt.vlm-evaluation-result/v1"
    report = result["preflight"]
    assert report["source_tasks"] == list(source_tasks)
    assert report["short_repetitions"] == 2
    assert report["lmms_eval_revision"] == checkpoint.LMMS_EVAL_REVISION
    generated = json.loads(
        (output / "task_configs/modelopt_vlm_benchmark_realworldqa.yaml").read_text()
    )
    assert generated["dataset_path"].endswith(
        profile.VLM_BENCHMARK_DATASETS["realworldqa"].revision
    )
    expected_tasks = (
        "modelopt_vlm_benchmark_realworldqa",
        "modelopt_vlm_benchmark_mmmu_val",
    )
    assert [call["checkpoint"] for call in calls] == [model, model]
    assert [call["output_root"] for call in calls] == [
        output / "short-repetition-1",
        output / "short-repetition-2",
    ]
    assert all(not any(call["credentials"].values()) for call in calls)
    assert all(call["settings"]["tasks"] == ",".join(expected_tasks) for call in calls)
    assert [run["attempt"] for run in result["runs"]] == [1, 2]
    for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES:
        assert os.environ[name] == f"inherited-{name.lower()}"
    settings = calls[0]["settings"]
    assert settings["model"] == "vllm"
    assert settings["log_samples"] is True
    assert settings["checkpoint_arg"] == "model"
    assert settings["reasoning_parser"] == "qwen3"
    chat_template = Path(settings["model_args"]["chat_template"])
    assert chat_template.read_text().startswith("{%- set enable_thinking = false %}\n")
    assert "topology" not in settings
    assert settings["env"]["HF_HUB_OFFLINE"] == "1"
    assert settings["env"]["API_TYPE"] == "openai"
    assert settings["env"]["MODEL_VERSION"] == "modelopt-disabled-lmms-eval-judge"
    assert settings["env"]["OPENAI_API_KEY"] == "modelopt-disabled-lmms-eval-judge"
    assert settings["env"]["OPENAI_API_URL"] == "http://127.0.0.1:9"
    assert report["model_backend"] == settings["model"]
    assert report["backend_limitations"] == [
        "generic vLLM video messages do not preserve native Qwen 3.5 timestamps",
    ]
    assert report["sample_limit"] == settings["limit"]
    assert report["timeout_seconds"] == settings["timeout_seconds"]
    assert report["frame_policy"] == {
        "reader": settings["env"]["FORCE_QWENVL_VIDEO_READER"],
        "fps": settings["model_args"]["fps"],
        "max_frames": settings["model_args"]["max_frame_num"],
    }
    assert report["generation_policy"] == settings["gen_kwargs"]


@pytest.mark.parametrize("suite", ["short", suites.TASK_PREFIX100_REPEAT2_SUITE])
def test_repeated_profile_resumes_completed_repetitions(monkeypatch, tmp_path, suite):
    model = _write_checkpoint(tmp_path)
    source_tasks = ("realworldqa", "mmmu_val")
    lmms_root = _write_lmms_tasks(tmp_path, source_tasks)
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    output = tmp_path / "results"
    calls = []

    def fake_runner(checkpoint_path, *, output_root, settings):
        calls.append(output_root)
        result_path = output_root / "attempt" / "summary.json"
        raw_result_path = output_root / "attempt" / "samples.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n")
        raw_result_path.write_text("{}\n")
        return {
            "metrics": {"accuracy": len(calls) / 10},
            "raw_result_path": str(raw_result_path),
            "result_path": str(result_path),
        }

    monkeypatch.setattr(checkpoint, "run_lmms_eval_checkpoint", fake_runner)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(output),
            "--suite",
            suite,
            "--hf-home",
            str(hf_home),
        ]
    )

    first = evaluation.evaluate(args)
    second = evaluation.evaluate(args)

    assert len(calls) == 2
    assert second["runs"] == first["runs"]
    for repetition in (1, 2):
        completed = json.loads(
            (output / f"{suite}-repetition-{repetition}" / "completed_run.json").read_text()
        )
        assert completed["schema"] == "modelopt.vlm-evaluation-completed-run/v1"
        assert completed["identity"]["repetition"] == repetition
        assert completed["identity"]["checkpoint"]["fingerprint"]
        assert completed["identity"]["profile"]["suite"] == suite


@pytest.mark.parametrize(
    ("corruption", "expected_calls"),
    [("checkpoint", 4), ("artifact", 3), ("result", 3)],
)
def test_short_profile_reruns_stale_completed_repetitions(
    monkeypatch,
    tmp_path,
    corruption,
    expected_calls,
):
    model = _write_checkpoint(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa", "mmmu_val"))
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    output = tmp_path / "results"
    calls = []

    def fake_runner(checkpoint_path, *, output_root, settings):
        calls.append(output_root)
        result_path = output_root / "attempt" / "summary.json"
        raw_result_path = output_root / "attempt" / "samples.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n")
        raw_result_path.write_text("{}\n")
        return {
            "metrics": {"accuracy": 0.5},
            "raw_result_path": str(raw_result_path),
            "result_path": str(result_path),
        }

    monkeypatch.setattr(checkpoint, "run_lmms_eval_checkpoint", fake_runner)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(output),
            "--suite",
            "short",
            "--hf-home",
            str(hf_home),
        ]
    )
    evaluation.evaluate(args)

    if corruption == "checkpoint":
        (model / "preprocessor_config.json").write_text('{"changed": true}\n')
    elif corruption == "artifact":
        (output / "short-repetition-1" / "attempt" / "samples.json").unlink()
    else:
        (output / "short-repetition-1" / "attempt" / "summary.json").unlink()

    evaluation.evaluate(args)
    assert len(calls) == expected_calls


@pytest.mark.parametrize(
    "record",
    [
        "{\n",
        json.dumps(
            {
                "identity": {},
                "result": {"metrics": []},
                "schema": "modelopt.vlm-evaluation-completed-run/v1",
            }
        ),
    ],
)
def test_completed_repetition_records_fail_closed_when_malformed(tmp_path, record):
    output = tmp_path / "results"
    output.mkdir()
    (output / "completed_run.json").write_text(record)

    with pytest.raises(RuntimeError, match="invalid completed VLM evaluation"):
        evaluator._load_completed_run(output, identity={})


def test_realworldqa_mmmu_prefix100_policy_is_explicit_and_repeated():
    suite = suites.TASK_PREFIX100_REPEAT2_SUITE
    assert suites.source_tasks(suite) == ("realworldqa", "mmmu_val")
    policy = suites.execution_policy(suite, timeout_seconds=14400)
    assert policy["limit"] == 100
    assert policy["repetitions"] == 2
    assert policy["generation"] == {"temperature": 0, "do_sample": False}
    assert suites.canonical_suite("e2e-full-eval") == suite
    assert suites.execution_policy("full", timeout_seconds=None)["limit"] is None
    assert suites.execution_policy("full-v1", timeout_seconds=None)["limit"] is None


def test_deprecated_suite_alias_records_the_canonical_identity(monkeypatch, tmp_path):
    model = _write_checkpoint(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa", "mmmu_val"))
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--suite",
            "e2e-full-eval",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.warns(FutureWarning, match="is deprecated"):
        prepared = preflight.prepare(args)

    assert prepared.suite == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert prepared.report["suite"] == suites.TASK_PREFIX100_REPEAT2_SUITE


def test_deprecated_full_name_reports_explicit_suite_identity(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--suite",
            "e2e-full-eval",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.warns(FutureWarning, match="realworldqa-mmmu-prefix100-x2"):
        prepared = preflight.prepare(args)

    assert prepared.suite == "realworldqa-mmmu-prefix100-x2"
    assert prepared.report["suite"] == prepared.suite
    assert prepared.execution_policy["limit"] == 100


def test_versioned_profile_contracts_pin_selection_and_fingerprints():
    profiles = {name: contracts.load_profile(name) for name in contracts.PROFILE_NAMES}
    assert {name: contract.fingerprint for name, contract in profiles.items()} == {
        "short-v1": "984c23ef0e7c05248895ece69c12327b3cdbb45051189ec540f7fc1ada763177",
        "short-native-v1": "217b8ba8fd1df0002407e75f6e7d5588e3a871a6df2ad24117b66377894b2f35",
        "short-all-native-v1": "06b17ea010ee0cd789e49c581bcb3be4a7624c8471b4b3102bfa2922e0929e68",
        "full-v1": "29b1db6123ea3e16a9c5693e81e0f31607ff8a08e436681c66c32bf5dcc7e67a",
    }

    short = profiles["short-v1"]
    short_all_native = profiles["short-all-native-v1"]
    assert {
        task: len(entry["rows"]) for task, entry in short_all_native.manifest["tasks"].items()
    } == {
        "realworldqa": 64,
        "mmmu_val": 120,
        "mvbench": 160,
        "video_mmmu": 72,
        "videomme": 72,
        "longvideobench_val_v": 68,
        "mlvu_dev": 70,
        "perceptiontest_val_mc": 64,
    }
    assert short.exact_rows is not None
    assert profiles["full-v1"].exact_rows is None
    mmmu_rows = short.manifest["tasks"]["mmmu_val"]["rows"]
    assert Counter(row["source_row_index"] // 30 for row in mmmu_rows) == Counter(
        dict.fromkeys(range(30), 4)
    )

    exact_rows = short.exact_rows
    assert exact_rows is not None
    path = tmp_path / "short-v1-rows.json"
    path.write_text(json.dumps(exact_rows))
    validated = suites.load_quick_manifest(path)
    assert suites.manifest_sha256(validated)


def test_short_all_native_profile_builds_grouped_and_single_selectors(tmp_path):
    contract = contracts.load_profile("short-all-native-v1")
    exact_rows = contract.exact_rows
    assert exact_rows is not None
    validated = suites.validate_exact_rows_manifest(
        exact_rows,
        expected_revision=checkpoint.LMMS_EVAL_QWEN35_NATIVE_REVISION,
        expected_tasks=contract.source_tasks,
    )

    tasks._write_quick_selection_module(tmp_path, validated)
    spec = importlib.util.spec_from_file_location(
        "short_all_selectors", tmp_path / "modelopt_quick_selection.py"
    )
    assert spec is not None and spec.loader is not None
    selectors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selectors)

    class Documents:
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, _index):
            return {}

        def select(self, indices):
            return list(indices)

    tasks_manifest = validated["tasks"]
    adaptation = [
        row["source_row_index"]
        for row in tasks_manifest["video_mmmu"]["rows"]
        if row["leaf_task"] == "video_mmmu_adaptation"
    ]
    assert (
        selectors.select_modelopt_vlm_benchmark_video_mmmu_adaptation(Documents(300)) == adaptation
    )
    videomme = [row["source_row_index"] for row in tasks_manifest["videomme"]["rows"]]
    assert selectors.select_modelopt_vlm_benchmark_videomme(Documents(2700)) == videomme


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
    assert prepared.report["source_tasks"] == list(contract.source_tasks)
    assert prepared.report["quick_selected_rows"] == 344


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
            "short-native-v1",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
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


def test_versioned_profile_rejects_seed_override(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "short-v1",
            "--seed",
            "7",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match="--seed cannot override"):
        preflight.prepare(args)


def test_versioned_profile_rejects_batch_size_override(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "short-v1",
            "--batch-size",
            "8",
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.raises(ValueError, match="--batch-size cannot override"):
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
            "short-all-native-v1",
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
    assert prepared.report["quick_selected_rows"] == 24
    assert prepared.report["quick_manifest_sha256"] == suites.manifest_sha256(
        prepared.quick_manifest
    )
    assert (
        prepared.report["profile_fingerprint"]
        == contracts.load_profile("short-all-native-v1").fingerprint
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


def test_post_mip_realworldqa_adapter_runs_pinned_profile(monkeypatch, tmp_path):
    model = _write_checkpoint(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("realworldqa",))
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    output = tmp_path / "output"
    captured = {}

    def fake_runner(checkpoint_path, *, output_root, settings):
        report = json.loads((output / "profile.json").read_text())
        assert report["configured_tasks"] == ["modelopt_vlm_benchmark_realworldqa"]
        assert report["sample_limit"] == 2
        captured.update(
            checkpoint=checkpoint_path,
            output_root=output_root,
            settings=settings,
        )
        result_path = output_root / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text("{}\n")
        return {
            "metrics": {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5},
            "result_path": str(result_path),
        }

    monkeypatch.setattr(checkpoint, "run_lmms_eval_checkpoint", fake_runner)
    result = post_mip.evaluate_realworldqa_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 900,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )

    assert captured["checkpoint"] == model
    assert captured["output_root"] == output
    assert captured["settings"]["tasks"] == "modelopt_vlm_benchmark_realworldqa"
    assert captured["settings"]["limit"] == 2
    assert captured["settings"]["timeout_seconds"] == 900
    assert captured["settings"]["dtype"] == "bfloat16"
    assert captured["settings"]["topology"] == {"tensor_parallel_size": 1}
    assert result["metrics"] == {"modelopt_vlm_benchmark_realworldqa.accuracy": 0.5}
    assert result["profile_path"] == str(output / "profile.json")


def test_post_mip_prefix100_adapter_averages_repeated_bounded_tasks(
    monkeypatch,
    tmp_path,
):
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    captured = {"invocations": 0}

    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        captured["invocations"] += 1
        captured.update(args=args, settings_overrides=settings_overrides)
        preflight_callback({"profile": suites.EVALUATION_PROFILE, "sample_limit": None})
        runs = []
        score_offset = (captured["invocations"] - 1) * 0.2
        for index, realworldqa_score in enumerate(
            (0.4 + score_offset, 0.6 + score_offset), start=1
        ):
            result_path = tmp_path / f"run-{index}.json"
            result_path.write_text("{}")
            runs.append(
                {
                    "metrics": {
                        "modelopt_vlm_benchmark_realworldqa.exact_match_none": (realworldqa_score),
                        "modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.3,
                    },
                    "result_path": str(result_path),
                }
            )
        return {"runs": runs}

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)
    result = post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 14400,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )

    assert captured["args"].suite == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert captured["args"].batch_size == 1
    assert captured["args"].seed == 42
    assert captured["settings_overrides"] == {
        "dtype": "bfloat16",
        "topology": {"tensor_parallel_size": 1},
    }
    assert result["metrics"] == {
        "modelopt_vlm_benchmark_mmmu_val.mmmu_acc_none": 0.3,
        "modelopt_vlm_benchmark_realworldqa.exact_match_none": 0.5,
    }
    assert result["profile"] == post_mip.TASK_PREFIX100_REPEAT2_PROFILE
    summary = json.loads(Path(result["result_path"]).read_text())
    assert summary["suite"] == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert summary["profile"] == post_mip.TASK_PREFIX100_REPEAT2_PROFILE
    assert summary["metrics"] == result["metrics"]
    assert summary["result_paths"] == result["run_result_paths"]

    refreshed = post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
        model,
        output_root=output,
        settings={
            "batch_size": 1,
            "timeout_seconds": 14400,
            "dtype": "bfloat16",
            "topology": {"tensor_parallel_size": 1},
        },
    )
    assert refreshed["metrics"][
        "modelopt_vlm_benchmark_realworldqa.exact_match_none"
    ] == pytest.approx(0.7)
    assert json.loads(Path(refreshed["result_path"]).read_text())["metrics"] == refreshed["metrics"]


def test_post_mip_prefix100_rejects_different_repetition_metrics(
    monkeypatch,
    tmp_path,
):
    def fake_evaluate(args, *, settings_overrides, preflight_callback):
        return {
            "runs": [
                {"metrics": {"realworldqa.accuracy": 0.5}, "result_path": "first.json"},
                {"metrics": {"mmmu.accuracy": 0.5}, "result_path": "second.json"},
            ]
        }

    monkeypatch.setattr(post_mip, "evaluate", fake_evaluate)

    with pytest.raises(RuntimeError, match="produced different metrics"):
        post_mip.evaluate_realworldqa_mmmu_prefix100_checkpoint(
            tmp_path / "model",
            output_root=tmp_path / "output",
            settings={},
        )


def test_deprecated_post_mip_profile_alias_forwards_to_canonical(monkeypatch, tmp_path):
    expected = {"metrics": {"accuracy": 0.5}}
    monkeypatch.setattr(
        post_mip,
        "evaluate_realworldqa_mmmu_prefix100_checkpoint",
        lambda *_args, **_kwargs: expected,
    )

    with pytest.warns(FutureWarning, match="is deprecated"):
        result = post_mip.evaluate_e2e_full_eval_checkpoint(
            tmp_path / "model",
            output_root=tmp_path / "output",
            settings={},
        )

    assert result is expected


def test_mmvu_guard_is_limited_to_full_suite(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    lmms_root = _write_lmms_tasks(tmp_path, profile.VLM_BENCHMARK_TASKS)
    monkeypatch.setattr(tasks, "_lmms_eval_root", lambda: lmms_root)

    tasks_root, configured_tasks = tasks.prepare(
        tmp_path / "results",
        suite="mmvu-smoke",
        dataset_snapshots={"mmvu_val": snapshot},
        quick_manifest=None,
    )

    assert configured_tasks == ("modelopt_vlm_benchmark_mmvu_val",)
    assert not (tasks_root / "modelopt_mmvu_guard.py").exists()
    generated = (tasks_root / "modelopt_vlm_benchmark_mmvu_val.yaml").read_text()
    assert "\nprocess_results:" not in generated
    assert (
        "process_docs: !function "
        "modelopt_mmvu_smoke_selection.select_modelopt_vlm_benchmark_mmvu_val\n" in generated
    )
    assert (tasks_root / "modelopt_mmvu_smoke_selection.py").is_file()
    assert not (tasks_root / "modelopt_quick_selection.py").exists()

    full_root, _ = tasks.prepare(
        tmp_path / "full-results",
        suite="full",
        dataset_snapshots=dict.fromkeys(profile.VLM_BENCHMARK_TASKS, snapshot),
        quick_manifest=None,
    )
    assert (full_root / "modelopt_mmvu_guard.py").is_file()
    full_generated = (full_root / "modelopt_vlm_benchmark_mmvu_val.yaml").read_text()
    assert "\nprocess_results: !function modelopt_mmvu_guard.process_results\n" in full_generated


def test_quick_manifest_requires_exact_pins_counts_and_leaf_balance(tmp_path):
    path = _quick_manifest(tmp_path / "quick.json")
    assert suites.manifest_sha256(suites.load_quick_manifest(path))

    manifest = json.loads(path.read_text())
    manifest["tasks"]["mmmu_val"]["rows"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly 120 rows"):
        suites.load_quick_manifest(path)

    path = _quick_manifest(path)
    manifest = json.loads(path.read_text())
    manifest["tasks"]["mvbench"]["rows"][-1] = {
        "leaf_task": "mvbench_action_sequence",
        "source_row_index": 8,
        "source_sample_id": "action_sequence:8",
    }
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exactly 8 rows per leaf task"):
        suites.load_quick_manifest(path)


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


def test_offline_preflight_scrubs_credentials_and_traverses_media(monkeypatch, tmp_path):
    for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES:
        monkeypatch.setenv(name, "must-not-reach-child")
    tasks_root = tmp_path / "tasks"
    package = tasks_root / "lmms_eval/tasks"
    package.mkdir(parents=True)
    (tasks_root / "lmms_eval/__init__.py").write_text("")
    (package / "__init__.py").write_text("""import os
class Config: task = "modelopt_vlm_benchmark_mvbench"
class Task:
    config = Config()
    def has_test_docs(self): return True
    def has_validation_docs(self): return False
    def has_training_docs(self): return False
    def test_docs(self): return [{"video": os.environ["FAKE_MEDIA_PATH"]}]
    def doc_to_visual(self, document): return [document["video"]]
class Group: group_name = "modelopt_vlm_benchmark_mvbench"
class TaskManager:
    def __init__(self, include_path, model_name):
        assert model_name == "qwen3_5"
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["API_TYPE"] == "openai"
        assert os.environ["MODEL_VERSION"] == "modelopt-disabled-lmms-eval-judge"
        assert os.environ["OPENAI_API_KEY"] == "modelopt-disabled-lmms-eval-judge"
        assert os.environ["OPENAI_API_URL"] == "http://127.0.0.1:9"
        credential_names = (
            "HF_TOKEN",
            "HUGGINGFACEHUB_API_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
        )
        assert all(name not in os.environ for name in credential_names)
    def load_task_or_group(self, tasks): return {Group(): {"leaf": Task()}}
""")
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    media = tmp_path / "sample.mp4"
    media.write_bytes(b"video")
    monkeypatch.setenv("FAKE_MEDIA_PATH", str(media))

    report = tasks.verify_offline(
        tasks_root,
        ("modelopt_vlm_benchmark_mvbench",),
        hf_home=hf_home,
        timeout_seconds=123,
        model_name="qwen3_5",
    )

    assert report["media_documents"] == 1
    assert report["status"] == "passed"


def test_video_adapter_normalizes_supported_suffixes_and_rejects_unknown(monkeypatch, tmp_path):
    uppercase = tmp_path / "sample.MP4"
    matroska = tmp_path / "sample.mkv"
    unknown = tmp_path / "sample.webm"
    for path in (uppercase, matroska, unknown):
        path.write_bytes(b"video")
    videomme = ModuleType("lmms_eval.tasks.videomme.utils")
    videomme.videomme_doc_to_visual = lambda _document: [str(matroska)]
    perception = ModuleType("lmms_eval.tasks.perceptiontest.val.utils")
    perception.perceptiontest_val_doc_to_visual = lambda _document: [str(uppercase)]
    packages = {
        "lmms_eval": ModuleType("lmms_eval"),
        "lmms_eval.tasks": ModuleType("lmms_eval.tasks"),
        "lmms_eval.tasks.videomme": ModuleType("lmms_eval.tasks.videomme"),
        "lmms_eval.tasks.videomme.utils": videomme,
        "lmms_eval.tasks.perceptiontest": ModuleType("lmms_eval.tasks.perceptiontest"),
        "lmms_eval.tasks.perceptiontest.val": ModuleType("lmms_eval.tasks.perceptiontest.val"),
        "lmms_eval.tasks.perceptiontest.val.utils": perception,
    }
    packages["lmms_eval.tasks.videomme"].utils = videomme
    packages["lmms_eval.tasks.perceptiontest.val"].utils = perception
    for name, module in packages.items():
        monkeypatch.setitem(sys.modules, name, module)

    tasks._write_video_path_adapter(tmp_path)
    spec = importlib.util.spec_from_file_location(
        "adapter_test", tmp_path / "modelopt_video_paths.py"
    )
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)

    for generated, source in (
        (adapter.videomme_doc_to_visual({})[0], matroska),
        (adapter.perceptiontest_doc_to_visual({})[0], uppercase),
    ):
        alias = Path(generated)
        assert alias.suffix == ".mp4"
        assert alias.resolve() == source
    with pytest.raises(ValueError, match="unsupported Qwen 3.5 video suffix"):
        adapter._normalize([str(unknown)])


def test_profile_contract_pins_every_task_and_revision():
    assert profile.VLM_BENCHMARK_TASKS == (
        "realworldqa",
        "mmmu_val",
        "video_mmmu",
        "mvbench",
        "mmvu_val",
        "videomme",
        "longvideobench_val_v",
        "mlvu_dev",
        "perceptiontest_val_mc",
    )
    assert all(len(item.revision) == 40 for item in profile.VLM_BENCHMARK_DATASETS.values())


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


def test_requirements_pin_matches_runtime_lmms_eval_revision():
    requirements = (checkpoint.REPOSITORY_ROOT / "examples/puzzletron/requirements.txt").read_text()
    assert (
        "-e git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git@"
        f"{checkpoint.LMMS_EVAL_REVISION}#egg=lmms-eval"
    ) in requirements.splitlines()
    assert 'eva-decord==0.6.1; platform_system == "Linux"' in requirements.splitlines()
    environment = json.loads(
        (checkpoint.REPOSITORY_ROOT / "examples/puzzletron/ci_environment.json").read_text()
    )
    assert environment["lmms_eval"]["commit"] == checkpoint.LMMS_EVAL_REVISION


def test_vlm_parser_exposes_only_suite_owned_sample_limits():
    help_text = evaluation._build_parser().format_help()
    assert "--limit" not in help_text
    assert "--full" not in help_text
    assert "--tasks" not in help_text
    assert "--evaluation-profile" not in help_text


def test_vlm_parser_defaults_to_short_suite():
    assert evaluation._build_parser().get_default("suite") == "short"


def test_huggingface_dependency_supports_range_metadata_api():
    pyproject = (checkpoint.REPOSITORY_ROOT / "pyproject.toml").read_text()
    assert '"huggingface_hub>=0.30.0",' in pyproject


def test_credential_scope_restores_inherited_values(monkeypatch):
    for index, name in enumerate(checkpoint.HUGGINGFACE_CREDENTIAL_NAMES):
        monkeypatch.setenv(name, f"secret-{index}")
    with checkpoint.without_huggingface_credentials():
        assert all(name not in os.environ for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES)
    assert all(name in os.environ for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES)
