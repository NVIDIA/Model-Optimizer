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

import hashlib
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
from hashlib import sha256
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
from modelopt.torch.puzzletron.evaluation import lmms

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


def _homogeneous_qwen_block_configs() -> list[dict[str, object]]:
    block = {
        "subblock_configs": [
            {
                "kind": "attention",
                "name": "attention",
                "no_op": False,
                "num_kv_heads": 2,
                "num_query_heads": 8,
            },
            {
                "kind": "ffn",
                "name": "ffn",
                "no_op": False,
                "intermediate_size": 3584,
            },
        ]
    }
    return [json.loads(json.dumps(block)) for _ in range(24)]


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


def _write_checkpoint_at(model: Path) -> Path:
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_QWEN_CONFIG) + "\n")
    (model / "preprocessor_config.json").write_text("{}\n")
    (model / "chat_template.jinja").write_text(
        "{% if enable_thinking is defined and enable_thinking is false %}"
        "<think>\n\n</think>\n\n{% else %}<think>\n{% endif %}"
    )
    return model


def _write_checkpoint(root: Path) -> Path:
    return _write_checkpoint_at(root / "model")


def _write_core3_teacher_snapshot(root: Path) -> tuple[Path, Path]:
    hf_home = root / "hf-home"
    snapshot = (
        hf_home
        / "hub/models--Qwen--Qwen3.5-0.8B/snapshots"
        / "2fc06364715b967f1860aea9cf38778875588b17"
    )
    snapshot.parent.mkdir(parents=True)
    return _write_checkpoint_at(snapshot), hf_home


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


def test_chat_template_fingerprint_accepts_file_and_inline_content(tmp_path):
    content = "{% if messages %}{{ messages[0]['content'] }}{% endif %}"
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text(content)
    expected = sha256(content.encode()).hexdigest()

    assert (
        evaluator._chat_template_sha256({"model_args": {"chat_template": str(template_path)}})
        == expected
    )
    assert evaluator._chat_template_sha256({"model_args": {"chat_template": content}}) == expected


def test_mmmu_parser_audit_is_attached_to_normalized_result(tmp_path):
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    task_name = suites.task_name("mmmu_val")
    result_path = attempt / "summary.json"
    result_path.write_text(json.dumps({"sample_counts": {task_name: 2}}))
    raw_result_path = attempt / "20260903_120000_results.json"
    raw_result_path.write_text("{}\n")
    sample_path = attempt / f"20260903_120000_samples_{task_name}.jsonl"
    sample_path.write_text(
        "\n".join(
            json.dumps({"mmmu_acc": {"parser_status": [status]}})
            for status in ("parsed", "fallback_random")
        )
        + "\n"
    )
    (attempt / f"20260903_110000_samples_{task_name}.jsonl").write_text("{not-json}\n")

    evaluator._attach_mmmu_parser_audit(
        {"raw_result_path": str(raw_result_path), "result_path": str(result_path)}
    )

    audit = json.loads(result_path.read_text())["mmmu_parser_audit"]
    assert audit["sample_count"] == 2
    assert audit["status_counts"] == {"fallback_random": 1, "parsed": 1}
    assert audit["sample_logs"] == [
        {
            "path": sample_path.name,
            "sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
            "size": sample_path.stat().st_size,
        }
    ]


def test_mmmu_parser_audit_rejects_unlabeled_sample(tmp_path):
    task_name = suites.task_name("mmmu_val")
    result_path = tmp_path / "summary.json"
    result_path.write_text(json.dumps({"sample_counts": {task_name: 1}}))
    raw_result_path = tmp_path / "new_results.json"
    raw_result_path.write_text("{}\n")
    (tmp_path / f"old_samples_{task_name}.jsonl").write_text(
        json.dumps({"mmmu_acc": {"parser_status": ["parsed"]}}) + "\n"
    )
    (tmp_path / f"new_samples_{task_name}.jsonl").write_text(
        json.dumps({"mmmu_acc": {"parsed_pred": ["A"]}}) + "\n"
    )

    with pytest.raises(RuntimeError, match="no valid parser status"):
        evaluator._attach_mmmu_parser_audit(
            {"raw_result_path": str(raw_result_path), "result_path": str(result_path)}
        )


def test_checkpoint_contract_accepts_only_matching_realized_anymodel(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["AnyModel"],
        base_architecture="Qwen3_5ForConditionalGeneration",
        block_configs=_homogeneous_qwen_block_configs(),
    )
    config_path.write_text(json.dumps(config) + "\n")

    vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config.pop("block_configs")
    config_path.write_text(json.dumps(config) + "\n")
    with pytest.raises(ValueError, match="cannot prove.*homogeneous"):
        vlm_model.verify_checkpoint(checkpoint_path, profile="VLM benchmark")

    config["block_configs"] = _homogeneous_qwen_block_configs()
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


def test_checkpoint_contract_routes_heterogeneous_anymodel_to_vllm(tmp_path):
    checkpoint_path = _write_checkpoint(tmp_path)
    config_path = checkpoint_path / "config.json"
    config = json.loads(config_path.read_text())
    config.update(
        architectures=["AnyModel"],
        base_architecture="Qwen3_5ForConditionalGeneration",
        block_configs=_homogeneous_qwen_block_configs(),
    )
    config["block_configs"][19]["subblock_configs"][0]["num_query_heads"] = 6
    config["text_config"]["per_layer_config"] = {
        "19": {"num_attention_heads": 6, "num_key_value_heads": 2}
    }
    config_path.write_text(json.dumps(config) + "\n")

    with pytest.raises(ValueError, match="native qwen3_5 backend cannot load"):
        vlm_model.verify_checkpoint(
            checkpoint_path,
            profile="VLM benchmark",
            model_backend="qwen3_5",
        )

    vlm_model.verify_checkpoint(
        checkpoint_path,
        profile="VLM benchmark",
        model_backend="vllm",
    )


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
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.setattr(tasks, "_lmms_eval_root", lambda: lmms_root)
    monkeypatch.setattr(preflight.importlib.util, "find_spec", lambda _name: object())
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


def _write_fake_mmmu_artifacts(result_path: Path) -> Path:
    result_path.write_text(json.dumps({"sample_counts": {suites.task_name("mmmu_val"): 1}}) + "\n")
    raw_result_path = result_path.parent / "run_results.json"
    raw_result_path.write_text("{}\n")
    sample_path = result_path.parent / f"run_samples_{suites.task_name('mmmu_val')}.jsonl"
    sample_path.write_text(json.dumps({"mmmu_acc": {"parser_status": ["parsed"]}}) + "\n")
    return raw_result_path


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


def test_short_profile_materializes_pinned_tasks_and_native_qwen_backend(
    monkeypatch, tmp_path, capsys
):
    model = _write_checkpoint(tmp_path)
    source_tasks = ("realworldqa", "mmmu_val")
    lmms_root = _write_lmms_tasks(tmp_path, source_tasks)
    _use_offline_fakes(monkeypatch, lmms_root)
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    output = tmp_path / "results"
    calls = []

    def fake_runner(checkpoint_path, *, output_root, settings):
        calls.append(
            {
                "checkpoint": checkpoint_path,
                "output_root": output_root,
                "settings": settings,
            }
        )
        result_path = output_root / "result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        raw_result_path = _write_fake_mmmu_artifacts(result_path)
        return {
            "attempt": len(calls),
            "metrics": {"accuracy": len(calls) / 10},
            "output_root": str(output_root),
            "raw_result_path": str(raw_result_path),
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
    report = result["preflight"]
    generated = json.loads(
        (output / "task_configs/modelopt_vlm_benchmark_realworldqa.yaml").read_text()
    )
    assert generated["dataset_path"].endswith(
        profile.VLM_BENCHMARK_DATASETS["realworldqa"].revision
    )
    assert generated["generation_kwargs"]["max_new_tokens"] == 16
    mmmu_text = (output / "task_configs/modelopt_vlm_benchmark_mmmu_val.yaml").read_text()
    assert '"max_new_tokens": 128' in mmmu_text
    expected_tasks = (
        "modelopt_vlm_benchmark_realworldqa",
        "modelopt_vlm_benchmark_mmmu_val",
    )
    assert [call["checkpoint"] for call in calls] == [model, model]
    assert [call["output_root"] for call in calls] == [
        output / "short-repetition-1",
        output / "short-repetition-2",
    ]
    assert all(call["settings"]["tasks"] == ",".join(expected_tasks) for call in calls)
    assert [run["attempt"] for run in result["runs"]] == [1, 2]
    settings = calls[0]["settings"]
    assert settings["model"] == "qwen3_5"
    assert report["backend_limitations"] == []
    assert report["output_budget_contract"] == {
        "mmmu_val": {
            "adapter": "qwen3_5",
            "effective_max_new_tokens": 128,
            "limitation": None,
            "requested_max_new_tokens": 128,
            "resolution": "task_max_new_tokens_overrides_adapter_default",
        },
        "realworldqa": {
            "adapter": "qwen3_5",
            "effective_max_new_tokens": 16,
            "limitation": None,
            "requested_max_new_tokens": 16,
            "resolution": "task_max_new_tokens_overrides_adapter_default",
        },
    }


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
        result_path.parent.mkdir(parents=True, exist_ok=True)
        raw_result_path = _write_fake_mmmu_artifacts(result_path)
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
    [("checkpoint", 4), ("artifact", 3), ("result", 3), ("profile", 4)],
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
        result_path.parent.mkdir(parents=True, exist_ok=True)
        raw_result_path = _write_fake_mmmu_artifacts(result_path)
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
        (output / "short-repetition-1" / "attempt" / "run_results.json").unlink()
    elif corruption == "result":
        (output / "short-repetition-1" / "attempt" / "summary.json").unlink()
    else:
        original_backend_policy = preflight._backend_policy

        def changed_backend_policy(profile_contract):
            return {
                **original_backend_policy(profile_contract),
                "attention_implementation": "eager",
            }

        monkeypatch.setattr(preflight, "_backend_policy", changed_backend_policy)

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
    assert policy["generation"] == {
        "enable_thinking": False,
        "temperature": 0,
        "do_sample": False,
    }
    assert suites.execution_policy("full", timeout_seconds=None)["limit"] is None
    assert suites.execution_policy("full-v1", timeout_seconds=None)["limit"] is None


@pytest.mark.parametrize("alias", suites.DEPRECATED_SUITE_ALIASES)
def test_deprecated_suite_alias_records_the_canonical_identity(monkeypatch, tmp_path, alias):
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
            alias,
            "--hf-home",
            str(hf_home),
        ]
    )

    with pytest.warns(FutureWarning, match="is deprecated"):
        prepared = preflight.prepare(args)

    assert prepared.suite == suites.TASK_PREFIX100_REPEAT2_SUITE
    assert prepared.report["suite"] == suites.TASK_PREFIX100_REPEAT2_SUITE


def test_versioned_profile_contracts_pin_backends_and_fingerprints():
    profiles = {name: contracts.load_profile(name) for name in contracts.PROFILE_NAMES}
    for name in profiles:
        composition = json.loads((contracts._PROFILE_ROOT / f"{name}.json").read_text())
        assert set(composition) == {
            "schema",
            "name",
            "sample_set",
            "backend_profile",
            "evaluator_profile",
        }
    assert {name: contract.fingerprint for name, contract in profiles.items()} == {
        "short-v1": "8286a094c3cfb5c2608a6e1469d6525bb1c4be11a7ab8789ed97dd249d7bea71",
        "short-native-v1": "b15054c251af54a5298233b1c01a3babf76c4281d055469fd26b76129c34f258",
        "core-3_344-examples_r1-native": "2017656d093de7d95d25c7e34241b1d708150157f0c4e6a0bf6bd48649c2191a",
        "core-3_344-examples_r1-vllm": "859908fdb32b6bcaddb5400cd4430f4c9026264db38c7a8b56a98f42109c1f78",
        "core-3_24-examples_r1-native": "0e51e27d57e27f0c5e4943d077308766387fa739b2d1c413b7b951327358cefc",
        "core-3_24-examples_r1-vllm": "9c68168f05003e695258dc119351610b4e98bafb3e4f3e773c4e64ce5d17835a",
        "short-all-native-v1": "9d7334371316a2a7774ee7e520ce0fc57e3c42ecfd7749ccd217e02ab59b6ee3",
        "judge-free-8_690-examples_r1-native": "78457702288ba2d9d7b903366f7030302936377690b0ec37a0704e3eda8fd851",
        "full-v1": "680483a7e2eceeab82a5e0b2767cc751f1951f58ffedc0aa190481d6ec978307",
        "core-3_full_r1-native": "976efbd056fecb686e64b912ed50251b1c16e5efe1d3ae3d179f208cc587c0a7",
        "core-3_full_r1-vllm": "40fd44fbb4812bd927d3f82e33d6eddec5b4641736c65c48250b8ea77acaec81",
    }

    current_short = profiles["core-3_344-examples_r1-native"]
    smoke = profiles["core-3_24-examples_r1-native"]
    materialized_short = profiles["core-3_344-examples_r1-vllm"]
    materialized_smoke = profiles["core-3_24-examples_r1-vllm"]
    assert current_short.manifest["lmms_eval_revision"] == checkpoint.LMMS_EVAL_REVISION
    assert current_short.manifest["backend"]["name"] == "qwen3_5"
    assert materialized_short.manifest["backend"]["name"] == "vllm"
    assert materialized_short.sample_set == current_short.sample_set
    assert materialized_short.backend_profile != current_short.backend_profile
    assert materialized_short.evaluator_profile == current_short.evaluator_profile
    assert materialized_short.exact_rows == current_short.exact_rows
    assert materialized_smoke.manifest["backend"]["name"] == "vllm"
    assert materialized_smoke.manifest["backend"]["enforce_eager"] is True
    assert materialized_smoke.exact_rows == smoke.exact_rows
    assert suites.manifest_selected_rows(current_short.exact_rows) == 344
    assert suites.manifest_selected_rows(smoke.exact_rows) == 24
    assert (
        suites.manifest_selected_rows(profiles["judge-free-8_690-examples_r1-native"].exact_rows)
        == 690
    )
    assert profiles["full-v1"].exact_rows is None


@pytest.mark.parametrize(
    ("name", "backend"),
    [
        (
            "core-3_full_r1-native",
            {
                "attention_implementation": "sdpa",
                "enable_thinking": False,
                "name": "qwen3_5",
            },
        ),
        (
            "core-3_full_r1-vllm",
            {"enable_thinking": False, "name": "vllm", "reasoning_parser": "qwen3"},
        ),
    ],
)
def test_core3_full_teacher_profiles_pin_paired_population_and_runtime(name, backend):
    contract = contracts.load_profile(name)

    assert contract.manifest["model"] == {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    }
    assert contract.manifest["lmms_eval_revision"] == checkpoint.LMMS_EVAL_REVISION
    assert contract.manifest["backend"] == backend
    assert contract.manifest["generation"] == {"do_sample": False, "temperature": 0}
    assert contract.manifest["seed"] == 42
    assert contract.manifest["repetitions"] == 1
    assert contract.manifest["batch_size"] == 1
    assert contract.manifest["selection"] == "all"
    assert contract.exact_rows is None
    assert {
        task: entry["population_rows"] for task, entry in contract.manifest["tasks"].items()
    } == {"realworldqa": 765, "mmmu_val": 900, "mvbench": 4000}
    assert contract.manifest["tasks"]["mvbench"]["leaf_populations"] == dict.fromkeys(
        suites.MVBENCH_LEAF_TASKS, 200
    )
    assert {
        task: entry["dataset_revision"] for task, entry in contract.manifest["tasks"].items()
    } == {
        task: profile.VLM_BENCHMARK_DATASETS[task].revision
        for task in ("realworldqa", "mmmu_val", "mvbench")
    }


@pytest.mark.parametrize(
    ("name", "field", "value", "message"),
    [
        (
            "core-3_full_r1-native",
            "backend",
            {"enable_thinking": False, "name": "vllm", "reasoning_parser": "qwen3"},
            "backend profile differs",
        ),
        (
            "core-3_full_r1-vllm",
            "model",
            {"repository": "Qwen/Qwen3.5-0.8B", "revision": "different"},
            "model pin differs",
        ),
        ("core-3_full_r1-native", "population", 764, "population differs"),
    ],
)
def test_core3_full_teacher_profiles_reject_contract_overrides(
    monkeypatch, tmp_path, name, field, value, message
):
    shutil.copytree(contracts._PROFILE_ROOT, tmp_path, dirs_exist_ok=True)
    if field == "backend":
        manifest_path = tmp_path / "backends" / "qwen-3.5-native_r1.json"
    else:
        manifest_path = tmp_path / "sample_sets" / "core-3_full_r1.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "population":
        manifest["tasks"]["realworldqa"]["population_rows"] = value
    elif field == "backend":
        manifest["settings"] = value
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(contracts, "_PROFILE_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match=message):
        contracts.load_profile(name)


def test_audited_profile_rejects_rows_that_drift_from_systematic_selection(monkeypatch, tmp_path):
    shutil.copytree(contracts._PROFILE_ROOT, tmp_path, dirs_exist_ok=True)
    manifest_path = tmp_path / "sample_sets" / "core-3_344-examples_r1.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tasks"]["realworldqa"]["rows"][0]["source_row_index"] = 6
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(contracts, "_PROFILE_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="rows differ from its sampling audit"):
        contracts.load_profile("core-3_344-examples_r1-native")


def test_short_all_native_profile_builds_grouped_and_single_selectors(tmp_path):
    contract = contracts.load_profile("judge-free-8_690-examples_r1-native")
    exact_rows = contract.exact_rows
    assert exact_rows is not None
    validated = suites.validate_exact_rows_manifest(
        exact_rows,
        expected_revision=checkpoint.LMMS_EVAL_REVISION,
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
        def __init__(self, size, *, columns=None, rows=None):
            self.size = size
            self.columns = columns or {}
            self.rows = rows or {}

        def __len__(self):
            return self.size

        def __getitem__(self, index):
            if isinstance(index, str):
                return self.columns[index]
            return self.rows.get(index, {})

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
    with pytest.raises(ValueError, match="source population drifted"):
        selectors.select_modelopt_vlm_benchmark_video_mmmu_adaptation(Documents(299))
    with pytest.raises(ValueError, match="source population drifted"):
        selectors.select_modelopt_vlm_benchmark_realworldqa(Documents(764))
    with pytest.raises(ValueError, match="source population drifted"):
        selectors.select_modelopt_vlm_benchmark_mvbench_action_sequence(Documents(199))

    mmmu_task = tasks_manifest["mmmu_val"]
    mmmu_ids = [
        f"validation_{stratum['name']}_{index + 1}"
        for stratum in mmmu_task["selection"]["strata"]
        for index in range(stratum["population_rows"])
    ]
    mmmu_rows = {
        row["source_row_index"]: {"id": row["source_sample_id"]} for row in mmmu_task["rows"]
    }
    mmmu_documents = Documents(900, columns={"id": mmmu_ids}, rows=mmmu_rows)
    assert selectors.select_modelopt_vlm_benchmark_mmmu_val(mmmu_documents) == [
        row["source_row_index"] for row in mmmu_task["rows"]
    ]
    mmmu_ids[0] = mmmu_ids[30]
    with pytest.raises(ValueError, match="source strata drifted"):
        selectors.select_modelopt_vlm_benchmark_mmmu_val(mmmu_documents)

    assert callable(selectors.select_modelopt_vlm_benchmark_videomme)


@pytest.mark.parametrize("task", ["videomme", "mlvu_dev", "perceptiontest_val_mc"])
@pytest.mark.parametrize(
    ("drift", "expected_rank", "observed_strata"),
    [
        (None, 1, ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"]),
        ("selected stratum", 1, ["alpha|kind", "beta|kind", "beta|kind", "alpha|kind"]),
        ("local rank", 0, ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"]),
    ],
)
def test_audited_selector_checks_selected_stratum_and_local_rank(
    tmp_path, task, drift, expected_rank, observed_strata
):
    expected_stratum = "alpha" if task == "mlvu_dev" else "alpha|kind"
    other_stratum = "beta" if task == "mlvu_dev" else "beta|kind"
    upstream_id = "video:q" if task == "perceptiontest_val_mc" else "q"
    manifest = {
        "tasks": {
            task: {
                "rows": [
                    {
                        "source_row_index": 2,
                        "source_sample_id": f"{task}:2",
                        "sampling_stratum": expected_stratum,
                        "source_stratum_index": expected_rank,
                        "upstream_sample_id": upstream_id,
                    }
                ],
                "selection": {
                    "population_rows": 4,
                    "strata": [
                        {"name": expected_stratum, "population_rows": 2},
                        {"name": other_stratum, "population_rows": 2},
                    ],
                },
            }
        }
    }
    tasks._write_quick_selection_module(tmp_path, manifest)
    spec = importlib.util.spec_from_file_location(
        f"sampling_position_{task}_{drift}", tmp_path / "modelopt_quick_selection.py"
    )
    assert spec is not None and spec.loader is not None
    selectors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selectors)

    class Documents:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            if isinstance(index, str):
                if task == "mlvu_dev":
                    return [value.split("|", 1)[0] for value in observed_strata]
                column = 0 if index in {"duration", "area"} else 1
                return [value.split("|", 1)[column] for value in observed_strata]
            if task == "perceptiontest_val_mc":
                return {"video_name": "video", "question_id": "q"}
            return {"question_id": "q"}

        def select(self, indices):
            return list(indices)

    selector = getattr(selectors, f"select_{suites.task_name(task)}")
    if drift is None:
        assert selector(Documents()) == [2]
    else:
        with pytest.raises(ValueError, match="source sampling positions drifted"):
            selector(Documents())


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


@pytest.mark.parametrize("name", ["core-3_full_r1-native", "core-3_full_r1-vllm"])
def test_core3_full_teacher_profiles_preserve_backend_prompt_policy(monkeypatch, tmp_path, name):
    model, hf_home = _write_core3_teacher_snapshot(tmp_path)
    lmms_root = _write_lmms_tasks(tmp_path, ("mmmu_val",))
    _use_offline_fakes(monkeypatch, lmms_root)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            name,
            "--profile-task",
            "mmmu_val",
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
        configured_tasks=("modelopt_vlm_benchmark_mmmu_val",),
        prepared=prepared,
    )

    assert prepared.report["model_pin"] == {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    }
    assert prepared.report["profile_population_rows"] == {"mmmu_val": 900}
    assert prepared.report["output_budget_contract"]["mmmu_val"]["effective_max_new_tokens"] == 128
    if name == "core-3_full_r1-native":
        assert settings["model"] == "qwen3_5"
        assert prepared.report["backend_limitations"] == []
    else:
        assert settings["model"] == "vllm"
        assert prepared.report["output_budget_contract"]["mmmu_val"] == {
            "adapter": "vllm",
            "effective_max_new_tokens": 128,
            "limitation": (
                "the pinned generic vLLM adapter treats its model-level max_new_tokens as a floor"
            ),
            "requested_max_new_tokens": 128,
            "resolution": "max(task_max_new_tokens, model_max_new_tokens_floor=1)",
        }
        assert prepared.report["backend_limitations"] == [
            "generic vLLM video messages do not preserve native Qwen 3.5 timestamps",
            "pinned generic vLLM max_new_tokens is a model-level lower bound",
        ]


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


def test_smoke_profile_generates_only_manifest_backed_mvbench_leaves(monkeypatch, tmp_path):
    model, hf_home = _full_inputs(monkeypatch, tmp_path)
    args = evaluation._build_parser().parse_args(
        [
            "--checkpoint",
            str(model),
            "--output-dir",
            str(tmp_path / "results"),
            "--profile",
            "core-3_24-examples_r1-native",
            "--hf-home",
            str(hf_home),
        ]
    )

    prepared = preflight.prepare(args)
    tasks_root, configured_tasks = tasks.prepare(
        args.output_dir,
        suite=prepared.suite,
        source_tasks=prepared.source_tasks,
        profile_task_leaves=prepared.profile_task_leaves,
        dataset_snapshots=prepared.dataset_snapshots,
        quick_manifest=prepared.quick_manifest,
    )

    assert configured_tasks == (
        "modelopt_vlm_benchmark_realworldqa",
        "modelopt_vlm_benchmark_mmmu_val",
        "modelopt_vlm_benchmark_mvbench",
    )
    group = json.loads((tasks_root / "modelopt_vlm_benchmark_mvbench.yaml").read_text())
    assert group["task"] == ["modelopt_vlm_benchmark_mvbench_action_sequence"]
    assert (tasks_root / "modelopt_vlm_benchmark_mvbench_action_sequence.yaml").is_file()
    assert not (tasks_root / "modelopt_vlm_benchmark_mvbench_egocentric_navigation.yaml").exists()

    spec = importlib.util.spec_from_file_location(
        "smoke_selectors", tasks_root / "modelopt_quick_selection.py"
    )
    assert spec is not None and spec.loader is not None
    selectors = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selectors)
    assert hasattr(selectors, "select_modelopt_vlm_benchmark_mvbench_action_sequence")
    assert not hasattr(selectors, "select_modelopt_vlm_benchmark_mvbench_egocentric_navigation")


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
            result_path.write_text(
                json.dumps(
                    {
                        "sample_counts": {"realworldqa": 100, "mmmu_val": 100},
                        "mmmu_parser_audit": {
                            "sample_count": 100,
                            "status_counts": {"parsed": 90, "fallback_random": 10},
                        },
                    }
                )
            )
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
    assert summary["sample_counts"] == {"mmmu_val": 200, "realworldqa": 200}
    assert summary["mmmu_parser_audit"] == {
        "sample_count": 200,
        "status_counts": {"fallback_random": 20, "parsed": 180},
    }

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


def test_mmmu_adapter_labels_parser_fallback_without_changing_prediction(monkeypatch, tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    lmms_root = _write_lmms_tasks(tmp_path, ("mmmu_val",))
    monkeypatch.setattr(tasks, "_lmms_eval_root", lambda: lmms_root)
    tasks_root, _ = tasks.prepare(
        tmp_path / "results",
        suite="short",
        source_tasks=("mmmu_val",),
        dataset_snapshots={"mmmu_val": snapshot},
        quick_manifest=None,
    )

    upstream = ModuleType("lmms_eval.tasks.mmmu.utils")

    def get_multi_choice_info(options):
        choices = [chr(ord("A") + index) for index in range(len(options))]
        return dict(zip(choices, options, strict=True)), choices

    def parse_multi_choice_response(response, all_choices, _index_to_answer):
        return random.choice(all_choices) if response == "unparseable" else "A"

    def mmmu_process_results(document, results):
        if document["question_type"] == "multiple-choice":
            index_to_answer, choices = get_multi_choice_info(json.loads(document["options"]))
            parsed = [
                parse_multi_choice_response(response, choices, index_to_answer)
                for response in results
            ]
        else:
            parsed = [""] * len(results)
        accuracy = {"parsed_pred": parsed}
        return {"mmmu_acc": accuracy, "mmmu_acc_pass_at_k": accuracy}

    upstream.get_multi_choice_info = get_multi_choice_info
    upstream.parse_multi_choice_response = parse_multi_choice_response
    upstream.mmmu_process_results = mmmu_process_results
    package_modules = {
        "lmms_eval": ModuleType("lmms_eval"),
        "lmms_eval.tasks": ModuleType("lmms_eval.tasks"),
        "lmms_eval.tasks.mmmu": ModuleType("lmms_eval.tasks.mmmu"),
        "lmms_eval.tasks.mmmu.utils": upstream,
    }
    package_modules["lmms_eval.tasks.mmmu"].utils = upstream
    for name, module in package_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location(
        "modelopt_mmmu_audit", tasks_root / "modelopt_mmmu_audit.py"
    )
    assert spec is not None and spec.loader is not None
    audit_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_module)
    document = {
        "options": json.dumps(["first", "second"]),
        "question_type": "multiple-choice",
    }

    parsed = audit_module.process_results(document, ["(A)"])
    random.seed(123)
    parse_multi_choice_response("unparseable", ["A", "B"], {})
    expected_random_state = random.getstate()
    random.seed(123)
    fallback = audit_module.process_results(document, ["unparseable"])
    invalid_open = audit_module.process_results(
        {"question_type": "open"},
        ["unparseable"],
    )

    assert parsed["mmmu_acc"] == {
        "parsed_pred": ["A"],
        "parser_status": ["parsed"],
    }
    assert fallback["mmmu_acc"]["parsed_pred"][0] in {"A", "B"}
    assert fallback["mmmu_acc"]["parser_status"] == ["fallback_random"]
    assert random.getstate() == expected_random_state
    assert invalid_open["mmmu_acc"]["parser_status"] == ["invalid_open"]
    generated = (tasks_root / f"{suites.task_name('mmmu_val')}.yaml").read_text()
    assert "process_results: !function modelopt_mmmu_audit.process_results\n" in generated


def test_quick_manifest_requires_exact_pins_counts_and_leaf_balance(tmp_path):
    path = _quick_manifest(tmp_path / "quick.json")
    suites.load_quick_manifest(path)

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
class Config: task = "modelopt_vlm_benchmark_mvbench_action_sequence"
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
    def load_task_or_group(self, tasks):
        first = Task()
        if os.environ.get("FAKE_DISTINCT_DUPLICATE"):
            return {Group(): {"first": first, "second": Task()}}
        return {Group(): {"first": first, "repeat": first}}
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
        expected_populations={"modelopt_vlm_benchmark_mvbench_action_sequence": 1},
    )

    assert report["document_counts"] == {"modelopt_vlm_benchmark_mvbench_action_sequence": 1}
    assert report["media_documents"] == 1
    assert report["observed_populations"] == {"modelopt_vlm_benchmark_mvbench_action_sequence": 1}
    assert report["status"] == "passed"

    with pytest.raises(RuntimeError, match="configured task population mismatch"):
        tasks.verify_offline(
            tasks_root,
            ("modelopt_vlm_benchmark_mvbench",),
            hf_home=hf_home,
            timeout_seconds=123,
            model_name="qwen3_5",
            expected_populations={"modelopt_vlm_benchmark_mvbench_action_sequence": 2},
        )

    monkeypatch.setenv("FAKE_DISTINCT_DUPLICATE", "1")
    with pytest.raises(RuntimeError, match="distinct task objects share configured task name"):
        tasks.verify_offline(
            tasks_root,
            ("modelopt_vlm_benchmark_mvbench",),
            hf_home=hf_home,
            timeout_seconds=123,
            model_name="qwen3_5",
            expected_populations={"modelopt_vlm_benchmark_mvbench_action_sequence": 1},
        )


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
    assert "lmms-eval.git" not in requirements
    assert 'eva-decord==0.6.1; platform_system == "Linux"' in requirements.splitlines()
    assert "wandb==0.29.0" in requirements.splitlines()
    environment = json.loads(
        (checkpoint.REPOSITORY_ROOT / "examples/puzzletron/ci_environment.json").read_text()
    )
    assert environment["lmms_eval"]["commit"] == checkpoint.LMMS_EVAL_REVISION
    patch = (
        checkpoint.REPOSITORY_ROOT
        / "examples/puzzletron/patches"
        / environment["lmms_eval"]["compatibility_patch"]
    )
    assert (
        hashlib.sha256(patch.read_bytes()).hexdigest()
        == environment["lmms_eval"]["compatibility_patch_sha256"]
    )


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
