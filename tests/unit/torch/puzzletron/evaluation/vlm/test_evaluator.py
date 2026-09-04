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

"""Tests for VLM evaluation execution and result recovery."""

import hashlib
import json

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import evaluator, preflight, profile, suites
from examples.puzzletron.evaluation.vlm import run as evaluation
from tests.unit.torch.puzzletron.evaluation.vlm._test_utils import (
    _use_offline_fakes,
    _write_checkpoint,
    _write_fake_mmmu_artifacts,
    _write_lmms_tasks,
)


def test_chat_template_fingerprint_accepts_file_and_inline_content(tmp_path):
    content = "{% if messages %}{{ messages[0]['content'] }}{% endif %}"
    template_path = tmp_path / "chat_template.jinja"
    template_path.write_text(content)
    expected = hashlib.sha256(content.encode()).hexdigest()

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


def test_short_profile_preserves_default_vllm_backend(monkeypatch, tmp_path, capsys):
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
    assert settings["model"] == "vllm"
    assert report["backend_limitations"] == [
        "generic vLLM video messages do not preserve native Qwen 3.5 timestamps",
        "pinned generic vLLM max_new_tokens is a model-level lower bound",
    ]
    for task, expected_tokens in {"mmmu_val": 128, "realworldqa": 16}.items():
        budget = report["output_budget_contract"][task]
        assert budget["adapter"] == "vllm"
        assert budget["requested_max_new_tokens"] == expected_tokens
        assert budget["effective_max_new_tokens"] == expected_tokens
        assert budget["limitation"] is not None
        assert budget["resolution"] == "max(task_max_new_tokens, model_max_new_tokens_floor=1)"


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
                "enforce_eager": True,
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
