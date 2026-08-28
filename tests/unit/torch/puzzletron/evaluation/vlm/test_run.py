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
import sys
from pathlib import Path
from types import ModuleType

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import preflight, profile, suites, tasks
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


def _write_checkpoint(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps(_QWEN_CONFIG) + "\n")
    return model


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
        checkpoint, "verify_lmms_eval_revision", lambda: checkpoint.LMMS_EVAL_REVISION
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
        return {"attempt": len(calls), "output_root": str(output_root)}

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
    assert result["runs"] == [
        {"attempt": 1, "output_root": str(output / "short-repetition-1")},
        {"attempt": 2, "output_root": str(output / "short-repetition-2")},
    ]
    for name in checkpoint.HUGGINGFACE_CREDENTIAL_NAMES:
        assert os.environ[name] == f"inherited-{name.lower()}"
    settings = calls[0]["settings"]
    assert settings["model"] == "vllm"
    assert settings["checkpoint_arg"] == "model"
    assert settings["reasoning_parser"] == "qwen3"
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

    preflight._verify_video_reader("short")
    with pytest.raises(RuntimeError, match="decord-compatible reader"):
        preflight._verify_video_reader("quick")


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
