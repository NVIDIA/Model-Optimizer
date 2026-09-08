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

"""Shared local fixtures for VLM evaluation unit tests."""

import json
from pathlib import Path

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import preflight, profile, suites, tasks

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
