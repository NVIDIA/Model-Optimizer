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

"""Tests for generated lmms-eval task adapters and selectors."""

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import contracts, preflight, profile, suites, tasks
from examples.puzzletron.evaluation.vlm import run as evaluation
from tests.unit.torch.puzzletron.evaluation.vlm.vlm_test_utils import (
    _full_inputs,
    _write_lmms_tasks,
)


def test_judge_free_profile_builds_grouped_and_single_selectors(tmp_path):
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


@pytest.mark.parametrize(
    ("task", "drift", "expected_rank", "observed_strata"),
    [
        ("videomme", None, 1, ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"]),
        ("mlvu_dev", None, 1, ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"]),
        (
            "perceptiontest_val_mc",
            None,
            1,
            ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"],
        ),
        (
            "videomme",
            "selected stratum",
            1,
            ["alpha|kind", "beta|kind", "beta|kind", "alpha|kind"],
        ),
        (
            "videomme",
            "local rank",
            0,
            ["alpha|kind", "beta|kind", "alpha|kind", "beta|kind"],
        ),
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
