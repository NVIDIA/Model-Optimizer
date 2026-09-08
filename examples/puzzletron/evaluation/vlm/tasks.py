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

"""Generate and validate local lmms-eval task adapters for VLM suites."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import profile, suites

__all__ = ["prepare", "task_config", "verify_offline"]

_MMVU_SMOKE_SELECTION_MODULE = "modelopt_mmvu_smoke_selection"


def _lmms_eval_root() -> Path:
    spec = importlib.util.find_spec("lmms_eval")
    locations = spec.submodule_search_locations if spec is not None else None
    if not locations:
        raise RuntimeError(
            "lmms_eval is not installed; run this command in the Puzzletron worker image"
        )
    return Path(next(iter(locations))).absolute()


def task_config(relative_path: str) -> Path:
    """Resolve one task configuration from the pinned lmms-eval installation."""
    config = _lmms_eval_root() / relative_path
    if not config.is_file():
        raise RuntimeError(f"installed lmms_eval is missing task config: {relative_path}")
    return config


def _write_task_config(
    path: Path,
    *,
    include: Path,
    task: str,
    dataset_path: Path,
    generation_kwargs: dict[str, object],
    doc_to_visual: str | None = None,
    process_docs: str | None = None,
    process_docs_module: str = "modelopt_quick_selection",
    process_results: str | None = None,
    process_results_module: str = "modelopt_mmvu_guard",
) -> None:
    values = {
        "include": str(include),
        "task": task,
        "dataset_path": str(dataset_path),
        "dataset_kwargs": {},
        "generation_kwargs": generation_kwargs,
    }
    if doc_to_visual is None and process_docs is None and process_results is None:
        checkpoint.write_generated(path, json.dumps(values, indent=2, sort_keys=True) + "\n")
        return

    lines = [f"{key}: {json.dumps(value, sort_keys=True)}" for key, value in values.items()]
    if process_docs is not None:
        lines.append(f"process_docs: !function {process_docs_module}.{process_docs}")
    if doc_to_visual is not None:
        lines.append(f"doc_to_visual: !function modelopt_video_paths.{doc_to_visual}")
    if process_results is not None:
        lines.append(f"process_results: !function {process_results_module}.{process_results}")
    checkpoint.write_generated(path, "\n".join(lines) + "\n")


def _manifest_group_leaves(
    manifest: dict[str, object], task: str, leaves: tuple[str, ...]
) -> tuple[str, ...]:
    manifest_tasks = cast("dict[str, dict[str, object]]", manifest["tasks"])
    rows = cast("list[dict[str, object]]", manifest_tasks[task]["rows"])
    selected_leaf_tasks = {cast("str", row["leaf_task"]) for row in rows}
    selected = tuple(leaf for leaf in leaves if f"{task}_{leaf}" in selected_leaf_tasks)
    if not selected:
        raise ValueError(f"exact-row manifest selects no leaves for grouped task {task}")
    return selected


def _write_quick_selection_module(tasks_root: Path, manifest: dict[str, object]) -> None:
    entries: dict[str, dict[str, object]] = {}
    functions: list[str] = []
    manifest_tasks = cast("dict[str, dict[str, object]]", manifest["tasks"])
    for task in manifest_tasks:
        task_entry = manifest_tasks[task]
        rows = cast("list[dict[str, object]]", task_entry["rows"])
        sampling = task_entry.get("selection")
        audited = cast("dict[str, object]", sampling) if isinstance(sampling, dict) else None
        if task in {"mvbench", "video_mmmu"}:
            all_leaves = (
                suites.MVBENCH_LEAF_TASKS if task == "mvbench" else suites.VIDEO_MMMU_LEAF_TASKS
            )
            leaves = _manifest_group_leaves(manifest, task, all_leaves)
            for leaf in leaves:
                leaf_task = f"{task}_{leaf}"
                selected = [row for row in rows if row["leaf_task"] == leaf_task]
                key = suites.task_name(task, leaf=leaf)
                stratum = (
                    next(
                        item
                        for item in cast("list[dict[str, object]]", audited["strata"])
                        if item["name"] == leaf
                    )
                    if audited is not None
                    else None
                )
                entries[key] = {
                    "kind": task,
                    "config": leaf,
                    "indices": [row["source_row_index"] for row in selected],
                    "source_ids": [row["source_sample_id"] for row in selected],
                    "population_rows": stratum["population_rows"] if stratum else None,
                    "strata": {leaf: stratum["population_rows"]} if stratum else None,
                    "upstream_ids": [row.get("upstream_sample_id") for row in selected],
                    "sampling_positions": None,
                }
                functions.append(
                    f"def select_{key}(documents):\n    return _select(documents, {key!r})\n"
                )
        else:
            key = suites.task_name(task)
            entries[key] = {
                "kind": task,
                "indices": [row["source_row_index"] for row in rows],
                "source_ids": [row["source_sample_id"] for row in rows],
                "population_rows": audited.get("population_rows") if audited else None,
                "strata": (
                    {
                        item["name"]: item["population_rows"]
                        for item in cast("list[dict[str, object]]", audited["strata"])
                    }
                    if audited
                    else None
                ),
                "upstream_ids": [row.get("upstream_sample_id") for row in rows],
                "sampling_positions": (
                    [(row["sampling_stratum"], row["source_stratum_index"]) for row in rows]
                    if all(
                        "sampling_stratum" in row and "source_stratum_index" in row for row in rows
                    )
                    else None
                ),
            }
            functions.append(
                f"def select_{key}(documents):\n    return _select(documents, {key!r})\n"
            )
    source = f'''"""Generated exact-row selectors for a VLM benchmark profile."""

from collections import Counter

_SELECTIONS = {entries!r}


def _column(documents, name):
    values = documents[name]
    if len(values) != len(documents):
        raise ValueError(f"exact-row manifest source column {{name}} has the wrong length")
    return values


def _source_strata(documents, kind):
    if kind == "mmmu_val":
        return [value.removeprefix("validation_").rsplit("_", 1)[0] for value in _column(documents, "id")]
    if kind == "videomme":
        return [f"{{duration}}|{{domain}}" for duration, domain in zip(
            _column(documents, "duration"), _column(documents, "domain"), strict=True
        )]
    if kind == "mlvu_dev":
        return list(_column(documents, "task_type"))
    if kind == "perceptiontest_val_mc":
        return [f"{{area}}|{{reasoning}}" for area, reasoning in zip(
            _column(documents, "area"), _column(documents, "reasoning"), strict=True
        )]
    return None


def _verify_population(documents, name, selection):
    expected_rows = selection["population_rows"]
    if expected_rows is None:
        return None
    if len(documents) != expected_rows:
        raise ValueError(
            f"exact-row manifest source population drifted for {{name}}: "
            f"{{len(documents)}} != {{expected_rows}}"
        )
    observed_strata = _source_strata(documents, selection["kind"])
    if observed_strata is not None and Counter(observed_strata) != selection["strata"]:
        raise ValueError(f"exact-row manifest source strata drifted for {{name}}")
    return observed_strata


def _stratum_ranks(strata):
    counts = Counter()
    ranks = []
    for stratum in strata:
        ranks.append(counts[stratum])
        counts[stratum] += 1
    return ranks


def _upstream_id(document, kind):
    if kind in {{"videomme", "mlvu_dev"}}:
        return str(document["question_id"])
    if kind == "perceptiontest_val_mc":
        return f"{{document['video_name']}}:{{document['question_id']}}"
    return None


def _select(documents, name):
    selection = _SELECTIONS[name]
    observed_strata = _verify_population(documents, name, selection)
    indices = selection["indices"]
    expected_positions = selection["sampling_positions"]
    if observed_strata is not None and expected_positions is not None:
        ranks = _stratum_ranks(observed_strata)
        observed_positions = [(observed_strata[index], ranks[index]) for index in indices]
        if observed_positions != expected_positions:
            raise ValueError(f"exact-row manifest source sampling positions drifted for {{name}}")
    observed = []
    for position, index in enumerate(indices):
        if index >= len(documents):
            raise ValueError(f"exact-row manifest row {{index}} is outside {{name}}")
        document = documents[index]
        if selection["kind"] == "realworldqa":
            observed.append(f"test:{{index}}")
        elif selection["kind"] == "mmmu_val":
            observed.append(str(document["id"]))
        elif selection["kind"] in {"mvbench", "video_mmmu"}:
            observed.append(f"{{selection['config']}}:{{index}}")
        else:
            observed.append(f"{{selection['kind']}}:{{index}}")
        expected_upstream_id = selection["upstream_ids"][position]
        if expected_upstream_id is not None and _upstream_id(document, selection["kind"]) != expected_upstream_id:
            raise ValueError(f"exact-row manifest upstream identity drifted for {{name}}")
    if observed != selection["source_ids"]:
        raise ValueError(f"exact-row manifest source identities drifted for {{name}}")
    return documents.select(indices)


{"".join(functions)}'''
    checkpoint.write_generated(tasks_root / "modelopt_quick_selection.py", source)


def _write_mmvu_smoke_selection_module(tasks_root: Path) -> None:
    function_name = f"select_{suites.task_name('mmvu_val')}"
    source = f'''"""Generated exact-row selector for the judge-free MMVU smoke."""

_ROWS = {suites.MMVU_SMOKE_ROWS!r}


def {function_name}(documents):
    indices = []
    for index, expected_id, expected_video_path in _ROWS:
        if index >= len(documents):
            raise ValueError(f"MMVU smoke row {{index}} is outside the pinned dataset")
        document = documents[index]
        observed = (document.get("id"), document.get("video_path"))
        if observed != (expected_id, expected_video_path):
            raise ValueError(f"MMVU smoke source identity drifted at row {{index}}")
        if document.get("question_type") != "multiple-choice":
            raise ValueError(f"MMVU smoke row {{index}} would require a judge")
        indices.append(index)
    return documents.select(indices)
'''
    checkpoint.write_generated(tasks_root / f"{_MMVU_SMOKE_SELECTION_MODULE}.py", source)


def _write_mmvu_guard(tasks_root: Path) -> None:
    checkpoint.write_generated(
        tasks_root / "modelopt_mmvu_guard.py",
        '''"""Fail closed when MMVU needs a judge but the judge did not complete."""

from lmms_eval.tasks.mmvu import utils as _upstream


class _StrictJudgeServer:
    def __init__(self, delegate):
        self._delegate = delegate

    def evaluate_binary(self, **kwargs):
        result = self._delegate.evaluate_binary(**kwargs)
        if result.get("success") and str(result.get("result", "")).strip() not in {"0", "1"}:
            raise RuntimeError("MMVU judge returned a non-binary response")
        return result


_original_get_llm_judge_server = _upstream.get_llm_judge_server


def _get_strict_llm_judge_server():
    server = _original_get_llm_judge_server()
    if not isinstance(server, _StrictJudgeServer):
        server = _StrictJudgeServer(server)
        _upstream._server = server
    return server


_upstream.get_llm_judge_server = _get_strict_llm_judge_server


def process_results(document, results):
    processed = _upstream.mmvu_process_results(document, results)
    accuracy = processed.get("accuracy", {})
    if (
        document.get("question_type") != "multiple-choice"
        and accuracy.get("correct") == 0
        and accuracy.get("eval_method") != "gpt-based"
    ):
        raise RuntimeError("MMVU judge did not complete for an open-ended answer")
    return processed
''',
    )


def _write_mmmu_audit(tasks_root: Path) -> None:
    checkpoint.write_generated(
        tasks_root / "modelopt_mmmu_audit.py",
        '''"""Expose whether the pinned MMMU parser used its random fallback."""

import ast
import random

from lmms_eval.tasks.mmmu import utils as _upstream


class _TrackedChoices(list):
    def __init__(self, choices):
        super().__init__(choices)
        self.used_random_fallback = False

    def __getitem__(self, index):
        self.used_random_fallback = True
        return super().__getitem__(index)


def _multiple_choice_status(document, response):
    index_to_answer, choices = _upstream.get_multi_choice_info(
        ast.literal_eval(document["options"])
    )
    tracked_choices = _TrackedChoices(choices)
    random_state = random.getstate()
    try:
        _upstream.parse_multi_choice_response(response, tracked_choices, index_to_answer)
    finally:
        random.setstate(random_state)
    return "fallback_random" if tracked_choices.used_random_fallback else "parsed"


def process_results(document, results):
    processed = _upstream.mmmu_process_results(document, results)
    accuracy = processed.get("mmmu_acc")
    if not isinstance(accuracy, dict):
        raise RuntimeError("MMMU result is missing its per-sample accuracy leaf")
    parsed_predictions = accuracy.get("parsed_pred")
    if not isinstance(parsed_predictions, list) or len(parsed_predictions) != len(results):
        raise RuntimeError("MMMU result has invalid parsed predictions")
    if document.get("question_type") == "multiple-choice":
        statuses = [
            _multiple_choice_status(document, response)
            for response in results
        ]
    else:
        statuses = [
            "parsed_open" if str(prediction).strip() else "invalid_open"
            for prediction in parsed_predictions
        ]
    accuracy["parser_status"] = statuses
    return processed
''',
    )


def _write_video_path_adapter(tasks_root: Path) -> None:
    alias_root = tasks_root / "video_path_aliases"
    checkpoint.write_generated(
        tasks_root / "modelopt_video_paths.py",
        f'''"""Normalize upstream video paths for the pinned native Qwen 3.5 backend."""

import hashlib
import os
from pathlib import Path

from lmms_eval.tasks.perceptiontest.val import utils as _perceptiontest
from lmms_eval.tasks.videomme import utils as _videomme

_ALIAS_ROOT = Path({str(alias_root)!r})
_NATIVE_SUFFIXES = {{".mp4", ".avi", ".mov"}}
_ALIASABLE_SUFFIXES = _NATIVE_SUFFIXES | {{".mkv"}}


def _normalize(paths):
    normalized = []
    for raw_path in paths:
        source = Path(raw_path).expanduser().absolute()
        source.stat()
        if source.suffix in _NATIVE_SUFFIXES:
            normalized.append(str(source))
            continue
        if source.suffix.lower() not in _ALIASABLE_SUFFIXES:
            raise ValueError(f"unsupported Qwen 3.5 video suffix: {{source.suffix}}")
        _ALIAS_ROOT.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(os.fsencode(source)).hexdigest()
        alias = _ALIAS_ROOT / f"{{digest}}.mp4"
        if alias.exists() or alias.is_symlink():
            if not os.path.samefile(alias, source):
                raise FileExistsError(f"video alias collision: {{alias}}")
        else:
            alias.symlink_to(source)
        normalized.append(str(alias))
    return normalized


def videomme_doc_to_visual(document):
    return _normalize(_videomme.videomme_doc_to_visual(document))


def perceptiontest_doc_to_visual(document):
    return _normalize(_perceptiontest.perceptiontest_val_doc_to_visual(document))
''',
    )


def verify_offline(
    tasks_root: Path,
    configured_tasks: tuple[str, ...],
    *,
    hf_home: Path,
    timeout_seconds: float,
    model_name: str = "vllm",
    expected_populations: dict[str, int] | None = None,
) -> dict[str, object]:
    """Instantiate every generated task with network access disabled."""
    script = """
import json
import os
import sys
from pathlib import Path

from lmms_eval.tasks import TaskManager

model_name, root, expected_json, *tasks = sys.argv[1:]
expected_populations = json.loads(expected_json)
credential_names = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGING_FACE_HUB_TOKEN")
if inherited := [name for name in credential_names if name in os.environ]:
    raise RuntimeError(f"offline task preflight inherited Hub credentials: {inherited}")
manager = TaskManager(include_path=root, model_name=model_name)
loaded = manager.load_task_or_group(tasks)
loaded_names = {getattr(key, "group_name", key) for key in loaded}
if loaded_names != set(tasks):
    raise RuntimeError(
        f"configured task identity mismatch: {sorted(loaded_names)} != {sorted(tasks)}"
    )


def task_objects(value):
    if isinstance(value, dict):
        for nested in value.values():
            yield from task_objects(nested)
    else:
        yield value


image_tasks = {
    "modelopt_vlm_benchmark_realworldqa",
    "modelopt_vlm_benchmark_mmmu_val",
}
media_documents = 0
document_counts = {}
seen_task_objects = set()
for task in task_objects(loaded):
    task_object_id = id(task)
    if task_object_id in seen_task_objects:
        continue
    seen_task_objects.add(task_object_id)
    task_name = task.config.task
    if task_name in document_counts:
        raise RuntimeError(f"distinct task objects share configured task name: {task_name}")
    if task.has_test_docs():
        documents = task.test_docs()
    elif task.has_validation_docs():
        documents = task.validation_docs()
    elif task.has_training_docs():
        documents = task.training_docs()
    else:
        raise RuntimeError(f"configured task has no evaluation split: {task_name}")
    document_counts[task_name] = len(documents)
    if task_name in image_tasks:
        _ = len(documents)
        continue
    for document in documents:
        visuals = task.doc_to_visual(document)
        if not visuals:
            raise RuntimeError(f"configured media task returned no visual: {task_name}")
        for visual in visuals:
            if isinstance(visual, str):
                path = Path(visual)
                path.stat()
                if path.suffix not in {".mp4", ".avi", ".mov"}:
                    raise RuntimeError(
                        f"unsupported resolved video suffix for {task_name}: {path.suffix}"
                    )
        media_documents += 1

observed_populations = {
    task_name: document_counts.get(task_name, 0) for task_name in expected_populations
}
if observed_populations != expected_populations:
    raise RuntimeError(
        f"configured task population mismatch: {observed_populations} != {expected_populations}"
    )

print(
    json.dumps(
        {
            "configured_tasks": tasks,
            "document_counts": document_counts,
            "media_documents": media_documents,
            "observed_populations": observed_populations,
            "status": "passed",
        },
        sort_keys=True,
    )
)
"""
    env = {
        **checkpoint.credential_free_environment(os.environ),
        **checkpoint.lmms_eval_disabled_judge_environment(),
        "HF_DATASETS_OFFLINE": "1",
        "HF_HOME": str(hf_home),
        "HF_HUB_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join(
            value for value in (str(tasks_root), os.environ.get("PYTHONPATH")) if value
        ),
    }
    # A child interpreter is required to import lmms-eval in a clean offline environment. The
    # fixed interpreter/script and argument-vector invocation avoid shell parsing or interpolation.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            model_name,
            str(tasks_root),
            json.dumps(expected_populations or {}, sort_keys=True),
            *configured_tasks,
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=timeout_seconds,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip().splitlines()
        tail = "\n".join(detail[-40:]) if detail else "no stderr"
        raise RuntimeError(f"offline task/data preflight failed: {tail}")
    try:
        report = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("offline task/data preflight returned no valid report") from error
    return cast("dict[str, object]", report)


def prepare(
    output_root: Path,
    *,
    suite: str,
    source_tasks: tuple[str, ...] | None = None,
    profile_task_leaves: tuple[str, ...] | None = None,
    dataset_snapshots: dict[str, Path],
    quick_manifest: dict[str, object] | None,
) -> tuple[Path, tuple[str, ...]]:
    """Generate the exact local task set selected by a VLM suite."""
    tasks_root = output_root.expanduser().absolute() / "task_configs"
    tasks_root.mkdir(parents=True, exist_ok=True)
    if source_tasks is None:
        source_tasks = suites.source_tasks(suite)
    if quick_manifest is not None:
        _write_quick_selection_module(tasks_root, quick_manifest)
    if suite == "mmvu-smoke":
        _write_mmvu_smoke_selection_module(tasks_root)
    if suite == "full":
        _write_mmvu_guard(tasks_root)
    if "mmmu_val" in source_tasks:
        _write_mmmu_audit(tasks_root)
    if set(source_tasks) & {"videomme", "perceptiontest_val_mc"}:
        _write_video_path_adapter(tasks_root)

    configured_tasks: list[str] = []
    for task in source_tasks:
        if task == "mvbench":
            configured_tasks.append(
                _write_task_group(
                    tasks_root,
                    task=task,
                    leaves=profile_task_leaves or suites.MVBENCH_LEAF_TASKS,
                    include_root="tasks/mvbench/mvbench_{}.yaml",
                    dataset_path=dataset_snapshots[task],
                    quick_manifest=quick_manifest,
                )
            )
            continue
        if task == "video_mmmu":
            configured_tasks.append(
                _write_task_group(
                    tasks_root,
                    task=task,
                    leaves=profile_task_leaves or suites.VIDEO_MMMU_LEAF_TASKS,
                    include_root="tasks/videommmu/{}.yaml",
                    dataset_path=dataset_snapshots[task],
                    quick_manifest=quick_manifest,
                )
            )
            continue
        configured_tasks.append(
            _write_single_task(
                tasks_root,
                suite=suite,
                task=task,
                dataset_path=dataset_snapshots[task],
                quick_manifest=quick_manifest,
            )
        )
    return tasks_root, tuple(configured_tasks)


def _write_task_group(
    tasks_root: Path,
    *,
    task: str,
    leaves: tuple[str, ...],
    include_root: str,
    dataset_path: Path,
    quick_manifest: dict[str, object] | None,
) -> str:
    if quick_manifest is not None:
        leaves = _manifest_group_leaves(quick_manifest, task, leaves)
    generated_leaves = []
    for leaf in leaves:
        leaf_task = suites.task_name(task, leaf=leaf)
        generated_leaves.append(leaf_task)
        _write_task_config(
            tasks_root / f"{leaf_task}.yaml",
            include=task_config(include_root.format(leaf)),
            task=leaf_task,
            dataset_path=dataset_path,
            generation_kwargs=suites.generation_kwargs(task),
            process_docs=f"select_{leaf_task}" if quick_manifest is not None else None,
        )
    group_task = suites.task_name(task)
    checkpoint.write_generated(
        tasks_root / f"{group_task}.yaml",
        json.dumps({"group": group_task, "task": generated_leaves}, indent=2, sort_keys=True)
        + "\n",
    )
    return group_task


def _write_single_task(
    tasks_root: Path,
    *,
    suite: str,
    task: str,
    dataset_path: Path,
    quick_manifest: dict[str, object] | None,
) -> str:
    configured_task = suites.task_name(task)
    process_docs = None
    process_docs_module = "modelopt_quick_selection"
    if quick_manifest is not None or (suite == "mmvu-smoke" and task == "mmvu_val"):
        process_docs = f"select_{configured_task}"
    if suite == "mmvu-smoke" and task == "mmvu_val":
        process_docs_module = _MMVU_SMOKE_SELECTION_MODULE
    doc_to_visual = {
        "videomme": "videomme_doc_to_visual",
        "perceptiontest_val_mc": "perceptiontest_doc_to_visual",
    }.get(task)
    _write_task_config(
        tasks_root / f"{configured_task}.yaml",
        include=task_config(profile.VLM_BENCHMARK_DATASETS[task].task_config),
        task=configured_task,
        dataset_path=dataset_path,
        generation_kwargs=suites.generation_kwargs(task),
        doc_to_visual=doc_to_visual,
        process_docs=process_docs,
        process_docs_module=process_docs_module,
        process_results=(
            "process_results"
            if task == "mmmu_val" or (suite == "full" and task == "mmvu_val")
            else None
        ),
        process_results_module=(
            "modelopt_mmmu_audit" if task == "mmmu_val" else "modelopt_mmvu_guard"
        ),
    )
    return configured_task
