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

"""Versioned, executable VLM evaluation profile contracts."""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import profile

__all__ = [
    "PROFILE_NAMES",
    "SHORT_PROFILE_NAMES",
    "ProfileContract",
    "load_profile",
    "warn_deprecated_profile",
]

_PROFILE_SCHEMA = "modelopt.vlm-evaluation-profile/v2"
_SAMPLE_SET_SCHEMA = "modelopt.vlm-sample-set/v1"
_BACKEND_PROFILE_SCHEMA = "modelopt.vlm-backend-profile/v1"
_EVALUATOR_PROFILE_SCHEMA = "modelopt.vlm-evaluator-profile/v1"
_PROFILE_ROOT = Path(__file__).with_name("profiles")
PROFILE_NAMES = (
    "short-v1",
    "short-native-v1",
    "core-3_344-examples_r1-native",
    "core-3_344-examples_r1-vllm",
    "core-3_24-examples_r1-native",
    "core-3_24-examples_r1-vllm",
    "short-all-native-v1",
    "judge-free-8_690-examples_r1-native",
    "full-v1",
    "core-3_full_r1-native",
    "core-3_full_r1-vllm",
)
_PROFILE_COMPONENTS = {
    # short-v1, short-native-v1, short-all-native-v1, and full-v1 are temporary
    # compatibility compositions. They preserve the exact rows, backend, and
    # evaluator revision of names that predate component profiles. New runs
    # should use descriptive profiles. Remove these entries and their legacy
    # sample sets after downstream callers have migrated.
    "short-v1": (
        "core-3_344-examples_legacy-r1",
        "qwen-3.5-vllm_r1",
        "lmms-eval-modelopt_r1",
    ),
    "short-native-v1": (
        "core-3_344-examples_legacy-r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_344-examples_r1-native": (
        "core-3_344-examples_r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_344-examples_r1-vllm": (
        "core-3_344-examples_r1",
        "anymodel-vllm_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_24-examples_r1-native": (
        "core-3_24-examples_r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_24-examples_r1-vllm": (
        "core-3_24-examples_r1",
        "anymodel-vllm-eager_r1",
        "lmms-eval-modelopt_r1",
    ),
    "short-all-native-v1": (
        "judge-free-8_690-examples_legacy-r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "judge-free-8_690-examples_r1-native": (
        "judge-free-8_690-examples_r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "full-v1": (
        "judge-free-8_full_legacy-r1",
        "qwen-3.5-vllm_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_full_r1-native": (
        "core-3_full_r1",
        "qwen-3.5-native_r1",
        "lmms-eval-modelopt_r1",
    ),
    "core-3_full_r1-vllm": (
        "core-3_full_r1",
        "qwen-3.5-vllm_r1",
        "lmms-eval-modelopt_r1",
    ),
}
_DEPRECATED_PROFILE_REPLACEMENTS = {
    "short-v1": "core-3_344-examples_r1-vllm",
    "short-native-v1": "core-3_344-examples_r1-native",
    "short-all-native-v1": "judge-free-8_690-examples_r1-native",
    "full-v1": None,
}
_CORE_3_TASKS = ("realworldqa", "mmmu_val", "mvbench")
_JUDGE_FREE_8_TASKS = (
    "realworldqa",
    "mmmu_val",
    "mvbench",
    "video_mmmu",
    "videomme",
    "longvideobench_val_v",
    "mlvu_dev",
    "perceptiontest_val_mc",
)
_SAMPLE_SET_TASKS = {
    "core-3_344-examples_legacy-r1": _CORE_3_TASKS,
    "core-3_344-examples_r1": _CORE_3_TASKS,
    "core-3_24-examples_r1": _CORE_3_TASKS,
    "judge-free-8_690-examples_legacy-r1": _JUDGE_FREE_8_TASKS,
    "judge-free-8_690-examples_r1": _JUDGE_FREE_8_TASKS,
    "judge-free-8_full_legacy-r1": tuple(
        task for task in profile.VLM_BENCHMARK_TASKS if task != "mmvu_val"
    ),
    "core-3_full_r1": _CORE_3_TASKS,
}
_SAMPLE_SET_SELECTIONS = {
    "core-3_344-examples_legacy-r1": "exact-rows",
    "core-3_344-examples_r1": "exact-rows",
    "core-3_24-examples_r1": "exact-rows",
    "judge-free-8_690-examples_legacy-r1": "exact-rows",
    "judge-free-8_690-examples_r1": "exact-rows",
    "judge-free-8_full_legacy-r1": "all",
    "core-3_full_r1": "all",
}
SHORT_PROFILE_NAMES = tuple(
    name
    for name in PROFILE_NAMES
    if _SAMPLE_SET_SELECTIONS[_PROFILE_COMPONENTS[name][0]] == "exact-rows"
)
_BACKEND_SETTINGS = {
    "qwen-3.5-vllm_r1": {
        "enable_thinking": False,
        "name": "vllm",
        "reasoning_parser": "qwen3",
    },
    "qwen-3.5-native_r1": {
        "attention_implementation": "sdpa",
        "enable_thinking": False,
        "name": "qwen3_5",
    },
    "anymodel-vllm_r1": {
        "attention_config": {"flash_attn_version": 2},
        "enable_thinking": False,
        "name": "vllm",
        "reasoning_parser": "qwen3",
    },
    "anymodel-vllm-eager_r1": {
        "attention_config": {"flash_attn_version": 2},
        "enable_thinking": False,
        "enforce_eager": True,
        "name": "vllm",
        "reasoning_parser": "qwen3",
    },
}
_EVALUATOR_REVISIONS = {
    "lmms-eval-modelopt_r1": checkpoint.LMMS_EVAL_REVISION,
}
_SAMPLE_SET_MODELS = {
    "core-3_full_r1": {
        "repository": "Qwen/Qwen3.5-0.8B",
        "revision": "2fc06364715b967f1860aea9cf38778875588b17",
    }
}
_SAMPLE_SET_POPULATIONS = {"core-3_full_r1": {"realworldqa": 765, "mmmu_val": 900, "mvbench": 4000}}
_MVBENCH_LEAF_POPULATIONS = {
    "action_sequence": 200,
    "moving_count": 200,
    "action_prediction": 200,
    "episodic_reasoning": 200,
    "action_antonym": 200,
    "action_count": 200,
    "scene_transition": 200,
    "object_shuffle": 200,
    "object_existence": 200,
    "fine_grained_pose": 200,
    "unexpected_action": 200,
    "moving_direction": 200,
    "state_change": 200,
    "object_interaction": 200,
    "character_order": 200,
    "action_localization": 200,
    "counterfactual_inference": 200,
    "fine_grained_action": 200,
    "moving_attribute": 200,
    "egocentric_navigation": 200,
}
_AUDITED_SAMPLE_SETS = frozenset(
    {
        "core-3_24-examples_r1",
        "core-3_344-examples_r1",
        "judge-free-8_690-examples_r1",
    }
)
_SAMPLING_AUDIT_SCHEMA = "modelopt.vlm-sampling-audit/v1"
_SAMPLING_GENERATOR = {"name": "systematic-midpoint", "version": 1}
_SAMPLING_STRATA = {
    "realworldqa": "split",
    "mmmu_val": "subject",
    "mvbench": "leaf_task",
    "video_mmmu": "leaf_task",
    "videomme": "duration+domain",
    "longvideobench_val_v": "split",
    "mlvu_dev": "task_type",
    "perceptiontest_val_mc": "area+reasoning",
}


@dataclass(frozen=True)
class ProfileContract:
    """One validated profile manifest and its stable content identity."""

    name: str
    manifest: dict[str, object]
    fingerprint: str

    @property
    def sample_set(self) -> str:
        """Return the selected example-set contract name."""
        return cast("str", self.manifest["sample_set"])

    @property
    def backend_profile(self) -> str:
        """Return the selected model-backend contract name."""
        return cast("str", self.manifest["backend_profile"])

    @property
    def evaluator_profile(self) -> str:
        """Return the selected evaluator contract name."""
        return cast("str", self.manifest["evaluator_profile"])

    @property
    def source_tasks(self) -> tuple[str, ...]:
        """Return benchmark tasks in their declared evaluation order."""
        tasks = cast("dict[str, object]", self.manifest["tasks"])
        return tuple(tasks)

    @property
    def exact_rows(self) -> dict[str, object] | None:
        """Return the executable exact-row selector payload when the profile uses one."""
        if self.manifest["selection"] != "exact-rows":
            return None
        tasks = cast("dict[str, dict[str, object]]", self.manifest["tasks"])
        exact_rows: dict[str, object] = {
            "schema": "modelopt.vlm-benchmark-quick/v1",
            "lmms_eval_revision": self.manifest["lmms_eval_revision"],
            "tasks": {
                task: {
                    "dataset_revision": entry["dataset_revision"],
                    "rows": entry["rows"],
                    **({"selection": entry["selection"]} if "selection" in entry else {}),
                }
                for task, entry in tasks.items()
            },
        }
        if "sampling" in self.manifest:
            exact_rows["selection"] = self.manifest["sampling"]
        return exact_rows


def warn_deprecated_profile(name: str) -> None:
    """Warn when a temporary compatibility profile is selected for execution."""
    if name in _DEPRECATED_PROFILE_REPLACEMENTS:
        replacement = _DEPRECATED_PROFILE_REPLACEMENTS[name]
        guidance = (
            f"use {replacement} for new runs"
            if replacement is not None
            else "choose a descriptive profile for new runs"
        )
        warnings.warn(
            f"{name} is a deprecated compatibility profile; {guidance}. "
            "It will be removed after downstream callers migrate.",
            FutureWarning,
            stacklevel=2,
        )


def load_profile(name: str) -> ProfileContract:
    """Load a named profile after validating every executable pin."""
    if name not in PROFILE_NAMES:
        raise ValueError(f"unsupported VLM evaluation profile: {name}")
    composition = _load_component(_PROFILE_ROOT, name, _PROFILE_SCHEMA, "profile")
    expected_components = _PROFILE_COMPONENTS[name]
    observed_components = tuple(
        composition.get(key) for key in ("sample_set", "backend_profile", "evaluator_profile")
    )
    if observed_components != expected_components:
        raise RuntimeError(f"{name} profile composition differs from the runtime policy")

    sample_set_name, backend_name, evaluator_name = expected_components
    sample_set = _resolve_sample_set(sample_set_name)
    backend = _load_component(
        _PROFILE_ROOT / "backends", backend_name, _BACKEND_PROFILE_SCHEMA, "backend profile"
    )
    evaluator = _load_component(
        _PROFILE_ROOT / "evaluators",
        evaluator_name,
        _EVALUATOR_PROFILE_SCHEMA,
        "evaluator profile",
    )
    if backend.get("settings") != _BACKEND_SETTINGS[backend_name]:
        raise RuntimeError(f"{backend_name} backend profile differs from the runtime policy")
    if evaluator.get("lmms_eval_revision") != _EVALUATOR_REVISIONS[evaluator_name]:
        raise RuntimeError(f"{evaluator_name} evaluator profile differs from the runtime pin")

    manifest = {
        **composition,
        "lmms_eval_revision": evaluator["lmms_eval_revision"],
        "model_family": {
            "architecture": "Qwen3_5ForConditionalGeneration",
            "model_type": "qwen3_5",
        },
        "backend": backend["settings"],
        "preprocessing": {"fps": 2, "max_frames": 32, "video_reader": "decord"},
        "generation": {"do_sample": False, "temperature": 0},
        "seed": 42,
        "repetitions": 1,
        "batch_size": 1,
        **{
            key: value
            for key, value in sample_set.items()
            if key not in {"schema", "name", "extends"}
        },
    }
    _validate_manifest(name, manifest)
    canonical = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode()
    return ProfileContract(
        name=name,
        manifest=manifest,
        fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


def _load_component(root: Path, name: str, schema: str, label: str) -> dict[str, object]:
    """Load one named component and validate its identity envelope."""
    path = root / f"{name}.json"
    try:
        component = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"VLM evaluation {label} is unreadable: {path}") from error
    if not isinstance(component, dict):
        raise RuntimeError(f"VLM evaluation {label} must contain an object: {path}")
    if component.get("schema") != schema:
        raise RuntimeError(f"{name} {label} schema must be {schema}")
    if component.get("name") != name:
        raise RuntimeError(f"{name} {label} name does not match its filename")
    return component


def _resolve_sample_set(name: str, *, ancestors: tuple[str, ...] = ()) -> dict[str, object]:
    """Resolve sample-set inheritance while rejecting cycles."""
    if name not in _SAMPLE_SET_TASKS:
        raise RuntimeError(f"unsupported VLM evaluation sample set: {name}")
    sample_set = _load_component(
        _PROFILE_ROOT / "sample_sets", name, _SAMPLE_SET_SCHEMA, "sample set"
    )
    if "extends" not in sample_set:
        return sample_set
    base_name = sample_set.get("extends")
    if (
        not isinstance(base_name, str)
        or base_name not in _SAMPLE_SET_TASKS
        or base_name == name
        or base_name in ancestors
    ):
        raise RuntimeError(f"{name} sample set extends an unsupported base sample set")
    base = _resolve_sample_set(base_name, ancestors=(*ancestors, name))
    overrides = {key: value for key, value in sample_set.items() if key != "extends"}
    if isinstance(base.get("tasks"), dict) and isinstance(overrides.get("tasks"), dict):
        tasks = dict(cast("dict[str, object]", base["tasks"]))
        override_tasks = cast("dict[str, object]", overrides["tasks"])
        for task, entry in override_tasks.items():
            base_entry = tasks.get(task)
            tasks[task] = (
                {**base_entry, **entry}
                if isinstance(base_entry, dict) and isinstance(entry, dict)
                else entry
            )
        overrides["tasks"] = tasks
    return {**base, **overrides}


def _validate_manifest(name: str, manifest: dict[str, object]) -> None:
    if manifest.get("schema") != _PROFILE_SCHEMA:
        raise RuntimeError(f"{name} profile schema must be {_PROFILE_SCHEMA}")
    if manifest.get("name") != name:
        raise RuntimeError(f"{name} profile name does not match its filename")
    sample_set_name, backend_name, evaluator_name = _PROFILE_COMPONENTS[name]
    if manifest.get("sample_set") != sample_set_name:
        raise RuntimeError(f"{name} profile sample set differs from its composition")
    if manifest.get("backend_profile") != backend_name:
        raise RuntimeError(f"{name} profile backend differs from its composition")
    if manifest.get("evaluator_profile") != evaluator_name:
        raise RuntimeError(f"{name} profile evaluator differs from its composition")
    if manifest.get("lmms_eval_revision") != _EVALUATOR_REVISIONS[evaluator_name]:
        raise RuntimeError(f"{name} profile lmms_eval_revision differs from the evaluator pin")
    if manifest.get("model_family") != {
        "architecture": "Qwen3_5ForConditionalGeneration",
        "model_type": "qwen3_5",
    }:
        raise RuntimeError(f"{name} profile model family is unsupported")
    expected_model = _SAMPLE_SET_MODELS.get(sample_set_name)
    if expected_model is not None and manifest.get("model") != expected_model:
        raise RuntimeError(f"{name} profile model pin differs from the runtime policy")
    if manifest.get("backend") != _BACKEND_SETTINGS[backend_name]:
        raise RuntimeError(f"{name} profile backend differs from the runtime policy")
    if manifest.get("preprocessing") != {
        "fps": 2,
        "max_frames": 32,
        "video_reader": "decord",
    }:
        raise RuntimeError(f"{name} profile preprocessing differs from the runtime policy")
    if manifest.get("generation") != {"do_sample": False, "temperature": 0}:
        raise RuntimeError(f"{name} profile generation differs from the runtime policy")
    if (
        manifest.get("seed") != 42
        or manifest.get("repetitions") != 1
        or manifest.get("batch_size") != 1
    ):
        raise RuntimeError(f"{name} profile execution identity differs from the runtime policy")
    selection = manifest.get("selection")
    if selection != _SAMPLE_SET_SELECTIONS[sample_set_name]:
        raise RuntimeError(f"{name} profile selection differs from its versioned policy")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict) or not tasks:
        raise RuntimeError(f"{name} profile tasks must contain an object")
    if tuple(tasks) != _SAMPLE_SET_TASKS[sample_set_name]:
        raise RuntimeError(f"{name} profile tasks differ from its versioned policy")
    for task, entry in tasks.items():
        _validate_task(sample_set_name, task, entry, selection=cast("str", selection))
    _validate_sampling(sample_set_name, manifest, tasks)


def _validate_task(name: str, task: object, entry: object, *, selection: str) -> None:
    if not isinstance(task, str) or task not in profile.VLM_BENCHMARK_DATASETS:
        raise RuntimeError(f"{name} profile contains an unsupported task: {task}")
    if not isinstance(entry, dict):
        raise RuntimeError(f"{name} profile task must contain an object: {task}")
    dataset = profile.VLM_BENCHMARK_DATASETS[task]
    expected = {
        "dataset_repository": dataset.repository,
        "dataset_revision": dataset.revision,
        "max_new_tokens": dataset.max_new_tokens,
        "scoring_task_config": dataset.task_config,
    }
    observed = {key: entry.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError(f"{name} profile task pins differ from the runtime catalog: {task}")
    expected_population = _SAMPLE_SET_POPULATIONS.get(name, {}).get(task)
    if expected_population is not None and entry.get("population_rows") != expected_population:
        raise RuntimeError(
            f"{name} profile task population differs from its versioned policy: {task}"
        )
    if expected_population is not None and task == "mvbench":
        if entry.get("leaf_populations") != _MVBENCH_LEAF_POPULATIONS:
            raise RuntimeError(
                f"{name} profile task leaf populations differ from its versioned policy: {task}"
            )
    rows = entry.get("rows")
    if selection == "all" and rows is not None:
        raise RuntimeError(f"{name} full-data task must not contain exact rows: {task}")
    if selection == "exact-rows" and (not isinstance(rows, list) or not rows):
        raise RuntimeError(f"{name} exact-row task must contain rows: {task}")


def _validate_sampling(
    name: str,
    manifest: dict[str, object],
    tasks: dict[str, object],
) -> None:
    sampling = manifest.get("sampling")
    if name in _AUDITED_SAMPLE_SETS and not isinstance(sampling, dict):
        raise RuntimeError(f"{name} profile must contain a sampling audit")
    if sampling is None:
        return
    if not isinstance(sampling, dict):
        raise RuntimeError(f"{name} profile sampling audit must contain an object")
    if sampling != {
        "schema": _SAMPLING_AUDIT_SCHEMA,
        "claim_scope": "deterministic-screening-only",
        "generator": _SAMPLING_GENERATOR,
    }:
        raise RuntimeError(f"{name} profile sampling audit policy is unsupported")
    for task, entry in tasks.items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"{name} profile task must contain an object: {task}")
        _validate_task_sampling(name, task, entry)


def _validate_task_sampling(name: str, task: str, entry: dict[str, object]) -> None:
    selection = entry.get("selection")
    rows = entry.get("rows")
    if not isinstance(selection, dict) or not isinstance(rows, list):
        raise RuntimeError(f"{name} profile task must contain an audited selection: {task}")
    strata = selection.get("strata")
    if (
        selection.get("method") != "systematic-midpoint"
        or selection.get("stratified_by") != _SAMPLING_STRATA.get(task)
        or selection.get("index_space") != "within-stratum"
        or not isinstance(strata, list)
        or not strata
        or selection.get("selected_rows") != len(rows)
        or selection.get("population_rows")
        != sum(stratum.get("population_rows", 0) for stratum in strata if isinstance(stratum, dict))
        or len(strata) != sum(isinstance(stratum, dict) for stratum in strata)
    ):
        raise RuntimeError(f"{name} profile task sampling audit is invalid: {task}")
    expected_counts: dict[str, tuple[int, int]] = {}
    for stratum in strata:
        stratum_name = stratum.get("name")
        population_rows = stratum.get("population_rows")
        selected_rows = stratum.get("selected_rows")
        if (
            not isinstance(stratum_name, str)
            or not stratum_name
            or not isinstance(population_rows, int)
            or isinstance(population_rows, bool)
            or not isinstance(selected_rows, int)
            or isinstance(selected_rows, bool)
            or selected_rows < 0
            or population_rows < selected_rows
            or stratum_name in expected_counts
        ):
            raise RuntimeError(f"{name} profile task sampling stratum is invalid: {task}")
        expected_counts[stratum_name] = (population_rows, selected_rows)

    actual_indices: dict[str, list[int]] = {stratum: [] for stratum in expected_counts}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"{name} profile task sampling row is invalid: {task}")
        stratum, local_index = _sampling_identity(task, row)
        if stratum not in actual_indices:
            raise RuntimeError(f"{name} profile task sampling stratum differs: {task}")
        actual_indices[stratum].append(local_index)
    for stratum, (population_rows, selected_rows) in expected_counts.items():
        expected = [
            ((2 * index + 1) * population_rows) // (2 * selected_rows)
            for index in range(selected_rows)
        ]
        if actual_indices[stratum] != expected:
            raise RuntimeError(f"{name} profile task rows differ from its sampling audit: {task}")

    identity = hashlib.sha256(
        json.dumps(rows, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if selection.get("selected_row_identities_sha256") != identity:
        raise RuntimeError(f"{name} profile task row identity fingerprint differs: {task}")
    ordered_indices = sorted(index for indices in actual_indices.values() for index in indices)
    quantiles = {
        "method": "lower-order-statistic",
        **{
            f"p{percentile}": ordered_indices[(len(ordered_indices) - 1) * percentile // 100]
            for percentile in (0, 25, 50, 75, 100)
        },
    }
    if selection.get("selected_index_quantiles") != quantiles:
        raise RuntimeError(f"{name} profile task index quantiles differ: {task}")


def _sampling_identity(task: str, row: dict[str, object]) -> tuple[str, int]:
    source_id = row.get("source_sample_id")
    source_index = row.get("source_row_index")
    if not isinstance(source_id, str) or not isinstance(source_index, int):
        raise RuntimeError(f"profile sampling row identity is invalid: {task}")
    if task == "realworldqa":
        return "test", source_index
    if task == "mmmu_val":
        prefix, separator, item = source_id.rpartition("_")
        if not separator or not item.isdigit() or not prefix.startswith("validation_"):
            raise RuntimeError("profile sampling row identity is invalid: mmmu_val")
        return prefix.removeprefix("validation_"), int(item) - 1
    if task in {"mvbench", "video_mmmu"}:
        leaf = row.get("leaf_task")
        if not isinstance(leaf, str) or not leaf.startswith(f"{task}_"):
            raise RuntimeError(f"profile sampling row identity is invalid: {task}")
        return leaf.removeprefix(f"{task}_"), source_index
    if task in {
        "videomme",
        "longvideobench_val_v",
        "mlvu_dev",
        "perceptiontest_val_mc",
    }:
        stratum = row.get("sampling_stratum")
        stratum_index = row.get("source_stratum_index")
        if (
            not isinstance(stratum, str)
            or not stratum
            or not isinstance(stratum_index, int)
            or isinstance(stratum_index, bool)
            or stratum_index < 0
        ):
            raise RuntimeError(f"profile sampling row identity is invalid: {task}")
        return stratum, stratum_index
    raise RuntimeError(f"profile sampling is unsupported for task: {task}")
