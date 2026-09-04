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

"""Tests for composed VLM profiles and sample-set contracts."""

import json
import shutil

import pytest

from examples.puzzletron.evaluation import checkpoint
from examples.puzzletron.evaluation.vlm import contracts, profile, suites
from tests.unit.torch.puzzletron.evaluation.vlm._test_utils import _quick_manifest


def test_versioned_profile_contracts_pin_backends_and_fingerprints():
    profiles = {name: contracts.load_profile(name) for name in contracts.PROFILE_NAMES}
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
