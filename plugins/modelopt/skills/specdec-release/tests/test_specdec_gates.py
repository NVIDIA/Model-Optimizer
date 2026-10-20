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

"""Unit tests for the spec-dec readiness gates.

These are deterministic — no GPU, cluster, or network. Inputs carry the values of a real
Qwen3-8B DFlash2 run (2000 steps on GB300), trimmed to the fields the gates read. Run with:

    python -m pytest "$SKILL_DIR/tests/test_specdec_gates.py"
"""

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gate_smoke import evaluate_smoke
from gate_smoke import main as smoke_main
from gate_train import evaluate_training
from gate_train import main as train_main

TARGET = "/hf-local/Qwen/Qwen3-8B"
DRAFT = "/cicd/m0-drafts/qwen3-8b-dflash2/exported-checkpoint-2000"
SERVED_DRAFT = "/scratchspace/dflash2_bs16/exported-checkpoint-final"
WEIGHTS = [("model.safetensors", 1 << 20)]


def _regression():
    """check_regression.py --json output of the training run."""
    return {
        "pass": True,
        "failure_class": None,
        "metrics": {"loss": 6.5775, "accuracy": 0.1767},
        "thresholds": {"MAX_FINAL_LOSS": 7.0, "MIN_FINAL_ACC": 0.15},
        "messages": ["PASS: loss=6.578", "PASS: accuracy=0.177"],
    }


def _export_config():
    """The exported DFlash2 drafter's config.json, fields the gate reads."""
    return {
        "architectures": ["DFlash2DraftModel"],
        "block_size": 16,
        "hidden_size": 4096,
        "vocab_size": 151936,
        "num_target_layers": 36,
        "dflash_config": {
            "mask_token_id": 151669,
            "target_layer_ids": [1, 9, 17, 25, 33],
            "projector_type": "dflash2",
            "block_size": 16,
        },
    }


def _target_config():
    """Qwen/Qwen3-8B config.json, fields the gate reads."""
    return {
        "hidden_size": 4096,
        "vocab_size": 151936,
        "num_hidden_layers": 36,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
    }


def _smoke():
    """vllm_smoke_test.sh SMOKE_JSON of the same drafter."""
    return {
        "stage": "completed",
        "pass": True,
        "reason": None,
        "target": TARGET,
        "draft": SERVED_DRAFT,
        "num_speculative_tokens": 7,
        "prompts_passed": 8,
        "prompts_failed": 0,
        "mean_acceptance_length": 1.79,
        "min_acceptance_length": 1.5,
    }


def _with(obj, path, value):
    """A copy of obj with one nested key replaced (value None deletes it)."""
    obj = copy.deepcopy(obj)
    *parents, leaf = path
    node = obj
    for key in parents:
        node = node[key]
    if value is None:
        del node[leaf]
    else:
        node[leaf] = value
    return obj


# ── gate_train ────────────────────────────────────────────────────────


def _train(algorithm="dflash2", weights=WEIGHTS, **overrides):
    inputs = {
        "regression": _regression(),
        "export_config": _export_config(),
        "target_config": _target_config(),
    }
    inputs.update(overrides)
    return evaluate_training(
        inputs["regression"], inputs["export_config"], weights, inputs["target_config"], algorithm
    )


def test_train_real_run_passes():
    r = _train()
    assert r["pass"], r["detail"]
    assert r["recipe_thresholds"] == {"MAX_FINAL_LOSS": 7.0, "MIN_FINAL_ACC": 0.15}


@pytest.mark.parametrize(
    ("overrides", "failure_class"),
    [
        ({"regression": None}, "METRICS_MISSING"),
        ({"regression": _with(_regression(), ["thresholds"], {})}, "METRICS_MISSING"),
        ({"regression": ["not", "an", "object"]}, "METRICS_MISSING"),
        ({"regression": _with(_regression(), ["pass"], False)}, "TRAIN_REGRESSION"),
        ({"regression": _with(_regression(), ["pass"], "false")}, "TRAIN_REGRESSION"),
        ({"regression": _with(_regression(), ["metrics", "accuracy"], None)}, "METRICS_MISSING"),
        ({"export_config": None}, "EXPORT_MISSING"),
        ({"weights": [("model.safetensors", 0)]}, "EXPORT_MISSING"),
        ({"algorithm": "dflash"}, "DRAFT_CONFIG_MISMATCH"),  # DFlash2 export, DFlash requested
        (
            {
                "algorithm": "dflash",  # a DSpark export shares DFlash's architecture
                "export_config": _with(
                    _with(_export_config(), ["architectures"], ["DFlashDraftModel"]),
                    ["dflash_config", "projector_type"],
                    "dspark",
                ),
            },
            "DRAFT_CONFIG_MISMATCH",
        ),
        ({"algorithm": "medusa"}, "USER_CONFIG_ERROR"),
        (
            {"export_config": _with(_export_config(), ["dflash_config", "mask_token_id"], 151645)},
            "DRAFT_CONFIG_MISMATCH",  # the target's eos
        ),
        (
            {
                "export_config": _with(
                    _export_config(), ["dflash_config", "target_layer_ids"], [1, 36]
                )
            },
            "DRAFT_CONFIG_MISMATCH",
        ),
        (
            {"export_config": _with(_export_config(), ["dflash_config", "target_layer_ids"], 9)},
            "DRAFT_CONFIG_MISMATCH",
        ),
        (
            {"export_config": _with(_export_config(), ["dflash_config", "projector_type"], None)},
            "DRAFT_CONFIG_MISMATCH",  # a DFlash-architecture export can't prove it is DFlash2
        ),
        (
            {"export_config": _with(_export_config(), ["hidden_size"], 2048)},
            "DRAFT_CONFIG_MISMATCH",
        ),
        (
            {"export_config": _with(_export_config(), ["dflash_config", "block_size"], 8)},
            "DRAFT_CONFIG_MISMATCH",
        ),
    ],
)
def test_train_failures(overrides, failure_class):
    r = _train(**overrides)
    assert not r["pass"]
    assert r["failure_class"] == failure_class, r["detail"]


def test_train_generic_eagle3_llama_pair_passes():
    """A family with different config fields passes on what both configs carry."""
    target = {
        "hidden_size": 4096,
        "vocab_size": 128256,
        "num_hidden_layers": 32,
        "eos_token_id": [128001, 128009],
    }
    export = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "hidden_size": 4096,
        "vocab_size": 128256,
        "draft_vocab_size": 32000,
        "eagle_config": {"eagle_aux_hidden_state_layer_ids": [1, 15, 28]},
    }
    r = evaluate_training(_regression(), export, WEIGHTS, target, "eagle3")
    assert r["pass"], r["detail"]
    assert any("mask_token_id not checked" in n for n in r["notes"])


def test_train_inherited_qwen_mask_token_on_smaller_vocab_fails():
    """A DSpark export keeping the Qwen3 mask id on a 128K-vocab target trains on garbage."""
    target = {"hidden_size": 4096, "vocab_size": 128256, "num_hidden_layers": 32}
    export = _with(_export_config(), ["architectures"], ["DFlashDraftModel"])
    export.update(vocab_size=128256, num_target_layers=32)
    export["dflash_config"].update(projector_type="dspark", target_layer_ids=[1, 8, 15, 22, 29])
    r = evaluate_training(_regression(), export, WEIGHTS, target, "dspark")
    assert r["failure_class"] == "DRAFT_CONFIG_MISMATCH"
    assert "mask_token_id 151669 >= target vocab_size 128256" in r["detail"]


# ── gate_smoke ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("change", "kwargs", "failure_class"),
    [
        ({}, {}, None),
        ({"pass": False, "prompts_passed": 7, "prompts_failed": 1}, {}, "CHECKPOINT_NOT_SERVABLE"),
        ({"prompts_passed": 0}, {}, "CHECKPOINT_NOT_SERVABLE"),
        (
            {"stage": "server_died", "pass": False, "mean_acceptance_length": None},
            {},
            "CHECKPOINT_NOT_SERVABLE",
        ),
        (
            {"pass": False, "reason": "a failure the gate does not know"},
            {},
            "CHECKPOINT_NOT_SERVABLE",
        ),
        ({"stage": None}, {}, "USER_CONFIG_ERROR"),  # deleted below: predates "stage"
        ({"mean_acceptance_length": 1.0}, {}, "SPEC_NOT_ACTIVE"),
        ({"mean_acceptance_length": None}, {}, "SPEC_NOT_ACTIVE"),
        ({"pass": False, "mean_acceptance_length": 1.4}, {}, "ACCEPTANCE_REGRESSION"),
        ({"mean_acceptance_length": 1.5}, {}, None),  # the script's floor is inclusive too
        ({}, {"draft": SERVED_DRAFT, "target": TARGET}, None),
        ({}, {"draft": DRAFT}, "SPEC_CONFIG_MISMATCH"),  # a host path is not what was served
    ],
)
def test_smoke(change, kwargs, failure_class):
    smoke = {**_smoke(), **change}
    if smoke["stage"] is None:
        del smoke["stage"]
    r = evaluate_smoke(smoke, **kwargs)
    assert r["failure_class"] == failure_class, r["detail"]
    assert r["pass"] == (failure_class is None)


# ── CLIs ──────────────────────────────────────────────────────────────


def _write(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


def test_cli_exit_codes(tmp_path, capsys):
    export = tmp_path / "export"
    export.mkdir()
    _write(export / "config.json", _export_config())
    (export / "model.safetensors").write_bytes(b"\0" * 8)
    common = [
        "--regression", _write(tmp_path / "regression_check.json", _regression()),
        "--export", str(export),
        "--target-config", _write(tmp_path / "target.json", _target_config()),
    ]  # fmt: skip
    assert train_main([*common, "--algorithm", "dflash2"]) == 0
    assert train_main([*common, "--algorithm", "dspark"]) == 1  # projector_type mismatch
    assert train_main([*common, "--algorithm", "medusa"]) == 2
    bad_target = tmp_path / "bad_target.json"
    bad_target.write_text("{")
    capsys.readouterr()
    assert train_main([*common, "--target-config", str(bad_target), "--algorithm", "dflash2"]) == 2
    assert json.loads(capsys.readouterr().out)["inputs"]["target_config"] == str(bad_target)
    (export / "config.json").write_text("{")  # truncated mid-write: a broken export
    assert train_main([*common, "--algorithm", "dflash2"]) == 1
    _write(export / "config.json", [])  # not an object: also a broken export
    assert train_main([*common, "--algorithm", "dflash2"]) == 1
    assert (
        train_main([*common, "--target-config", _write(bad_target, []), "--algorithm", "dflash2"])
        == 2
    )
    # A missing directory is a mistyped path; a missing file in an existing one is a failed run.
    _write(export / "config.json", _export_config())
    nope = str(tmp_path / "nope" / "exported-checkpoint-final")
    assert train_main([*common, "--export", nope, "--algorithm", "dflash2"]) == 2
    crashed = str(tmp_path / "exported-checkpoint-final")  # training died before the export
    assert train_main([*common, "--export", crashed, "--algorithm", "dflash2"]) == 1
    missing = str(tmp_path / "nope" / "regression_check.json")
    assert train_main([*common, "--regression", missing, "--algorithm", "dflash2"]) == 2
    missing = str(tmp_path / "regression_missing.json")
    assert train_main([*common, "--regression", missing, "--algorithm", "dflash2"]) == 1

    for not_an_object in ([], "stage", None):
        bad = _write(tmp_path / "bad.json", not_an_object)
        assert smoke_main(["--smoke", bad]) == 2
