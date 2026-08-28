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

"""Tests for the vLLM skip-softmax calibration driver."""

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[3] / "examples" / "vllm_serve"


@pytest.fixture
def calibration_driver(monkeypatch):
    monkeypatch.syspath_prepend(str(_EXAMPLES_DIR))
    return importlib.import_module("calibrate_sparse_attn")


@pytest.mark.parametrize(
    "flags",
    [
        ["--target_sparse_ratio", "-0.1"],
        ["--target_sparse_ratio", "1.1"],
        ["--target_sparse_ratio", "nan"],
        ["--decode_tokens", "-1"],
        ["--engine_kwargs", "[]"],
        ["--engine_kwargs", "not-json"],
        ["--engine_kwargs", '{"model": "/other"}'],
        ["--engine_kwargs", '{"worker_cls": "other.Worker"}'],
        ["--engine_kwargs", '{"enforce_eager": false}'],
        ["--engine_kwargs", '{"enable_prefix_caching": true}'],
        ["--engine_kwargs", '{"pipeline_parallel_size": 2}'],
    ],
)
def test_parser_rejects_invalid_inputs_before_engine_start(calibration_driver, flags):
    with pytest.raises(SystemExit):
        calibration_driver._build_parser().parse_args(["/checkpoint", *flags])


def test_parser_accepts_safe_engine_kwargs(calibration_driver):
    args = calibration_driver._build_parser().parse_args(
        ["/checkpoint", "--engine_kwargs", '{"enable_expert_parallel": true}']
    )
    assert args.engine_kwargs == {"enable_expert_parallel": True}


def test_load_prompts_reads_nonempty_lines(calibration_driver, tmp_path):
    prompts_file = tmp_path / "prompts.txt"
    prompts_file.write_text(" first prompt\n\nsecond prompt \n")
    args = SimpleNamespace(prompts_file=str(prompts_file))

    assert calibration_driver._load_prompts(None, args) == ["first prompt", "second prompt"]


def test_existing_sparse_config_reads_only_dict(calibration_driver, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"sparse_attention_config": {"config_groups": {}}}))
    assert calibration_driver._existing_sparse_config(str(checkpoint)) == {"config_groups": {}}

    config_path.write_text(json.dumps({"sparse_attention_config": ["invalid"]}))
    assert calibration_driver._existing_sparse_config(str(checkpoint)) is None


@pytest.mark.parametrize("update_checkpoint", [False, True])
def test_write_config_emits_artifact_and_optionally_updates_checkpoint(
    calibration_driver, tmp_path, monkeypatch, update_checkpoint
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps({"model_type": "test"}))
    sparse_config = {"config_groups": {"group_0": {"algorithm": "skip_softmax"}}}
    monkeypatch.chdir(tmp_path)

    calibration_driver._write_config(str(checkpoint), sparse_config, update_checkpoint)

    assert json.loads((tmp_path / "sparse_attention_config.json").read_text()) == sparse_config
    checkpoint_config = json.loads(config_path.read_text())
    assert checkpoint_config["model_type"] == "test"
    if update_checkpoint:
        assert checkpoint_config["sparse_attention_config"] == sparse_config
    else:
        assert "sparse_attention_config" not in checkpoint_config
